from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
from gym import Env
from torch import Tensor

SortingState = Tuple[Tensor, List[int]]
SortingStateBatch = Tuple[Tensor, Tensor]

# Change this to 'cpu' if GPU is unavailable
DEVICE = torch.device('cuda')


class BatchSortingEnv(Env):
    """
    Version of SortingEnv that operates on multiple sequences of the same length simultaneously. Allows more efficient
    calculation in the policy network (as we can produce action probabilities for many states simultaneously), as well
    as more stable training (as we can aggregate gradients from many episodes).

    Throughout we call B the batch size and S the sequence length. S can vary based on the batch returned by
    batch_sequence_sampler.
    """
    def __init__(self, batch_reward_model: Callable[[SortingStateBatch], Tensor],
                 batch_sequence_sampler: Callable[[], SortingStateBatch]):
        self.batch_reward_model = batch_reward_model # Returns (B, 1) tensor of rewards
        self.batch_sequence_sampler = batch_sequence_sampler # Returns batch of initial states - the unsorted seqs
        # are a (B, S) tensor of ints

        self.state_batch = None
        self.reset()

    def reset(self) -> SortingStateBatch:
        self.state_batch = self.batch_sequence_sampler()
        return self.state_batch

    def step(self, actions: Tensor) -> Tuple[SortingStateBatch, Tensor, bool]:
        """
        Takes a step in all episodes in the batch.

        Args:
            actions: (B, 1) tensor of ints corresponding to the next elements that we want to add to each sequence in
                the batch.

        Returns: Batch of states (pair of a (B, S) tensor of unordered seqs and a (B, n) tensor of actions taken, n<=S),
            a (B, 1) tensor of rewards for transitioning into those states, and a bool representing whether or not
            the episodes in the batch have ended.
        """
        # Unpack states into unsorted sequences and previous actions
        unsorted_seq_batch, prev_actions_batch = self.state_batch

        # Check if actions are valid
        # This check might not work - it's supposed to broadcast along the second dimension (so actions are only checked
        # against actions in their own episode) and then return true if the actions match for any episode.
        if (actions == prev_actions_batch).any():
            # TODO: Extend ActionAlreadyChosen so the message tells you which action is invalid.
            raise ActionAlreadyChosenError
        elif (actions < 0).any() or (actions >= unsorted_seq_batch.shape[1]).any():
            raise ActionOutOfBoundsError
        else:
            # Append actions to state descriptions
            prev_actions_batch = torch.cat((prev_actions_batch, actions), 1)

        # Update state by
        self.state_batch = (unsorted_seq_batch, prev_actions_batch)

        # Calculate reward for transitioning into the new state.
        rewards = self.batch_reward_model(self.state_batch)

        # Check if the new states are terminal.
        # (This should occur for all episodes at the same time. Uniform episode length is guaranteed by the fact that
        # we store initial states as rows in a (B, S) tensor - each episode will have length S).
        terminal = (unsorted_seq_batch.shape[1] == prev_actions_batch.shape[1])

        return self.state_batch, rewards, terminal

    def render(self, mode='human'):
        pass


def generate_batch_uniform_sampler(size_low: int, size_high: int, entry_low: float, entry_high: float, batch_size: int) -> Callable[[], SortingStateBatch]:
    assert size_low <= size_high, "Upper sequence length bound must be greater than lower sequence length bound."
    assert entry_low < entry_high, "Upper entry bound must be greater than lower entry bound."

    # Create initial state sampling function
    def batch_sampler() -> SortingStateBatch:
        # Sample sequence length from {size_low, ..., size_high}
        length = np.random.randint(size_low, size_high + 1)
        # Sample each entry from [number_low, number_high]
        unsorted_seq_batch = entry_low + (entry_high - entry_low) * torch.rand((batch_size, length), device=DEVICE)
        initial_actions = torch.zeros((batch_size, 0), dtype=torch.int64, device=DEVICE)
        # Return initial state; action_seq is empty because we haven't sorted anything yet.
        return unsorted_seq_batch, initial_actions

    return batch_sampler


def batch_sorted_terminal_reward(state: SortingStateBatch) -> Tensor:
    unsorted_seq_batch, action_seq_batch = state

    # Get dimension parameters
    batch_size = unsorted_seq_batch.shape[0]
    episode_length = unsorted_seq_batch.shape[1]
    nb_actions_taken = action_seq_batch.shape[1]

    if nb_actions_taken < episode_length:
        # If we aren't at the end of the episodes, give 0 reward everywhere
        return torch.zeros((batch_size, 1), device=DEVICE)
    else:
        # If we're at the end, find the episodes where the unsorted sequences have been correctly sorted and give a
        # reward of 1 for these. Give a reward of -1 everywhere else.
        # Get correctly sorted sequences:
        answer_batch, _ = torch.sort(unsorted_seq_batch, dim=1)
        # Get the proposed sorted sequences:
        proposal_batch = torch.gather(unsorted_seq_batch, 1, action_seq_batch)
        # Check equality of each element
        element_matches = torch.eq(answer_batch, proposal_batch)
        # Reduce by taking AND along each episode (prod works like 'and' for boolean tensors)
        episodes_correct = torch.prod(element_matches, 1)
        # Obtain rewards by reshaping to shape (B, 1) and replacing 'True' values with 1, "False" values with -1.
        # TODO: use the proper way to do this, i.e. not using arithmetic operations.
        return (episodes_correct.int().unsqueeze(1) * 2) - 1


def batch_stepwise_reward(state_batch: SortingStateBatch) -> float:
    """
    Non-sparse reward for the sorting environment.

    Args:
        state: Calculate the reward for transitioning into this state.

    Returns:
        +1.0 if the sequence has been correctly sorted so far, and -1.0 if it is incorrectly sorted.
    """
    unsorted_seq_batch, action_seq_batch = state_batch

    nb_actions_taken = action_seq_batch.shape[1]

    # Get the partially sorted sequence from the indices
    partial_proposal_batch = torch.gather(unsorted_seq_batch, 1, action_seq_batch)
    partial_answer_batch = torch.sort(unsorted_seq_batch, dim=1)[0][:, :nb_actions_taken]

    element_matches = torch.eq(partial_answer_batch, partial_proposal_batch)
    episodes_correct = torch.prod(element_matches, 1)
    reward_batch = (episodes_correct.int().unsqueeze(1) * 2) - 1

    return reward_batch

class SortingEnv(Env):
    """
    Gym environment where the goal is to sort a set of numbers into ascending order.

    At the start of each episode, the agent is presented with a set of numbers.
    At each step, it must choose the next number to append to the end of the sorted sequence. We interpret this
    choice as an action in an MDP.
    The episode ends once the agent has arranged all of the numbers into a sequence.

    States are represented by pairs (unsorted_seq, action_seq):
        - unsorted_seq is a list of numbers that we wish to sort. This stays the same throughout the episode. We
          do **not** remove numbers from this once they have been selected.
        - action_seq represents the ordering that the agent has created so far. It is a list of indices for
          unsorted_seq.

    We deliberately do not specify a reward function or initial state sampling distribution for this environment.
    These are instead supplied via reward_model and sequence_sampler arguments.

    Attributes:
        reward_model: Function that returns a reward for transitioning into a given state.
        sequence_sampler: Function that returns a random initial state for a new episode.
        state: Current state of the environment.
    """

    def __init__(self, reward_model: Callable[[SortingState], float], sequence_sampler: Callable[[], SortingState]):
        """
        Creates a new sorting environment with given reward model and initial state distribution.

        Args:
            reward_model: Function that returns a reward for transitioning into a given state.
            sequence_sampler: Function that returns an initial state.
        """
        self.reward_model = reward_model
        self.sequence_sampler = sequence_sampler

        self.state = None
        # Set the initial state
        self.reset()

    def reset(self) -> SortingState:
        """
        Ends the current episode and start a new one.

        Returns:
            The initial state for the new episode.
        """
        self.state = self.sequence_sampler()
        return self.state

    def step(self, action: int) -> Tuple[SortingState, float, bool]:
        """
        Takes a step in the current episode.

        Args:
            action: The index of the next element that we want to add to the sequence.

        Returns:
            The state, the reward for transitioning into this state and a bool indicating if the state is terminal (i.e.
                if the agent has finished sorting the sequence).

        Raises:
            ActionAlreadyChosenError: The chosen element has already been added to the sorted sequence.
            ActionOutOfBoundsError: The action does not correspond to any element of the unsorted sequence.
        """
        unsorted_seq, action_seq = self.state

        if action in action_seq:
            raise ActionAlreadyChosenError
        elif action < 0 or action >= len(self.state[0]):
            raise ActionOutOfBoundsError
        else:
            # If the action is valid, modify the action_seq in self.state by appending the action to it
            # (Note that this modifies self.state in place).
            action_seq.append(action)

        # Calculate reward for transitioning into the new state.
        reward = self.reward_model(self.state)

        # Check if the new state is terminal.
        terminal = (len(unsorted_seq) == len(action_seq))

        return self.state, reward, terminal

    def render(self, mode='human'):
        pass


class ActionOutOfBoundsError(Exception):
    pass


class ActionAlreadyChosenError(Exception):
    pass


class InvalidStateError(Exception):
    pass


def _is_terminal(state: SortingState) -> bool:
    # Checks if a sorting state is terminal.
    unsorted_seq, action_seq = state

    if len(unsorted_seq) > len(action_seq):
        return False
    if len(unsorted_seq) == len(action_seq):
        return True
    else:
        raise InvalidStateError


def _is_correctly_sorted(state: SortingState) -> bool:
    pass


def sorted_terminal_reward(state: SortingState) -> float:
    """
    Terminal (sparse) reward for the sorting environment.

    Args:
        state: Calculate the reward for transitioning into this state.

    Returns:
        1.0 for a correctly sorted complete sequence, -1.0 for an incorrectly sorted complete sequence, and 0 for all
            non-terminal states.
    """
    unsorted_seq, action_seq = state

    if len(unsorted_seq) > len(action_seq):
        # Return 0 if the state is not terminal.
        return 0
    if len(unsorted_seq) == len(action_seq):
        if [unsorted_seq[i] for i in action_seq] == sorted(unsorted_seq):
            # If the state is terminal and the sequence has been correctly sorted, return 1.
            return 1
        else:
            # If the state is terminal and the sequence is incorrectly sorted, return -1.
            return -1
    else:
        raise InvalidStateError


def stepwise_reward(state: SortingState) -> float:
    """
    Non-sparse reward for the sorting environment.

    Args:
        state: Calculate the reward for transitioning into this state.

    Returns:
        +1.0 if the sequence has been correctly sorted so far, and -1.0 if it is incorrectly sorted.
    """
    unsorted_seq, action_seq = state

    # Get the partially sorted sequence from the indices
    partial_seq = [unsorted_seq[i] for i in action_seq]

    # Check if the partially sorted sequence matches the correct answer so far.
    if partial_seq == sorted(unsorted_seq)[0: len(partial_seq)]:
        return 1
    else:
        return -1


def uniform_sampler_generator(size_low: int, size_high: int, number_low: float, number_high: float) -> Callable[[], SortingState]:
    """
    Creates an initial state sampler that uniformly samples unsorted sequences of numbers within given bounds.
    The size of the unsorted sequence is also uniformly randomly sampled, i.e.
    sequence_length ~ U{size_low, .., size_high}, entries ~ i.i.d. U[number_low, number_high].

    Args:
        size_low: Lower bound for the length of the sampled unsorted sequences.
        size_high: Upper bound for the length of the sampled unsorted sequences.
        number_low: Lower bound for the sampled sequence entries.
        number_high: Upper bound for the sampled sequence entries.

    Returns:
        A function that returns initial states with sequence lengths and entries sampled according to the given bounds.
    """
    assert size_low <= size_high, "Upper sequence length bound must be greater than lower sequence length bound."
    assert number_low < number_high, "Upper entry bound must be greater than lower entry bound."

    # Create initial state sampling function
    def sampler() -> SortingState:
        # Sample sequence length from {size_low, ..., size_high}
        length = np.random.randint(size_low, size_high + 1)
        # Sample each entry from [number_low, number_high]
        unsorted_seq = number_low + (number_high - number_low) * torch.rand(length)
        # Return initial state; action_seq is empty because we haven't sorted anything yet.
        return unsorted_seq, []

    return sampler


sampler = uniform_sampler_generator(3, 6, 0.0, 1.0)
sorting_env = SortingEnv(sorted_terminal_reward, sampler)