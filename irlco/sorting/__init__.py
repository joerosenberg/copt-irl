from typing import Tuple, List, Callable

import numpy as np
import torch
from gym import Env
from torch import Tensor

SortingState = Tuple[Tensor, List[int]]


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