from typing import Callable, Tuple, List
from torch import Tensor
import torch
import numpy as np
from gym import Env

SortingState = Tuple[Tensor, List[int]]


class SortingEnv(Env):
    # Environment where the goal is to sequentially sort a sequence of numbers into ascending order.
    def __init__(self, reward_model: Callable[[SortingState], float], sequence_sampler: Callable[[], SortingState]):
        self.reward_model = reward_model
        self.sequence_sampler = sequence_sampler

        self.state = None
        self.reset()

    def reset(self) -> SortingState:
        self.state = self.sequence_sampler()
        return self.state

    def step(self, action: int) -> Tuple[SortingState, float, dict]:
        unsorted_seq, action_seq = self.state

        if action in action_seq:
            raise InvalidActionError
        else:
            new_action_seq = action_seq.append(action)

        next_state = (unsorted_seq, new_action_seq)
        reward = self.reward_model(next_state)

        # Check if next_state is terminal
        terminal = (len(unsorted_seq) == len(new_action_seq))

        self.state = next_state
        return next_state, reward, {'terminal': terminal}

    def render(self, mode='human'):
        pass


class InvalidActionError(Exception):
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
    # Returns 1.0 for a correctly sorted sequence, -1.0 for an incorrectly sorted sequence, and 0 for all non-terminal
    # states.
    unsorted_seq, action_seq = state
    if len(unsorted_seq) > len(action_seq):
        return 0.0
    if len(unsorted_seq) == len(action_seq):
        if [unsorted_seq[i] for i in action_seq] == sorted(unsorted_seq):
            return 1.0
        else:
            return -1.0
    else:
        raise InvalidStateError


def pairwise_sorted_reward(state: SortingState) -> float:
    pass


def uniform_sequence_sampler(size_low: int, size_high: int, number_low: float, number_high: float) -> Callable[[], SortingState]:
    def f() -> SortingState:
        length = np.random.randint(size_low, size_high)
        unsorted_seq = number_low + (number_high - number_low) * torch.rand(length)
        return unsorted_seq, []

    return f

