import copt
from gym import Env
from typing import Callable, Tuple, List
import numpy as np


class CircuitRoutingState(object):
    pass


class CircuitRoutingState(object):
    def __init__(self, problem_instance: List[Tuple[int, int, int, int]], connection_sequence: List[int] = None):
        # problem_instance: Description of a circuit routing problem instance as an array of vectors in R^4.
        #   Each vector gives the positions of the start and end nodes in a base pair as [x1, y1, x2, y2].
        #   The vectors should be ordered so that the y-coordinate of the start nodes is decreasing - the problems
        #   produced by the copt-irl module do this by default.
        # connection_sequence: Description of a partial solution of the problem instance, given as a sequence of base
        #   pairs to connect. A base pair is specified by its index in the problem_instance tuple.
        #   A full solution is therefore a permutation of {0, 1, ..., len(problem_instance) - 1}.

        if connection_sequence is None:
            connection_sequence = []

        self.problem_instance = problem_instance
        self.connection_sequence = connection_sequence

    def is_terminal(self) -> bool:
        # Returns whether or not this state is terminal (i.e., whether or not all base pairs have been connected).
        return self.get_problem_size() - len(self.connection_sequence) == 0

    def get_problem_size(self) -> int:
        # Returns the number of base pairs that are present in the problem instance.
        return len(self.problem_instance)

    def get_problem_instance(self) -> List[Tuple[int, int, int, int]]:
        return self.problem_instance

    def is_connected(self, base_pair: int) -> bool:
        # Returns whether or not a specified base pair is connected.
        return base_pair in self.connection_sequence

    def get_connected_base_pairs(self) -> List[int]:
        # Get the list of already connected base pairs (in order of connection).
        return self.connection_sequence

    def get_unconnected_base_pairs(self) -> List[int]:
        # Get the list of unconnected base pairs.
        return list(set(range(self.get_problem_size())) - set(self.get_connected_base_pairs()))

    def connect_base_pair(self, base_pair: int) -> CircuitRoutingState:
        # Returns a new CircuitRoutingState that is created by connecting a base pair in this state.

        # Check if this is a valid connection, in terms of the base pair existing and not already being connected.
        # Validity in terms of checking if design rules are broken is not checked here - this is only checked when
        # evaluating a solution with copt-irl.
        if base_pair < 0 or base_pair >= self.get_problem_size():
            raise BasePairOutOfRangeError
        elif self.is_connected(base_pair):
            raise BasePairAlreadyConnectedError

        return CircuitRoutingState(self.problem_instance, self.connection_sequence + [base_pair])

    def evaluate(self) -> dict:
        # Evaluates a terminal state as a solution
        return copt.evaluate(self.problem_instance, self.connection_sequence)


class CircuitRoutingEnv(Env):
    def __init__(self, reward_model: Callable[[CircuitRoutingState], float], min_instance_size: int,
                 max_instance_size: int, use_terminal_reward: bool = True,
                 failed_connection_penalty: float = 1000):
        # reward_model is a function that returns the reward obtained upon entering a state.
        # This allows us to separate the reward from the environment, which is necessary for the IRL setting.
        self.reward_model = reward_model
        self.min_instance_size = min_instance_size
        self.max_instance_size = max_instance_size
        self.use_terminal_reward = use_terminal_reward
        self.failed_connection_penalty = failed_connection_penalty

        self.state = None
        self.reset()

    def step(self, action: int) -> (CircuitRoutingState, float, bool, dict):
        next_state = self.state.connect_base_pair(action)

        # Calculate reward for transitioning into the new state using the reward model
        reward = self.reward_model(next_state)

        # Check if the next state is terminal
        is_terminal = next_state.is_terminal()

        # If the next state is terminal, evaluate the solution and store the results in a dictionary
        # Otherwise, we have no info to return.
        # TODO: Add some way to deal with failed connections. Should ask Zuken if they could expose a partial
        # evaluation method in the copt-irl library, so we can check for failed connections at each step.
        if is_terminal:
            info = next_state.evaluate()
        else:
            info = {}

        # If the next state is terminal and we're using terminal rewards, obtain the terminal reward as the negative
        # of the total path length.
        # TODO: Add reward scaling/clipping for the terminal reward.
        if is_terminal and self.use_terminal_reward:
            reward = reward - info['measure']

        # Update the state.
        self.state = next_state

        return next_state, reward, is_terminal, info

    def reset(self) -> CircuitRoutingState:
        problem_size = np.random.randint(self.min_instance_size, self.max_instance_size)
        problem_instance = copt.getProblem(problem_size)
        self.state = CircuitRoutingState(problem_instance, [])
        return self.state

    def render(self):
        pass

    def seed(self, seed=None):
        pass


class InvalidBasePairError(Exception):
    pass


class BasePairOutOfRangeError(InvalidBasePairError):
    pass


class BasePairAlreadyConnectedError(InvalidBasePairError):
    pass
