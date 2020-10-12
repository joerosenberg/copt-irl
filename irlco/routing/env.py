from typing import Tuple, List, Callable
import torch
from gym import Env
from torch import Tensor
from random import randint
import copt

RoutingStateBatch = Tuple[Tensor, Tensor]

DEVICE = torch.device('cuda')


class BatchCircuitRoutingEnv(Env):
    def __init__(self, batch_size: int, min_instance_size: int, max_instance_size: int):
        self.batch_size = batch_size
        self.min_instance_size = min_instance_size
        self.max_instance_size = max_instance_size

        self.state_batch = None
        self.reset()

    def reset(self, instances=None) -> RoutingStateBatch:
        if instances is None:
            instance_size = randint(self.min_instance_size, self.max_instance_size)
            self.state_batch = (generate_valid_instances(instance_size, self.batch_size),
                                torch.zeros(self.batch_size, 0, device=DEVICE, dtype=torch.long))

        else:
            assert instances.shape[1] == self.batch_size
            instance_size = instances.shape[0]
            self.state_batch = (instances, torch.zeros(self.batch_size, 0, device=DEVICE, dtype=torch.long))

        return self.state_batch

    def step(self, actions: Tensor) -> Tuple[RoutingStateBatch, bool]:
        base_pairs_batch, prev_actions_batch = self.state_batch

        # Check if actions are valid
        if torch.eq(actions, prev_actions_batch).any():
            print(actions)
            print(prev_actions_batch)
            raise Exception
        elif torch.lt(actions, 0).any() or torch.ge(actions, base_pairs_batch.shape[1]).any():
            raise Exception
        else:
            prev_actions_batch = torch.cat((prev_actions_batch, actions), 1)

        # Update stored state
        self.state_batch = (base_pairs_batch, prev_actions_batch)

        # Check if state is terminal
        terminal = (base_pairs_batch.shape[1] == prev_actions_batch.shape[1])

        return self.state_batch, terminal


def measures_to_terminal_rewards(episode_length: int, measures: Tensor, successes=None) -> Tensor:
    """
    Prototype mapping from measures to terminal rewards.
    Args:
        episode_length:
        measures:
        successes:

    Returns:

    """
    if successes is None:
        return 2.0 - measures / (1000 * episode_length)
    else:
        # Give terminal reward of -2 to unsuccessful connections
        return (2.0 - measures / (1000 * episode_length)).masked_fill(torch.logical_not(successes), -5.0)


def generate_valid_instances(instance_size: int, batch_size: int):
    """
    Generates a list of valid problems (i.e. neighbouring points are >=30 units away from eachother.)
    Args:
        batch_size: Number of problems to generate

    Returns:

    """
    base_pairs = torch.zeros(instance_size, batch_size, 4, device=DEVICE)
    for i in range(batch_size):
        while True:
            instance = torch.FloatTensor(copt.getProblem(instance_size))
            # Compute (euclidean) distances between neighbouring points
            sq = torch.square(instance[1:] - instance[:-1])
            start_dists_sq = sq[:, 0] + sq[:, 1]
            end_dists_sq = sq[:, 2] + sq[:, 3]
            if torch.ge(start_dists_sq, 30**2).all() and torch.ge(end_dists_sq, 30**2).all():
                base_pairs[:, i, :] = instance
                break
    return base_pairs
