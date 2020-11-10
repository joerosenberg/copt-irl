import torch

from irlco.masking import generate_square_subsequent_mask, generate_batch_of_sorted_element_masks
from irlco.routing.env import measures_to_terminal_rewards
from irlco.routing.policy import evaluate_terminal_states, states_to_action_decoder_input


@torch.no_grad()
def greedy_rollout_baselines(base_pairs, action_sequences, env, net, device=torch.device('cuda')):
    """
    Computes the terminal rewards obtained if we had acted greedily (selected the most probable action) from each
    time step instead of taking the given actions.
    Args:
        base_pairs: Tensor of base pairs of shape (instance_size, batch_size, 4).
        action_sequences: Tensor of complete action sequences of shape (batch_size, instance_size).
        env: BatchCircuitRoutingEnv that operates on batches of size batch_size.
        net: The policy network to sample greedy actions from.
        device: torch.device to perform computations and store tensors on.

    Returns: Tensor of terminal rewards obtained by performing greedy rollouts. Has shape (batch_size, instance_size).

    """
    # Given a batch of base pairs and action sequences, compute the terminal reward obtained by decoding greedily
    # from each time step.
    episode_length, batch_size, _ = base_pairs.shape
    baseline_rewards = torch.zeros((batch_size, episode_length), device=device)

    for t in range(episode_length):
        _, greedy_rollout_actions = greedy_rollout(base_pairs, action_sequences[:, :t], env, net, device=device)
        greedy_rollout_measures, greedy_rollout_successes = evaluate_terminal_states(
            (base_pairs, greedy_rollout_actions), device=device)
        baseline_rewards[:, t] = measures_to_terminal_rewards(episode_length, greedy_rollout_measures,
                                                              successes=greedy_rollout_successes).squeeze(1)
    return baseline_rewards


def greedy_rollout(base_pairs, actions_so_far, env, net, device=torch.device('cuda')):
    """
    Computes a greedy rollout from a single time step for a given batch of instances and previously taken actions.
    Args:
        base_pairs: Tensor of base pairs of shape (instance_size, batch_size, 4).
        actions_so_far: Tensor of partial action sequences of shape (batch_size, T).
        env: BatchCircuitRoutingEnv that operates on batches of size batch_size.
        net: The policy network to get greedy actions from.
        device: torch.device to perform computations and store tensors on.

    Returns: The terminal states obtained by acting greedily from time T as a tuple (base_pairs, action_sequences).

    """
    episode_length, batch_size, _ = base_pairs.shape
    T = actions_so_far.shape[1]
    nb_heads = net.nhead

    states = (base_pairs, actions_so_far)
    env.state_batch = states

    for t in range(T, episode_length):
        base_pairs, prev_actions = states
        decoder_input = states_to_action_decoder_input(states, device=device)

        tgt_mask = generate_square_subsequent_mask(t + 1)
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, nb_heads)
        action_probs = net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]
        actions = torch.argmax(action_probs, dim=1).unsqueeze(1)

        states, _ = env.step(actions)

    return states
