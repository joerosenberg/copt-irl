import copt
import torch
from irlco.masking import generate_square_subsequent_mask, generate_batch_of_sorted_element_masks
from irlco.routing.env import BatchCircuitRoutingEnv
from multiprocessing import Pool

from irlco.routing.mp_evaluation import mp_evaluate


def greedy_decode(env, initial_states, policy_net, device=torch.device('cuda')):
    episode_length = initial_states[0].shape[0]
    batch_size = initial_states[0].shape[1]
    nb_heads = policy_net.nhead

    trajectory_probs = torch.zeros((batch_size, episode_length), device=device)

    states = env.reset(instances=initial_states[0])

    for t in range(episode_length):
        base_pairs, prev_actions = states
        decoder_input = states_to_action_decoder_input(states, device=device)

        tgt_mask = generate_square_subsequent_mask(t + 1)
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, nb_heads)
        action_probs = policy_net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]
        actions = torch.argmax(action_probs, dim=1).unsqueeze(1)

        states, _ = env.step(actions)

        trajectory_probs[:, t] = action_probs[torch.arange(0, batch_size), actions.squeeze()]

    measures, successes = evaluate_terminal_states(states, device=device)
    _, actions = states
    return actions, trajectory_probs, measures, successes


def beam_search_decode(env, initial_states, policy_net, beam_width, device=torch.device('cuda')):
    """

    Args:
        env:
        initial_states:
        policy_net:
        k: beam width
        device:

    Returns:

    """
    episode_length = initial_states[0].shape[0]
    batch_size = initial_states[0].shape[1]
    nb_heads = policy_net.nhead

    # Create tensor to store actions as we decode - we store each of the {beam_width} most probable action sequences
    # at each step
    action_sequences = torch.zeros((episode_length, batch_size, beam_width), device=device, dtype=torch.long)

    base_pairs, _ = initial_states
    rep_base_pairs = torch.repeat_interleave(base_pairs, beam_width, dim=1)
    dummy_env = BatchCircuitRoutingEnv(batch_size * beam_width, 1, 100)
    states = dummy_env.reset(instances=rep_base_pairs)

    log_trajectory_probs = torch.zeros((batch_size, beam_width), device=device)

    for t in range(episode_length):
        base_pairs, prev_actions = states
        decoder_input = states_to_action_decoder_input(states, device=device)
        tgt_mask = generate_square_subsequent_mask(t + 1)
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, nb_heads)

        # Calculate next-step action probabilities from current best action sequences for each instance
        next_action_probs = policy_net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)

        # (Episode length - t) is equal to the number of remaining actions, so we shrink the number of actions to expand
        # on to this if it is smaller than the beam_width argument
        k = min(beam_width, episode_length - t)
        # Create tensor to store log joint probabilities of trajectories after we expand:
        # First index corresponds to problem instance, second index corresponds to trajectories we selected in the
        # previous step, third index corresponds to the top k actions for this step
        new_log_joint_trajectory_probs = log_trajectory_probs.unsqueeze(2).repeat(1, 1, k)

        # Get top k most probable next actions for each
        # next_action_probs has shape (episode_length, batch_size * beam_width, 1)
        # topk_next_action_probs and topk_next_actions have shape (k, batch_size * beam_width, 1)
        topk_next_action_probs, topk_next_actions = torch.topk(next_action_probs, k, dim=0)

        # Reshape action probs to calculate log-probs of each of the trajectories we just expanded on
        # new_log_joint_trajectory_probs has shape (batch_size, beam_width, k)
        new_log_joint_trajectory_probs += torch.log(topk_next_action_probs.squeeze(2).T.reshape(batch_size, beam_width, k))

        # reshape again to find {beam_width} most probable trajectories for each input
        log_trajectory_probs, best_trajectory_idx = torch.topk(
            new_log_joint_trajectory_probs.reshape(batch_size, beam_width * k), beam_width, dim=0)


def beam_search_decode2(initial_states, policy_net, beam_width, device=torch.device('cuda')):
    # Less efficient beam search decode - iterates over each entry and decodes separately.
    episode_length = initial_states[0].shape[0]
    batch_size = initial_states[0].shape[1]
    nb_heads = policy_net.nhead

    batch_action_sequences = torch.zeros((batch_size, episode_length), dtype=torch.long, device=device)
    batch_trajectory_probs = torch.zeros_like(batch_action_sequences, dtype=torch.float)

    for b in range(batch_size):
        base_pair = initial_states[0][:, b, :]
        encoder_input = base_pair.unsqueeze(1).repeat(1, beam_width, 1)
        best_action_sequences = torch.zeros((beam_width, 0), dtype=torch.long, device=device)
        log_joint_trajectory_probs = torch.zeros(beam_width, device=device)
        for t in range(episode_length):
            decoder_input = states_to_action_decoder_input((encoder_input, best_action_sequences), device=device)
            tgt_mask = generate_square_subsequent_mask(t+1)
            memory_masks = generate_batch_of_sorted_element_masks(best_action_sequences, episode_length, nb_heads)
            next_action_probs = policy_net(encoder_input, decoder_input,
                                           tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]

            k = min(beam_width, episode_length - t)
            # Get k next most probable actions + their probabilities
            expanded_action_probs, expanded_actions = torch.topk(next_action_probs.unsqueeze(2), k, dim=0)
            # Add the k most probable actions onto the existing trajectories to get beam_size * k possible trajectories
            expanded_trajectories = torch.cat((torch.repeat_interleave(best_action_sequences, k, dim=0), expanded_actions.flatten()), dim=1)
            # Calculate log-probabilities of the expanded trajectories
            log_expanded_trajectory_probs = torch.repeat_interleave(log_joint_trajectory_probs, k, dim=0) \
                                            + torch.log(expanded_action_probs)
            # Select beam_width most probable trajectories
            log_joint_trajectory_probs, best_trajectory_idx = torch.topk(log_expanded_trajectory_probs, beam_width)
            # Update chosen action sequences
            best_action_sequences = expanded_trajectories[best_trajectory_idx].copy()
        # Choose action sequence with largest probability
        batch_action_sequences[b, :] = best_action_sequences[torch.argmax(log_joint_trajectory_probs), :].copy()

    measures, successes = evaluate_terminal_states((initial_states[0], batch_action_sequences))

    return batch_action_sequences, batch_trajectory_probs, measures, successes


def sample_best_of_n_trajectories(env, initial_states, policy_net, n_sample, device=torch.device('cuda')):
    assert initial_states[1].shape[1] == 0, "The provided states are not initial states!"

    episode_length = initial_states[0].shape[0]
    batch_size = initial_states[0].shape[1]
    nb_heads = policy_net.nhead

    trajectory_probs = torch.zeros((batch_size, episode_length), device=device)

    best_trajectory_probs = torch.zeros_like(trajectory_probs)
    best_actions = torch.zeros_like(trajectory_probs, dtype=torch.long)
    best_measures = torch.zeros((batch_size, 1), device=device) * float('inf')
    best_successes = torch.zeros_like(best_measures, dtype=torch.bool)

    # Sample n_sample trajectories from each initial state according to the current policy, returning only the best
    for n in range(n_sample):
        states = env.reset(instances=initial_states[0])
        for t in range(episode_length):
            base_pairs, prev_actions = states
            decoder_input = states_to_action_decoder_input(states, device=device)

            # Create masks so that the decoder elements don't attend to future actions or base pairs that have
            # already been connected
            tgt_mask = generate_square_subsequent_mask(t + 1)
            memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, nb_heads)

            action_probs = policy_net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]
            actions = torch.multinomial(action_probs, 1)

            states, _ = env.step(actions)

            # Store probabilities for the chosen actions
            trajectory_probs[:, t] = action_probs[torch.arange(0, batch_size), actions.squeeze()]

        # Evaluate the trajectories + add them to the best so far if they were successful and had lower measure
        # than the previous best trajectories
        _, actions = states
        measures, successes = evaluate_terminal_states(states)
        # We consider the trajectory to be an improvement if one of the following holds:
        # 1. The current best trajectory is unsuccessful.
        # 2. The current best trajectory and the new trajectory are successful, and the new trajectory
        #    has a lower measure.
        is_improvement = torch.logical_or(
            torch.logical_not(best_successes),
            torch.logical_and(torch.lt(measures, best_measures),
                              torch.logical_and(best_successes, successes))
        )
        best_trajectory_probs = torch.where(is_improvement, trajectory_probs, best_trajectory_probs)
        best_actions = torch.where(is_improvement, actions, best_actions)
        best_measures = torch.where(is_improvement, measures, best_measures)
        best_successes = torch.where(is_improvement, successes, best_successes)

    return best_actions, best_trajectory_probs, best_measures, best_successes


def states_to_action_decoder_input(states, device=torch.device('cuda')):
    base_pairs, prev_actions = states
    batch_size = base_pairs.shape[1]
    t = prev_actions.shape[1]

    decoder_input = torch.zeros(t + 1, batch_size, 4, device=device)
    indices = prev_actions.T.unsqueeze(2).repeat(1, 1, 4)
    decoder_input[1:, :, :] = torch.gather(base_pairs, 0, indices)
    return decoder_input


def evaluate_terminal_states2(terminal_states, device=torch.device('cuda')):
    base_pairs, actions = terminal_states
    batch_size = base_pairs.shape[1]

    measures = torch.zeros((batch_size, 1))
    successes = torch.zeros((batch_size, 1), dtype=torch.bool)

    for i in range(batch_size):
        problem = [tuple(base_pair) for base_pair in base_pairs[:, i, :].tolist()]
        ordering = actions[i, :].tolist()
        evaluation = copt.evaluate(problem, ordering)
        measures[i, 0] = evaluation['measure']
        successes[i, 0] = evaluation['success']

    return measures.to(device), successes.to(device)


def evaluate_terminal_states(terminal_states, device=torch.device('cuda')):
    base_pairs, actions = terminal_states
    batch_size = base_pairs.shape[1]

    measures = torch.zeros((batch_size, 1))
    successes = torch.zeros((batch_size, 1), dtype=torch.bool)

    problems = [[tuple(base_pair) for base_pair in base_pairs[:, i, :].tolist()] for i in range(batch_size)]
    orderings = [actions[i, :].tolist() for i in range(batch_size)]

    for i, (measure, success) in enumerate(mp_evaluate(problems, orderings)):
        measures[i, 0] = measure
        successes[i, 0] = success

    return measures.to(device), successes.to(device)




def trajectory_action_probabilities(terminal_states, policy_net, device=torch.device('cuda')):
    base_pairs, actions = terminal_states
    decoder_input = states_to_action_decoder_input((base_pairs, actions[:, :-1]), device=device)

    episode_length = base_pairs.shape[0]
    batch_size = base_pairs.shape[1]
    nb_heads = policy_net.nhead

    tgt_mask = generate_square_subsequent_mask(episode_length)
    memory_masks = generate_batch_of_sorted_element_masks(actions[:, :-1], episode_length, nb_heads)

    action_probs = policy_net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)
    trajectory_probs = torch.gather(action_probs, 2, actions.unsqueeze(2)).transpose(0, 1)

    return trajectory_probs
