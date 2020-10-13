import torch
from irlco.masking import generate_square_subsequent_mask


def states_to_reward_decoder_input(states, device=torch.device('cuda')):
    base_pairs, actions = states

    indices = actions.T.unsqueeze(2).repeat(1, 1, 4)
    decoder_input = torch.gather(base_pairs, 0, indices)
    return decoder_input


def compute_shaping_terms(terminal_states, reward_net, device=torch.device('cuda')):
    decoder_input = states_to_reward_decoder_input(terminal_states, device=device)
    base_pairs, _ = terminal_states
    episode_length = base_pairs.shape[0]
    tgt_mask = generate_square_subsequent_mask(episode_length)
    return reward_net.shaping_terms(base_pairs, decoder_input, tgt_mask=tgt_mask)


def shaping_terms_to_rewards(shaping_terms, terminal_rewards, device=torch.device('cuda')):
    episode_length = shaping_terms.shape[0]
    batch_size = shaping_terms.shape[1]

    rewards = torch.zeros((episode_length, batch_size, 1), device=device)
    rewards += shaping_terms # + h(s')
    rewards[1:] -= shaping_terms[:-1] # - h(s)
    rewards[-1, :, :] += terminal_rewards # + T(s')
    return rewards
