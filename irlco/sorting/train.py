import torch
import numpy as np
import irlco.sorting.sorting_env as sorting_env
import irlco.pointer_transformer as pt

# Initialise sequence sampler and reward model
sequence_sampler = sorting_env.uniform_sequence_sampler(3, 8, 0.0, 10.0)
reward_model = sorting_env.sorted_terminal_reward

# Initialise environment that samples initial states using sequence_sampler and returns reward from states
# according to reward_model
env = sorting_env.SortingEnv(reward_model, sequence_sampler)

# Initialise pointer transformer policy network
policy = pt.PointerTransformer(32, 1, 3, 3, 64)

# Number of episodes to train for
nb_episodes = 10_000

# Gradient optimizer
optimizer = torch.optim.SGD(policy.parameters(), lr=0.001, momentum=0.9)

# State & action representation:
# ([x_1, ..., x_n], [a_1, ..., a_m]) where m <= n
# [x_1, ..., x_n] is the unordered sequence (we want to learn how to sort this!)
# a_1, ..., a_m describes our partial solution up to the m_th step - a_i is the index of the entry in the unordered
# sequence which should be at position i in the ordered sequence

for episode in range(nb_episodes):
    # Reset the episode
    state = env.reset()

    episode_length = len(state[0])

    episode_energies = torch.zeros(episode_length)
    episode_rewards = torch.zeros(episode_length)

    for timestep in range(episode_length):
        # Unpack the state into the unordered sequence and the sequence of actions taken
        unordered_seq, actions = state

        # Calculate decoder input by fetching the entries in the unordered sequence that are indexed by the action ids
        # (i.e. from a sequence of indices, get the corresponding items)
        partial_ordered_seq = torch.tensor([unordered_seq[action] for action in actions])

        # Get action probabilities from the policy
        # TODO: add masking for actions to pointer transformer
        print(unordered_seq)
        print(partial_ordered_seq)
        action_probs = policy.forward(unordered_seq, partial_ordered_seq)

        # Sample action according to action_probs
        action = np.random.choice(len(action_probs), p=action_probs)

        # Take step in environment
        next_state, reward, _ = env.step(action)

        # Store the action energy for the selected action
        episode_energies[timestep] = torch.log(action_probs[action])
        # Store the reward for the current timestep
        episode_rewards[timestep] = reward

        # Update the state
        state = next_state

    # Calculate episode returns (no time decay)
    episode_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards, [0]), 0), [0])

    # Calculate policy loss and gradients
    loss = torch.dot(episode_energies, episode_returns)
    loss.backward()

    # Take gradient step
    optimizer.step()





