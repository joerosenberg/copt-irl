import torch
import numpy as np
import matplotlib.pyplot as plt

import irlco.sorting
import irlco.pointer_transformer as pt

# Constants for dimensions of transformer embedding dim. and number of heads for multihead attention
EMBEDDING_DIM = 16
NB_HEADS = 4
assert EMBEDDING_DIM % NB_HEADS == 0, "Embedding dim must be divisible by the number of heads"

# Initialise sequence sampler and reward model
sequence_sampler = irlco.sorting.uniform_sampler_generator(3, 8, 0.0, 10.0)
reward_model = irlco.sorting.stepwise_reward

# Initialise environment that samples initial states using sequence_sampler and returns reward from states
# according to reward_model
env = irlco.sorting.SortingEnv(reward_model, sequence_sampler)

# Initialise pointer transformer policy network
policy = pt.PointerTransformer(d_model=EMBEDDING_DIM, nhead=NB_HEADS, num_encoder_layers=3, num_decoder_layers=3,
                               dim_feedforward=64)

# Number of episodes to train for
nb_episodes = 20_000

# Gradient optimizer
optimizer = torch.optim.SGD(policy.parameters(), lr=0.001, momentum=0.9)


# Mask generator (code taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    """
    Generates a mask that prevents actions attending to subsequent actions in the transformer decoder.
    Args:
        size: Size of the mask (i.e. length of the solution so far.)

    Returns:
        Mask for transformer decoder.

    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_sorted_element_mask(previous_actions, input_length: int) -> torch.Tensor:
    """
    Generates a mask that prevents actions from attending to elements of the unordered set that have already been
    placed into the ordered sequence.
    Args:
        previous_actions: List of previous actions (in order) that we need to mask
        input_length: Number of elements in the unordered sequence

    Returns:
        Memory mask of shape (nb of previous actions + 1, input sequence length) suitable for use in transformer

    """
    # Generate lower triangular matrix (creates columns for masked input elements)
    # i_th column of masked_cols is equal to the {a_i}'th column of the mask
    masked_cols = torch.tril(torch.ones(len(previous_actions) + 1, len(previous_actions)) * float('-inf'), diagonal=-1)
    # Create empty mask
    mask = torch.zeros(len(previous_actions) + 1, input_length)
    # For each previous action, prevent further actions from attending to its corresponding input element
    mask[:, previous_actions] = masked_cols
    return mask


# Input embedding: just use a random linear map R -> R^{EMBEDDING_DIM}
embedding_weights = torch.rand(EMBEDDING_DIM) - 0.5


def sorting_input_embedding(input: torch.Tensor) -> torch.Tensor:
    """
    Maps the sequence into a higher-dimensional vector space. The mapping is performed entrywise on the sequence.
    Args:
        input: The sequence we want to send to the higher-dimensional space. Dimension: (N).

    Returns:
        The image of the sequence in the higher-dimensional space. Dimension: (N, EMBEDDING_DIM).
    """
    return torch.einsum('i,j->ij', input, embedding_weights)

# State & action representation:
# ([x_1, ..., x_n], [a_1, ..., a_m]) where m <= n
# [x_1, ..., x_n] is the unordered sequence (we want to learn how to sort this!)
# a_1, ..., a_m describes our partial solution up to the m_th step - a_i is the index of the entry in the unordered
# sequence which should be at position i in the ordered sequence

epoch_returns = np.zeros(nb_episodes)

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
        # Append 0 to start of sequence (represents context node)
        partial_ordered_seq = torch.tensor([0.0] + [unordered_seq[action] for action in actions])

        # Apply embedding to inputs
        embedded_unordered_seq = sorting_input_embedding(unordered_seq)
        embedded_partial_ordered_seq = sorting_input_embedding(partial_ordered_seq)

        # Reshape inputs to add extra dimension for batch indexing - this is to match the shape of the transformer input
        embedded_unordered_seq = embedded_unordered_seq.unsqueeze(1)
        embedded_partial_ordered_seq = embedded_partial_ordered_seq.unsqueeze(1)

        # Generate masks:
        tgt_mask = generate_square_subsequent_mask(len(actions) + 1) # Stops actions attending to future actions
        memory_mask = generate_sorted_element_mask(actions, len(unordered_seq)) # Stops actions attending to elements
        # that have already been sorted

        # Get action probabilities from the policy
        action_probs = policy.forward(embedded_unordered_seq, embedded_partial_ordered_seq, tgt_mask=tgt_mask,
                                      memory_mask=memory_mask)[0]

        # Sample action according to action_probs
        action = np.random.choice(action_probs.numel(), p=action_probs.detach().numpy().flatten())

        # Take step in environment
        next_state, reward, _ = env.step(action)

        # Store the action energy for the selected action
        episode_energies[timestep] = torch.log(action_probs[action])
        # Store the reward for the current timestep
        episode_rewards[timestep] = reward

        # Update the state
        state = next_state

    # Calculate episode returns (no time decay)
    # the conjugation by torch.flip is because we want to cumulatively sum episode_rewards backwards.
    episode_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards, [0]), 0), [0])

    # Store entire episode return for validation purposes
    epoch_returns[episode] = episode_returns[0]

    # Calculate policy loss and gradients
    loss = torch.dot(episode_energies, episode_returns)
    loss.backward()

    # Take gradient step
    optimizer.step()

    # Print progress once every 100 episodes.
    if (episode + 1) % 100 == 0:
        print("Completed episode {}/{}.".format(episode+1, nb_episodes))

plt.plot(np.cumsum(epoch_returns))
plt.savefig('sorting_epoch_returns.png')




