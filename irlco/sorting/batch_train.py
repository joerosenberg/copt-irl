import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import csv

import irlco.sorting
import irlco.pointer_transformer as pt


# Check if CUDA device is available - use this if possible, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Open file for logging
LOG_PATH = "../../logs/sorting.csv"
log_file = open(LOG_PATH, 'w', newline='\n')
csv_writer = csv.writer(log_file)

# Constants:
# Dimensions of transformer embedding dim. and number of heads for multihead attention
EMBEDDING_DIM = 128
NB_HEADS = 8
assert EMBEDDING_DIM % NB_HEADS == 0, "Embedding dim must be divisible by the number of heads"
# Min and max sequence length
MIN_SEQ_LENGTH = 3
MAX_SEQ_LENGTH = 8
# Lower and upper bound on sequence entries
MIN_ENTRY = 0.0
MAX_ENTRY = 10.0
# Batch size
BATCH_SIZE = 2048
# Epsilon
EPS = 1e-8

# Optimiser parameters
LR = 1e-3


# Initialise sequence sampler and reward model
sequence_sampler = irlco.sorting.generate_batch_uniform_sampler(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH,
                                                                MIN_ENTRY, MAX_ENTRY, BATCH_SIZE)
reward_model = irlco.sorting.batch_sorted_terminal_reward
#reward_model = irlco.sorting.batch_stepwise_reward

# Initialise environment that samples initial states using sequence_sampler and returns reward from states
# according to reward_model
env = irlco.sorting.BatchSortingEnv(reward_model, sequence_sampler)

# Initialise pointer transformer for the policy network
policy = pt.PointerTransformer(d_model=EMBEDDING_DIM, nhead=NB_HEADS, num_encoder_layers=3, num_decoder_layers=3,
                               dim_feedforward=64).to(device=device)

# Number of batches to train for:
nb_batches = 20_000

# Initialise optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)


# Mask generator (code taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
def generate_square_subsequent_mask(size: int) -> Tensor:
    """
    Generates a mask that prevents actions attending to subsequent actions in the transformer decoder.
    Args:
        size: Size of the mask (i.e. length of the solution so far.)

    Returns:
        Mask for transformer decoder.

    """
    mask = torch.eq(torch.triu(torch.ones(size, size, device=device)), 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_sorted_element_mask(previous_actions, input_length: int) -> Tensor:
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
    masked_cols = torch.tril(torch.ones(len(previous_actions) + 1, len(previous_actions), device=device) * float('-inf'), diagonal=-1)
    # Create empty mask
    mask = torch.zeros(len(previous_actions) + 1, input_length, device=device)
    # For each previous action, prevent further actions from attending to its corresponding input element
    mask[:, previous_actions] = masked_cols
    return mask


def generate_batch_of_sorted_element_masks(prev_actions_batch: Tensor, input_sequence_length: int, nb_heads: int) -> Tensor:
    """

    Args:
        prev_actions_batch: Batch of previous actions as a tensor of shape (B, nb actions taken so far)
        input_sequence_length: Batch of initial states (unordered sequences) of shape (B, episode length).

    Returns: Tensor of shape (batch_size, decoder input length, encoder input length) that is meant to
    be used as a memory mask for the transformer.

    Note that decoder input length = nb. of previous actions + 1.

    """
    nb_actions_taken = prev_actions_batch.shape[1]
    batch_size = prev_actions_batch.shape[0]
    # Get mask columns
    mask_cols = torch.tril(torch.ones(nb_actions_taken + 1, nb_actions_taken, device=device) * float('-inf'), diagonal=-1)
    # Create empty mask
    mask = torch.zeros((batch_size, nb_actions_taken + 1, input_sequence_length), device=device)
    # Flatten mask and input actions so we can calculate
    # For each previous action, prevent further actions from attending to its corresponding input element

    # Unvectorised prototype:
    # TODO: Replace this with faster vectorised version using torch tensor operations
    for i in range(batch_size):
        mask[i, :, prev_actions_batch[i, :]] = mask_cols

    # Need to repeat each 2D mask nb_heads times - repeats like abcd -> aaabbbcccddd, since we want to use the same mask
    # across all heads for each sequence
    mask = torch.repeat_interleave(mask, nb_heads, dim=0)
    return mask


# Input embedding: just use a random linear map R -> R^{EMBEDDING_DIM}
embedding_weights = torch.rand(EMBEDDING_DIM, device=device) - 0.5


def sorting_input_embedding(input: Tensor) -> Tensor:
    """
    Maps the sequence into a higher-dimensional vector space. The mapping is performed entrywise on the sequence.

    Args:
        input: The batch of sequences we want to send to the higher-dimensional space. Shape: (batch_size, seq_length).

    Returns:
        The image of the sequence in the higher-dimensional space. Has shape (seq_length, batch_size, embedding_dim).
    """
    # This einsum both computes the embeddings and reshapes the output to match the format required by the transformer.
    return torch.einsum('ij,k->jik', input, embedding_weights)


# State & action representation:
# ([x_1, ..., x_n], [a_1, ..., a_m]) where m <= n
# [x_1, ..., x_n] is the unordered sequence (we want to learn how to sort this!)
# a_1, ..., a_m describes our partial solution up to the m_th step - a_i is the index of the entry in the unordered
# sequence which should be at position i in the ordered sequence

epoch_returns = torch.zeros(BATCH_SIZE, nb_batches, device=device)

for b in range(nb_batches):
    # Reset the episode
    state_batch = env.reset()

    episode_length = state_batch[0].shape[1]

    episode_energies = torch.zeros((BATCH_SIZE, episode_length), device=device)
    episode_rewards = torch.zeros((BATCH_SIZE, episode_length), device=device)

    for timestep in range(episode_length):
        # Unpack the state into the unordered sequence and the sequence of actions taken
        unordered_seq_batch, prev_actions_batch = state_batch

        # Calculate decoder input by fetching the entries in the unordered sequence that are indexed by the action ids
        # (i.e. from a sequence of indices, get the corresponding items)
        # Append 0 to start of sequence (represents context node)
        proposal_batch = torch.gather(unordered_seq_batch, 1, prev_actions_batch)
        unembedded_decoder_input = torch.cat((torch.zeros(BATCH_SIZE, 1, device=device), proposal_batch), dim=1)

        # Apply embedding to inputs
        encoder_input = sorting_input_embedding(unordered_seq_batch)
        decoder_input = sorting_input_embedding(unembedded_decoder_input)

        # Generate masks:
        tgt_mask = generate_square_subsequent_mask(timestep + 1) # Stops actions attending to future actions
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions_batch, episode_length, NB_HEADS)  # Stops actions
        # attending to elements that have already been sorted

        # Get action probabilities from the policy
        action_batch_probs = policy.forward(encoder_input, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)

        # Sample actions according to action_batch_probs
        action_batch = torch.multinomial(action_batch_probs, 1)

        # Take step in environment
        next_state_batch, reward_batch, _ = env.step(action_batch)

        # Store the action energies for the selected actions
        # Unvectorised prototype:
        for i in range(BATCH_SIZE):
            episode_energies[i, timestep] = torch.log(action_batch_probs[i, action_batch[i, 0]] + EPS)

        # TODO: Fix this vectorised version of the above.
        # episode_energies[:, timestep] = torch.log(action_batch_probs[:, action_batch[:, 0]] + EPS)

        # Store the rewards for the current timestep
        episode_rewards[:, timestep] = reward_batch.squeeze()

        # Update the state
        state_batch = next_state_batch

    # Calculate episode returns (no time decay)
    # the conjugation by torch.flip is because we want to cumulatively sum episode_rewards backwards.
    episode_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards, [1]), 1), [1])

    # Store entire episode return for validation purposes
    epoch_returns[:, b] = episode_returns[:, 0].clone()

    # Log average entire episode returns
    print([b, torch.mean(epoch_returns[:, b]).tolist()])
    csv_writer.writerow([b, torch.mean(epoch_returns[:, b]).tolist()])

    # Calculate policy loss and gradients
    # Minus sign is because J = sum(episode_energies * episode_returns) is the 'average return' performance measure
    # that we're trying to maximise, so we take its negative to get a loss.
    loss = -torch.sum(episode_energies * episode_returns)
    loss.backward()

    # Take gradient step
    optimizer.step()

    # Print progress & write log to disk once every 100 batches.
    if (b + 1) % 100 == 0:
        print(f"Completed episode {b+1}/{nb_batches}.")
        print(f"Average return was {torch.mean(epoch_returns[:, b])}.")
        log_file.flush()

plt.plot(np.cumsum(epoch_returns))
plt.savefig('sorting_epoch_returns.png')




