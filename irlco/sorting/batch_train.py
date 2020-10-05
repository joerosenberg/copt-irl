import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import csv
import wandb
import irlco.sorting
import irlco.pointer_transformer as pt
from irlco.masking import generate_batch_of_sorted_element_masks, generate_square_subsequent_mask

# Check if CUDA device is available - use this if possible, otherwise use CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constants:
# Dimensions of transformer embedding dim. and number of heads for multihead attention
EMBEDDING_DIM = 16
NB_HEADS = 4
assert EMBEDDING_DIM % NB_HEADS == 0, "Embedding dim must be divisible by the number of heads"
# Min and max sequence length
MIN_SEQ_LENGTH = 3
MAX_SEQ_LENGTH = 8
# Lower and upper bound on sequence entries
MIN_ENTRY = 0.0
MAX_ENTRY = 10.0
# Batch size
BATCH_SIZE = 2048

NB_ENCODER_LAYERS = 6
NB_DECODER_LAYERS = 6

DROPOUT = 0.1
DIM_FEEDFORWARD = 64


# Epsilon
EPS = 1e-8

# Optimiser parameters
LR = 1e-4

# Initialise wandb logging
wandb.init(project='sorting', config={
    "embedding_dimension": EMBEDDING_DIM,
    "nb_heads": NB_HEADS,
    "min_seq_length": MIN_SEQ_LENGTH,
    "max_seq_length": MAX_SEQ_LENGTH,
    "min_seq_entry": MIN_ENTRY,
    "max_seq_entry": MAX_ENTRY,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "nb_encoder_layers": NB_ENCODER_LAYERS,
    "nb_decoder_layers": NB_DECODER_LAYERS,
    "dropout": DROPOUT,
    "dim_feedforward": DIM_FEEDFORWARD
})


# Initialise sequence sampler and reward model
sequence_sampler = irlco.sorting.generate_batch_uniform_sampler(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH,
                                                                MIN_ENTRY, MAX_ENTRY, BATCH_SIZE)
reward_model = irlco.sorting.batch_sorted_terminal_reward
# reward_model = irlco.sorting.batch_stepwise_reward

# Initialise environment that samples initial states using sequence_sampler and returns reward from states
# according to reward_model
env = irlco.sorting.BatchSortingEnv(reward_model, sequence_sampler)

# Initialise pointer transformer for the policy network
policy = pt.TwinDecoderPointerTransformer(d_model=EMBEDDING_DIM, nhead=NB_HEADS, num_encoder_layers=NB_ENCODER_LAYERS,
                                          num_decoder_layers=NB_DECODER_LAYERS, d_input=1, dropout=DROPOUT,
                                          dim_feedforward=DIM_FEEDFORWARD).to(device=device)

# Watch model
wandb.watch(policy)

# Number of batches to train for:
nb_batches = 20_000

# Initialise optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

# State & action representation:
# ([x_1, ..., x_n], [a_1, ..., a_m]) where m <= n
# [x_1, ..., x_n] is the unordered sequence (we want to learn how to sort this!)
# a_1, ..., a_m describes our partial solution up to the m_th step - a_i is the index of the entry in the unordered
# sequence which should be at position i in the ordered sequence


epoch_returns = torch.zeros(BATCH_SIZE, nb_batches, device=device)

for b in range(nb_batches):
    # Reset the episode
    state_batch = env.reset()

    # Get episode length so we know how many timesteps we need to iterate for
    episode_length = state_batch[0].shape[1]

    # Create empty tensors for storing action energies (log-probabilities) and rewards - used for model update
    episode_energies = torch.zeros((BATCH_SIZE, episode_length), device=device)
    episode_rewards = torch.zeros((BATCH_SIZE, episode_length), device=device)

    for timestep in range(episode_length):
        # Unpack the state into the unordered sequence and the sequence of actions taken
        unordered_seq_batch, prev_actions_batch = state_batch

        # Calculate decoder input by fetching the entries in the unordered sequence that are indexed by the action ids
        # (i.e. from a sequence of indices, get the corresponding items)
        # Append 0 to start of sequence (represents context node)
        proposal_batch = torch.gather(unordered_seq_batch, 1, prev_actions_batch)
        decoder_input = torch.cat((torch.zeros(BATCH_SIZE, 1, device=device), proposal_batch), dim=1)
        decoder_input = decoder_input.T
        decoder_input = decoder_input.unsqueeze(2)

        encoder_input = unordered_seq_batch.T
        encoder_input = encoder_input.unsqueeze(2)

        # Generate masks:
        tgt_mask = generate_square_subsequent_mask(timestep + 1) # Stops actions attending to future actions
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions_batch, episode_length, NB_HEADS)  # Stops actions
        # attending to elements that have already been sorted

        # Get action probabilities from the policy
        action_batch_probs = policy.forward(encoder_input, decoder_input, tgt_mask=tgt_mask,
                                            memory_mask=memory_masks)

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
    wandb.log({'batch': b, 'average_return': torch.mean(epoch_returns[:, b]).tolist()})

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

plt.plot(np.cumsum(epoch_returns))
plt.savefig('sorting_epoch_returns.png')




