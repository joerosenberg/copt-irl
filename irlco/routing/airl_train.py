"""
Algorithm that uses shaped reward signal learned using the adversarial IRL method from 'Learning Robust Rewards with
Adversarial Inverse Reinforcement Learning' (Fu, Luo and Levine 2018).
"""
import copt
import torch
import wandb

import irlco.pointer_transformer as pt
from irlco.masking import generate_square_subsequent_mask, generate_batch_of_sorted_element_masks
from irlco.routing.data import CircuitSolutionDataset
from irlco.routing.env import BatchCircuitRoutingEnv, measures_to_terminal_rewards

EMBEDDING_DIM = 64  # Dimension of input embedding
NB_HEADS = 4  # Number of heads in multihead attention modules
assert EMBEDDING_DIM % NB_HEADS == 0
FF_DIM = 256  # Dimension of feedforward layers in transformer
DROPOUT = 0.1
AGENT_BATCH_SIZE = 256  # Number of batches
EXPERT_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
TEST_INTERVAL = 5
NB_EPISODES = 20_000
MIN_INSTANCE_SIZE = 3  # Minimum number of base pairs in training instances
MAX_INSTANCE_SIZE = 5  # Maximum number of base pairs in training instances
LR = 1e-5  # Learning rate
EPS = 1e-8
DEVICE = torch.device('cuda')
EXPERT_DATA_PATH = './data/irl_data_config.yaml'
TEST_DATA_PATH = './data/test_data_config.yaml'
# Set expert data path to test data for quicker load time...
EXPERT_DATA_PATH = TEST_DATA_PATH

# Enable remote logging
wandb.init(project='routing', config={
    'embedding_dimension': EMBEDDING_DIM,
    'nb_heads': NB_HEADS,
    'min_instance_size': MIN_INSTANCE_SIZE,
    'max_instance_size': MAX_INSTANCE_SIZE,
    'batch_size': AGENT_BATCH_SIZE,
    'learning_rate': LR
})

print("Creating environment...")
env = BatchCircuitRoutingEnv(AGENT_BATCH_SIZE, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)  # Circuit routing environment
net = pt.TwinDecoderPointerTransformer(4, EMBEDDING_DIM, NB_HEADS, 3, 3, FF_DIM, DROPOUT).cuda()  # Policy + reward
# network
print("Loading expert data...")
expert_data = CircuitSolutionDataset(EXPERT_DATA_PATH)
print("Loading test data...")
# test_data = CircuitSolutionDataset(TEST_DATA_PATH)
test_data = expert_data
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
wandb.watch(net)

for i in range(NB_EPISODES):
    # Reset the episodes
    states = env.reset()
    episode_length = states[0].shape[0]

    # Create empty tensor for storing the log-probabilities of selected actions (used for gradient update)
    episode_log_probs = torch.zeros((AGENT_BATCH_SIZE, episode_length), device=DEVICE)

    # Collect trajectories by executing current policy:
    for t in range(episode_length):
        base_pairs, prev_actions = states

        # Get decoder input by fetching the base pairs corresponding to the previous actions
        decoder_input = torch.zeros(t + 1, AGENT_BATCH_SIZE, 4, device=DEVICE)
        indices = prev_actions.T.unsqueeze(2).repeat(1, 1, 4)
        decoder_input[1:, :, :] = torch.gather(base_pairs, 0, indices)

        # Generate masks
        tgt_mask = generate_square_subsequent_mask(t + 1)
        memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, NB_HEADS)

        # Get action log-probabilities and shaping terms from rewards
        action_probs = net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]

        # Sample actions
        actions = torch.multinomial(action_probs, 1)

        # Take step in environment & update states
        states, is_terminal = env.step(actions)

        # Store the log-probabilities of the selected actions
        # TODO: Implement this using pytorch tensorops
        for j in range(AGENT_BATCH_SIZE):
            episode_log_probs[j, t] = torch.log(action_probs[j, actions[j, 0]])

    # Unpack terminal states into base pairs and complete sequences of actions
    agent_base_pairs, agent_actions = states

    # Compute measures of the agent's solutions:
    agent_measures = torch.zeros((AGENT_BATCH_SIZE, 1), device=DEVICE)
    for j in range(AGENT_BATCH_SIZE):
        # Convert jth set of base pairs into format required by copt.evaluate (list of 4-tuples)
        problem = [tuple(base_pair) for base_pair in agent_base_pairs[:, j, :].tolist()]
        ordering = agent_actions[j, :].tolist()
        agent_measures[j, 0] = copt.evaluate(problem, ordering)['measure']

    # Map agent's solution measures to terminal rewards
    agent_terminal_rewards = measures_to_terminal_rewards(episode_length, agent_measures)

    # Get batch of expert data (from instances with the same length as the ones the agent just attempted to solve)
    expert_base_pairs, expert_actions, expert_measures = expert_data.get_batch(episode_length, AGENT_BATCH_SIZE, DEVICE)
    # Map expert's solution measures to terminal rewards
    expert_terminal_rewards = measures_to_terminal_rewards(episode_length, expert_measures)

    # Concatenate agent and expert data together (so we can compute in a single batch)
    all_base_pairs = torch.cat((agent_base_pairs, expert_base_pairs), dim=1)
    all_actions = torch.cat((agent_actions, expert_actions.T), dim=0)
    all_terminal_rewards = torch.cat((agent_terminal_rewards, expert_terminal_rewards))
    # Make labels - agent has label 0, expert has label 1
    labels = torch.cat(
        (torch.zeros(AGENT_BATCH_SIZE, 1, device=DEVICE), torch.ones(EXPERT_BATCH_SIZE, 1, device=DEVICE)))

    # Compute masks for probability decoding
    tgt_mask = generate_square_subsequent_mask(episode_length)
    memory_masks = generate_batch_of_sorted_element_masks(all_actions[:, :-1], episode_length, NB_HEADS)
    # Compute action probabilities for all actions
    action_decoder_input = torch.zeros(episode_length, AGENT_BATCH_SIZE + EXPERT_BATCH_SIZE, 4, device=DEVICE)
    indices = all_actions[:, :-1].T.unsqueeze(2).repeat(1, 1, 4)
    action_decoder_input[1:, :, :] = torch.gather(all_base_pairs, 0, indices)
    all_action_probs = net(all_base_pairs, action_decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)

    # Compute shaping terms
    indices = all_actions.T.unsqueeze(2).repeat(1, 1, 4)
    reward_decoder_input = torch.gather(all_base_pairs, 0, indices)
    all_shaping_terms = net.shaping_terms(all_base_pairs, reward_decoder_input, tgt_mask=tgt_mask)

    # Compute rewards from shaping terms: reward for transitioning from s to s' is
    # R(s, s') = T(s') + h(s') - h(s)
    all_rewards = torch.zeros((episode_length, AGENT_BATCH_SIZE + EXPERT_BATCH_SIZE, 1), device=DEVICE)
    all_rewards += all_shaping_terms  # + h(s')
    all_rewards[1:] -= all_shaping_terms[:-1]  # - h(s)
    all_rewards[-1, :, :] += all_terminal_rewards  # + T(s')

    # Calculate mean cross-entropy loss for the discriminator & calculate gradients
    all_taken_action_probs = torch.gather(all_action_probs, 2, all_actions.unsqueeze(2)).transpose(0, 1)
    all_taken_action_probs_array = all_taken_action_probs.detach().cpu().numpy()
    # Detach action probs since we don't want to take their gradient when differentiating the discriminator loss
    D = torch.exp(all_rewards) / (torch.exp(all_rewards) + all_taken_action_probs.detach())
    discriminator_loss = - (torch.sum(torch.log(1 - D[:AGENT_BATCH_SIZE, :]))
                            + torch.sum(torch.log(D[AGENT_BATCH_SIZE:, :]))) / (AGENT_BATCH_SIZE + EXPERT_BATCH_SIZE)
    discriminator_loss.backward()

    # Calculate returns using learnt rewards
    # (detach since we don't want to differentiate the reward terms when finding the gradient of the policy objective)
    agent_rewards = all_rewards[:AGENT_BATCH_SIZE, :].detach()
    agent_action_log_probs = torch.log(all_taken_action_probs[:AGENT_BATCH_SIZE] + EPS)
    agent_returns = torch.flip(torch.cumsum(torch.flip(agent_rewards, [1]), 1), [1])
    # Calculate policy objective + gradients
    policy_loss = -torch.sum(agent_returns * agent_action_log_probs) / AGENT_BATCH_SIZE
    policy_loss.backward()

    # Update reward + policy net
    optimizer.step()

    if i % TEST_INTERVAL == 1:
        # Test new policy net + calculate optimality gap
        test_instances, _, test_measures = test_data.get_batch(episode_length, TEST_BATCH_SIZE, DEVICE)
        test_agent_measures = torch.zeros(TEST_BATCH_SIZE)
        state = env.reset(instances=test_instances)
        for t in range(episode_length):
            base_pairs, prev_actions = state
            decoder_input = torch.zeros(t + 1, AGENT_BATCH_SIZE, 4, device=DEVICE)
            indices = prev_actions.T.unsqueeze(2).repeat(1, 1, 4)
            decoder_input[1:, :, :] = torch.gather(base_pairs, 0, indices)
            tgt_mask = generate_square_subsequent_mask(t + 1)
            memory_masks = generate_batch_of_sorted_element_masks(prev_actions, episode_length, NB_HEADS)
            action_probs = net(base_pairs, decoder_input, tgt_mask=tgt_mask, memory_mask=memory_masks)[:, -1, :]
            actions = torch.multinomial(action_probs, 1)
            states, is_terminal = env.step(actions)
        base_pairs, prev_actions = state
        for j in range(AGENT_BATCH_SIZE):
            # Convert jth set of base pairs into format required by copt.evaluate (list of 4-tuples)
            problem = [tuple(base_pair) for base_pair in base_pairs[:, j, :].tolist()]
            ordering = prev_actions[j, :].tolist()
            test_agent_measures[j, 0] = copt.evaluate(problem, ordering)['measure']
        # Calculate mean optimality gap
        mean_optimality_gap = (1 - test_agent_measures/test_measures).mean()
        wandb.log({'mean_optimality_gap': mean_optimality_gap})





