"""
Algorithm that uses shaped reward signal learned using the adversarial IRL method from 'Learning Robust Rewards with
Adversarial Inverse Reinforcement Learning' (Fu, Luo and Levine 2018).
"""
import torch
import wandb

import irlco.pointer_transformer as pt
from irlco.routing.data import CircuitSolutionDataset
from irlco.routing.env import BatchCircuitRoutingEnv, measures_to_terminal_rewards
from irlco.routing.policy import sample_best_of_n_trajectories, trajectory_action_probabilities, greedy_decode
from irlco.routing.reward import compute_shaping_terms, shaping_terms_to_rewards

EMBEDDING_DIM = 256 # Dimension of input embedding
NB_HEADS = 8  # Number of heads in multihead attention modules
assert EMBEDDING_DIM % NB_HEADS == 0
FF_DIM = 256  # Dimension of feedforward layers in transformer
DROPOUT = 0.0
AGENT_BATCH_SIZE = 512  # Number of batches
EXPERT_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
TEST_INTERVAL = 5
NB_EPISODES = 20_000
MIN_INSTANCE_SIZE = 3  # Minimum number of base pairs in training instances
MAX_INSTANCE_SIZE = 5  # Maximum number of base pairs in training instances
LR = 1e-4  # Learning rate
EPS = 1e-8
DEVICE = torch.device('cuda')
EXPERT_DATA_PATH = './data/small_test_config.yaml'
TEST_DATA_PATH = './data/small_test_config.yaml'


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
    agent_base_pairs, _ = states
    episode_length = states[0].shape[0]

    # Collect trajectories by executing current policy:
    agent_actions, agent_action_probs, agent_measures, agent_successes = sample_best_of_n_trajectories(env, states, net, 1)

    # Map agent's solution measures to terminal rewards
    agent_terminal_rewards = measures_to_terminal_rewards(episode_length, agent_measures, successes=agent_successes)

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

    # Compute action probabilities for all actions
    all_trajectory_action_probs = trajectory_action_probabilities((all_base_pairs, all_actions), net)

    # Compute shaping terms
    all_shaping_terms = compute_shaping_terms((all_base_pairs, all_actions), net)

    # Compute rewards from shaping terms: reward for transitioning from s to s' is
    all_rewards = shaping_terms_to_rewards(all_shaping_terms, all_terminal_rewards)

    # Calculate mean cross-entropy loss for the discriminator & calculate gradients
    # Detach action probs since we don't want to take their gradient when differentiating the discriminator loss
    # D = torch.exp(all_rewards) / (torch.exp(all_rewards) + all_trajectory_action_probs.detach())
    # discriminator_loss = - (torch.sum(torch.log(1 - D[:AGENT_BATCH_SIZE, :]))
    #                         + torch.sum(torch.log(D[AGENT_BATCH_SIZE:, :]))) / (AGENT_BATCH_SIZE + EXPERT_BATCH_SIZE)
    # discriminator_loss.backward()

    # Calculate returns using learnt rewards
    # (detach since we don't want to differentiate the reward terms when finding the gradient of the policy objective)
    agent_rewards = all_rewards[:AGENT_BATCH_SIZE, :].detach()
    agent_action_log_probs = torch.log(all_trajectory_action_probs[:AGENT_BATCH_SIZE] + EPS)
    agent_returns = torch.flip(torch.cumsum(torch.flip(agent_rewards, [1]), 1), [1])
    # Calculate policy objective + gradients
    policy_loss = -torch.sum(agent_returns * agent_action_log_probs) / AGENT_BATCH_SIZE
    policy_loss.backward()

    # Log the losses + mean terminal reward
    mean_agent_terminal_reward = all_terminal_rewards[:AGENT_BATCH_SIZE].mean().cpu()
    # wandb.log({'discriminator_loss': discriminator_loss, 'policy_loss': policy_loss,
    #            'mean_terminal_reward': mean_agent_terminal_reward})
    wandb.log({'policy_loss': policy_loss, 'mean_terminal_reward': mean_agent_terminal_reward})

    # Update reward + policy net
    optimizer.step()

    if i % TEST_INTERVAL == 1:
        # Test new policy net + calculate optimality gap
        test_base_pairs, _, test_measures = test_data.get_batch(episode_length, TEST_BATCH_SIZE, DEVICE)
        test_states = env.reset(instances=test_base_pairs)
        _, _, test_agent_measures, test_agent_successes = sample_best_of_n_trajectories(env, test_states, net, 1)

        # Calculate mean optimality gap (unsuccessful solutions are considered to have an optimality gap of 1)
        optimality_gaps = (1 - test_measures/test_agent_measures).masked_fill(torch.logical_not(test_agent_successes), 1)
        print(optimality_gaps)
        mean_optimality_gap = optimality_gaps.mean().cpu()
        print(mean_optimality_gap)
        wandb.log({'mean_optimality_gap': mean_optimality_gap})





