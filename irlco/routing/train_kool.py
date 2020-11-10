import torch
import wandb
import irlco.pointer_transformer as pt
from irlco.routing.baselines import greedy_rollout_baselines
from irlco.routing.data import CircuitSolutionDataset
from irlco.routing.env import BatchCircuitRoutingEnv, measures_to_terminal_rewards
from irlco.routing.policy import sample_best_of_n_trajectories, trajectory_action_probabilities, greedy_decode
import pickle
import os
from multiprocessing import freeze_support

from irlco.routing.reward import compute_shaping_terms, shaping_terms_to_rewards


def load_pickled_data(data_config_path, data_pickle_path):
    if os.path.isfile(data_pickle_path):
        with open(data_pickle_path, 'rb') as pickled_data:
            data = pickle.load(pickled_data)
    else:
        data = CircuitSolutionDataset(data_config_path)
        data.config = None  # Get rid of yaml object so we can pickle
        with open(data_pickle_path, 'wb') as pickled_data:
            pickle.dump(data, pickled_data)
    return data


if __name__ == '__main__':
    # For multiprocessing support on Windows
    freeze_support()

    # Transformer model parameters
    EMBEDDING_DIM = 64
    NB_HEADS = 8
    FF_DIM = 512
    DROPOUT = 0.0
    NB_ENCODER_LAYERS = 3
    NB_DECODER_LAYERS = 3

    # Environment parameters
    MIN_INSTANCE_SIZE = 6
    MAX_INSTANCE_SIZE = 9

    # Training parameters
    NB_INSTANCES_PER_BATCH = 512  # Number of unique circuit routing problems to consider in each batch
    NB_TRAJECTORIES_PER_INSTANCE = 1  # Number of trajectories to sample for each unique circuit routing problem
    BATCH_SIZE = NB_TRAJECTORIES_PER_INSTANCE * NB_INSTANCES_PER_BATCH
    NB_EPISODES = 20_000
    LR = 1e-4  # Optimizer learning rate
    EPS = 1e-8  # Add when computing log-probabilities from probabilities to avoid numerical instability
    DEVICE = torch.device('cuda')
    ENTROPY_REGULARISATION_WEIGHT = 0.1

    # Qualitative training parameters
    BASELINE_METHOD = 'none'  # 'greedy' for greedy rollouts or 'none'
    REWARD_SHAPING_METHOD = 'none'  # 'ail' for adversarial imitation learning or 'none'
    SHARED_AIL_ENCODER = True  # Whether or not to share the transformer encoder between the policy and discriminator

    # Adversarial imitation learning (reward shaping) parameters
    NB_EXPERT_SAMPLES = BATCH_SIZE  # Keep it equal to batch size for now, so that the discriminator sees an equal
    # amount of expert and non-expert data
    USE_ACTION_PROBS_FOR_DISCRIMINATOR = False

    # PPO surrogate loss clipping parameter
    PPO_EPS = 0.2

    # Test parameters
    TEST_INTERVAL = 25
    TEST_BATCH_SIZE = 256
    TEST_DECODING_METHOD = 'greedy'  # or 'sampling'
    NB_TEST_SAMPLES = 128  # Number of samples to take if decoding method is 'sampling'

    # Model saving interval
    SAVE_INTERVAL = TEST_INTERVAL

    # Data file paths
    TEST_DATA_PATH = './data/test_data_config.yaml'
    TEST_DATA_PICKLE_PATH = './data/pickle/test_data.pkl'
    EXPERT_DATA_PATH = './data/irl_data_config.yaml'
    EXPERT_DATA_PICKLE_PATH = './data/pickle/irl_data.pkl'

    wandb.init(project='routing', config={
        'embedding_dim': EMBEDDING_DIM,
        'nb_heads': NB_HEADS,
        'ff_dim': FF_DIM,
        'dropout': DROPOUT,
        'nb_encoder_layers': NB_ENCODER_LAYERS,
        'nb_decoder_layers': NB_DECODER_LAYERS,
        'min_instance_size': MIN_INSTANCE_SIZE,
        'max_instance_size': MAX_INSTANCE_SIZE,
        'nb_instances_per_batch': NB_INSTANCES_PER_BATCH,
        'nb_trajectories_per_instance': NB_TRAJECTORIES_PER_INSTANCE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'entropy_regularisation_weight': ENTROPY_REGULARISATION_WEIGHT,
        'baseline_method': BASELINE_METHOD,
        'reward_shaping_method': REWARD_SHAPING_METHOD,
        'shared_ail_encoder': SHARED_AIL_ENCODER,
        'nb_expert_samples': NB_EXPERT_SAMPLES,
        'ppo_clipping_parameter': PPO_EPS,
        'use_actions_probs_for_discriminator': USE_ACTION_PROBS_FOR_DISCRIMINATOR
    })

    # Environments for sampling unique problems, stepping forward during training, and testing
    dummy_env = BatchCircuitRoutingEnv(NB_INSTANCES_PER_BATCH, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    env = BatchCircuitRoutingEnv(BATCH_SIZE, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    test_env = BatchCircuitRoutingEnv(TEST_BATCH_SIZE, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    # Shared net for policy + shaped rewards
    net = pt.KoolModel().cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    wandb.watch(net)

    # Make directory for saving model
    os.mkdir(f'./saved_models/{wandb.run.name}')

    if BASELINE_METHOD == 'greedy':
        # Create baseline net with same parameters as net
        baseline_net = pt.KoolModel()
        baseline_net.load_state_dict(net.state_dict())
        baseline_net.eval()
    elif BASELINE_METHOD == 'none':
        baseline_net = None
    else:
        raise Exception

    # Variables for tracking baseline
    best_success_rate = 0.0

    # Load data files - if a pickled copy of the data exists, load that instead
    test_data = load_pickled_data(TEST_DATA_PATH, TEST_DATA_PICKLE_PATH)
    expert_data = load_pickled_data(EXPERT_DATA_PATH, EXPERT_DATA_PICKLE_PATH)

    for i in range(NB_EPISODES):
        ''' Sample trajectories '''
        # Sample NB_INSTANCES_PER_BATCH unique circuit routing problems
        instances, _ = dummy_env.reset()
        # Duplicate each problem NB_TRAJECTORIES_PER_INSTANCE times so we sample that many trajectories for each problem
        states = env.reset(instances=instances.repeat(1, NB_TRAJECTORIES_PER_INSTANCE, 1))
        base_pairs, _ = states
        episode_length = base_pairs.shape[0]

        # Sample trajectories according to policy given by net
        if ENTROPY_REGULARISATION_WEIGHT > 0:
            actions, action_probs, measures, successes, all_action_probs = net.sample_decode(states, env)
        else:
            actions, action_probs, measures, successes = sample_best_of_n_trajectories(env, states, net, 1)

        ''' Compute rewards and returns '''
        # Compute terminal rewards for each solution
        terminal_rewards = measures_to_terminal_rewards(episode_length, measures, successes=successes)

        if REWARD_SHAPING_METHOD == 'ail':
            # Get expert data for discriminator
            expert_base_pairs, expert_actions, expert_measures = expert_data.get_batch(episode_length,
                                                                                       NB_EXPERT_SAMPLES, DEVICE)
            expert_actions = expert_actions.T
            # Get terminal rewards for expert solutions (they are guaranteed to be successful solutions, so we don't
            # need to pass successes
            expert_terminal_rewards = measures_to_terminal_rewards(episode_length, expert_measures)
            # Concatenate policy data and expert data together so we can compute in a single batch
            disc_base_pairs = torch.cat((base_pairs, expert_base_pairs), dim=1)
            disc_actions = torch.cat((actions, expert_actions), dim=0)
            disc_terminal_rewards = torch.cat((terminal_rewards, expert_terminal_rewards))
            # trajectory_action_probabilities computes the probabilities that the current agent would take the expert's
            # actions
            expert_action_probs = trajectory_action_probabilities((expert_base_pairs, expert_actions), net).squeeze(2).T
            disc_action_probs = torch.cat((action_probs, expert_action_probs), dim=0)

            # Compute shaping terms for both agent and expert trajectories
            disc_shaping_terms = compute_shaping_terms((disc_base_pairs, disc_actions), net)
            # Compute rewards from shaping terms
            disc_rewards = shaping_terms_to_rewards(disc_shaping_terms, disc_terminal_rewards).squeeze(2).T

            # Calculate mean cross-entropy loss for the discriminator
            if USE_ACTION_PROBS_FOR_DISCRIMINATOR:
                is_expert_transition_probs = torch.exp(disc_rewards) / (
                            torch.exp(disc_rewards) + disc_action_probs.detach())
            else:
                is_expert_transition_probs = torch.exp(disc_rewards) / (1 + torch.exp(disc_rewards))

            # Calculate misclassification rates for logging
            false_positive_rate = (is_expert_transition_probs[:BATCH_SIZE, :] > 0.5).float().mean()
            false_negative_rate = (is_expert_transition_probs[BATCH_SIZE:, :] < 0.5).float().mean()
            wandb.log({'false_positive_rate': false_positive_rate, 'false_negative_rate': false_negative_rate},
                      commit=False)

            discriminator_loss = - (
                    torch.sum(torch.log(1 - is_expert_transition_probs[:BATCH_SIZE, :])) +
                    torch.sum(torch.log(is_expert_transition_probs[BATCH_SIZE:, :]))
            ) / (BATCH_SIZE + NB_EXPERT_SAMPLES)
            discriminator_loss.backward()
            wandb.log({'discriminator_loss': discriminator_loss}, commit=False)

            # Compute returns for agent
            returns = torch.flip(torch.cumsum(torch.flip(disc_rewards[:BATCH_SIZE], [1]), 1), [1]).detach()
        elif REWARD_SHAPING_METHOD == 'none':
            returns = terminal_rewards.repeat(1, episode_length)
        else:
            raise Exception

        # Compute baselines
        if i > TEST_INTERVAL and BASELINE_METHOD == 'greedy':
            baselines = greedy_rollout_baselines(base_pairs, actions, env, baseline_net, device=DEVICE)
        else:
            baselines = torch.zeros((BATCH_SIZE, episode_length), device=DEVICE)

        ''' Compute loss and update policy network '''
        # Compute entropy penalty
        if ENTROPY_REGULARISATION_WEIGHT > 0:
            entropy_terms = torch.sum(all_action_probs * torch.log(all_action_probs + EPS), dim=2)
            entropy_returns = torch.flip(torch.cumsum(torch.flip(entropy_terms, [1]), 1), [1])
            returns = returns - ENTROPY_REGULARISATION_WEIGHT * entropy_returns

        # Compute PPO loss
        action_prob_ratios = action_probs / action_probs.detach()
        ppo_terms = torch.min(action_prob_ratios * (returns - baselines),
                              torch.clamp(action_prob_ratios, 1 - PPO_EPS, 1 + PPO_EPS) * (returns - baselines))
        policy_loss = - torch.sum(ppo_terms) / BATCH_SIZE
        policy_loss.backward()

        optimizer.step()

        wandb.log({'policy_loss': policy_loss, 'mean_terminal_reward': terminal_rewards.mean(),
                   'success_rate': successes.float().mean()}, commit=False)

        if i % TEST_INTERVAL == 0 and i != 0:
            # For storing aggregate stats over all episode lengths:
            overall_mean_optimality_gap = 0
            overall_success_rate = 0
            for test_episode_length in range(MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE + 1):
                instances, solutions, test_measures = test_data.get_batch(test_episode_length, TEST_BATCH_SIZE, DEVICE)
                test_states = test_env.reset(instances=instances)
                with torch.no_grad():
                    _, _, measures, successes, _ = net.greedy_decode(test_states, test_env)
                    optimality_gaps = (1 - test_measures / measures).masked_fill(torch.logical_not(successes), 1)
                    mean_optimality_gap = optimality_gaps.mean()  # For this instance size
                    success_rate = successes.float().mean()  # For this instance size

                    wandb.log({f'mean_optimality_gap_{test_episode_length}': mean_optimality_gap,
                               f'success_rate_{test_episode_length}': success_rate}, commit=False)

                    overall_mean_optimality_gap += mean_optimality_gap
                    overall_success_rate += success_rate

            overall_mean_optimality_gap = overall_mean_optimality_gap / (MAX_INSTANCE_SIZE + 1 - MIN_INSTANCE_SIZE)
            overall_success_rate = overall_success_rate / (MAX_INSTANCE_SIZE + 1 - MIN_INSTANCE_SIZE)

            wandb.log({f'overall_mean_optimality_gap': overall_mean_optimality_gap,
                       f'overall_success_rate': overall_success_rate}, commit=False)

            if overall_success_rate > best_success_rate and BASELINE_METHOD != 'none':
                best_success_rate = overall_success_rate
                baseline_net.load_state_dict(net.state_dict())

        if i % SAVE_INTERVAL == 0 and i != 0:
            torch.save(net.state_dict(), f'./saved_models/{wandb.run.name}/{wandb.run.name}_step_{i}_model')

        wandb.log({})  # Update log counter
