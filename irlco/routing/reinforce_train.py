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

if __name__ == '__main__':
    freeze_support()
    EMBEDDING_DIM = 32
    NB_HEADS = 4
    FF_DIM = 512
    DROPOUT = 0.0
    NB_INSTANCES_PER_BATCH = 16
    NB_TRAJECTORIES_PER_INSTANCE = 32
    BATCH_SIZE = NB_TRAJECTORIES_PER_INSTANCE * NB_INSTANCES_PER_BATCH
    NB_EPISODES = 20_000
    MIN_INSTANCE_SIZE = 6
    MAX_INSTANCE_SIZE = 9
    LR = 1e-4
    EPS = 1e-8
    DEVICE = torch.device('cuda')

    NB_ENCODER_LAYERS = 3
    NB_DECODER_LAYERS = 3

    PPO_EPS = 0.2

    TEST_INTERVAL = 25
    TEST_DATA_PATH = './data/test_data_config.yaml'
    TEST_BATCH_SIZE = 64
    NB_TEST_SAMPLES = 128
    TEST_DATA_PICKLE_PATH = './data/pickle/test_data.pkl'

    wandb.init(project='routing', config={
        'embedding_dimension': EMBEDDING_DIM,
        'nb_heads': NB_HEADS,
        'min_instance_size': MIN_INSTANCE_SIZE,
        'max_instance_size': MAX_INSTANCE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'nb_encoder_layers': NB_ENCODER_LAYERS,
        'nb_decoder_layers': NB_DECODER_LAYERS
    })

    dummy_env = BatchCircuitRoutingEnv(NB_INSTANCES_PER_BATCH, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    env = BatchCircuitRoutingEnv(BATCH_SIZE, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    test_env = BatchCircuitRoutingEnv(TEST_BATCH_SIZE, MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE)
    net = pt.TwinDecoderPointerTransformer(4, EMBEDDING_DIM, NB_HEADS, NB_ENCODER_LAYERS, NB_DECODER_LAYERS, FF_DIM, DROPOUT).cuda()
    # Create baseline net with same parameters as net
    baseline_net = pt.TwinDecoderPointerTransformer(4, EMBEDDING_DIM, NB_HEADS, NB_ENCODER_LAYERS, NB_DECODER_LAYERS, FF_DIM, DROPOUT).cuda()
    baseline_net.load_state_dict(net.state_dict())
    baseline_net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    wandb.watch(net)

    # Variables for tracking baseline
    best_success_rate = 0.0

    if os.path.isfile(TEST_DATA_PICKLE_PATH):
        with open(TEST_DATA_PICKLE_PATH, 'rb') as pickled_test_data:
            test_data = pickle.load(pickled_test_data)
    else:
        test_data = CircuitSolutionDataset(TEST_DATA_PATH)
        test_data.config = None  # Get rid of yaml object so we can pickle
        with open(TEST_DATA_PICKLE_PATH, 'wb') as pickled_test_data:
            pickle.dump(test_data, pickled_test_data)

    for i in range(NB_EPISODES):
        instances, _ = dummy_env.reset()
        states = env.reset(instances=instances.repeat(1, NB_TRAJECTORIES_PER_INSTANCE, 1))
        base_pairs, _ = states
        episode_length = base_pairs.shape[0]

        actions, action_probs, measures, successes = sample_best_of_n_trajectories(env, states, net, 1)
        terminal_rewards = measures_to_terminal_rewards(episode_length, measures, successes=successes)
        log_action_probs = torch.log(action_probs + EPS)

        returns = terminal_rewards.repeat(1, episode_length)

        if i > TEST_INTERVAL:
            greedy_baselines = greedy_rollout_baselines(base_pairs, actions, env, baseline_net, device=DEVICE)
        else:
            greedy_baselines = torch.zeros((BATCH_SIZE, episode_length), device=DEVICE)

        action_prob_ratios = action_probs / action_probs.detach()
        ppo_terms = torch.min(action_prob_ratios * (returns - greedy_baselines),
                              torch.clamp(action_prob_ratios, 1-PPO_EPS, 1+PPO_EPS) * (returns - greedy_baselines))
        # policy_loss = -torch.sum(returns * log_action_probs) / BATCH_SIZE
        policy_loss = - torch.sum(ppo_terms) / BATCH_SIZE
        policy_loss.backward()

        optimizer.step()

        torch.save(net.state_dict(), 'reinforce_model')

        wandb.log({'policy_loss': policy_loss, 'mean_terminal_reward': terminal_rewards.mean(),
                   'success_rate': successes.float().mean()})

        if i % TEST_INTERVAL == 0 and i != 0:
            # For storing aggregate stats over all episode lengths:
            overall_mean_optimality_gap = 0
            overall_success_rate = 0
            for test_episode_length in range(MIN_INSTANCE_SIZE, MAX_INSTANCE_SIZE + 1):
                instances, solutions, test_measures = test_data.get_batch(test_episode_length, TEST_BATCH_SIZE, DEVICE)
                test_states = test_env.reset(instances=instances)
                with torch.no_grad():
                    _, _, measures, successes = greedy_decode(test_env, test_states, net, device=DEVICE)
                    optimality_gaps = (1 - test_measures/measures).masked_fill(torch.logical_not(successes), 1)
                    mean_optimality_gap = optimality_gaps.mean() # For this instance size
                    success_rate = successes.float().mean() # For this instance size

                    wandb.log({f'mean_optimality_gap_{test_episode_length}': mean_optimality_gap,
                               f'success_rate_{test_episode_length}': success_rate}, commit=False)

                    overall_mean_optimality_gap += mean_optimality_gap
                    overall_success_rate += success_rate

            overall_mean_optimality_gap = overall_mean_optimality_gap / (MAX_INSTANCE_SIZE + 1 - MIN_INSTANCE_SIZE)
            overall_success_rate = overall_success_rate / (MAX_INSTANCE_SIZE + 1 - MIN_INSTANCE_SIZE)

            wandb.log({f'overall_mean_optimality_gap': overall_mean_optimality_gap,
                       f'overall_success_rate': overall_success_rate}, commit=False)

            if overall_success_rate > best_success_rate:
                best_success_rate = overall_success_rate
                baseline_net.load_state_dict(net.state_dict())





