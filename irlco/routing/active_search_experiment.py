import time

from irlco.routing.reward import compute_shaping_terms, shaping_terms_to_rewards
from irlco.routing.train import load_pickled_data
from irlco.pointer_transformer import TwinDecoderPointerTransformer
from irlco.routing.policy import beam_search_decode, sample_best_of_n_trajectories
from irlco.routing.env import generate_valid_instances, BatchCircuitRoutingEnv, measures_to_terminal_rewards
import torch
import wandb
from time import perf_counter

if __name__ == '__main__':
    for MODEL_TYPE in ['ail_separate', 'no_shaped', 'ail_shared']:
    # MODEL_TYPE = 'ail_shared'  # or 'no_shaped' or 'ail_shared'

        if MODEL_TYPE == 'ail_separate':
            RUN_NAME = 'drawn-planet-268'
            RUN_ID = '2qjzuvj0'
            STEP = 2100
            REWARD_SHAPING = True
        elif MODEL_TYPE == 'no_shaped':
            RUN_NAME = 'copper-lion-267'
            RUN_ID = '108xpzcc'
            STEP = 1750
            REWARD_SHAPING = False
        elif MODEL_TYPE == 'ail_shared':
            RUN_NAME = 'volcanic-rain-265'
            RUN_ID = '3jfcmcp1'
            STEP = 2150
            REWARD_SHAPING = True

        PATH = f'./saved_models/{RUN_NAME}/{RUN_NAME}_step_{str(STEP)}_model'

        TEST_DATA_PATH = './data/test_data_config.yaml'
        PICKLED_TEST_DATA_PATH = './data/pickle/test_data.pkl'

        USE_RANDOM_DATA = False

        NB_BATCHES = 5
        BATCH_SIZE = 128

        LR = 1e-4
        PPO_EPS = 0.2

        NB_TESTS = 500

        DEVICE = torch.device('cuda')

        # Get model hyperparameters and load model in evaluation mode
        api = wandb.Api()
        run = api.run(f'routing/{RUN_ID}')
        p = run.config

        # Load test data
        test_data = load_pickled_data(TEST_DATA_PATH, PICKLED_TEST_DATA_PATH)
        env = BatchCircuitRoutingEnv(BATCH_SIZE, 2, 9)
        beam_env = BatchCircuitRoutingEnv(1, 2, 9)

        successful_searches = 0
        sum_optimality_gaps = 0
        start = perf_counter()
        for i in range(NB_TESTS):
            instance, solution, test_measure = test_data.get_batch(9, 1, DEVICE)
            net = TwinDecoderPointerTransformer(d_input=4,
                                                d_model=p['embedding_dim'],
                                                dim_feedforward=p['ff_dim'],
                                                nhead=p['nb_heads'],
                                                dropout=p['dropout'],
                                                num_encoder_layers=p['nb_encoder_layers'],
                                                num_decoder_layers=p['nb_decoder_layers'],
                                                shared_encoder=p['shared_ail_encoder']).cuda()
            net.load_state_dict(torch.load(PATH))
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)

            for j in range(NB_BATCHES):
                states = env.reset(instances=instance.repeat(1, BATCH_SIZE, 1))
                base_pairs = states[0]
                actions, action_probs, measures, successes = sample_best_of_n_trajectories(env, states, net, 1)

                # If we found a successful solution, stop iterating for this problem
                if successes.float().sum() > 0:
                    successful_searches += 1
                    best_measure = torch.min(measures.masked_fill(torch.logical_not(successes), float('inf')))
                    sum_optimality_gaps += 1 - test_measure / best_measure
                    break

                # Otherwise, compute loss and take gradient step
                terminal_rewards = measures_to_terminal_rewards(9, measures, successes=successes)

                # Compute returns
                if REWARD_SHAPING:
                    shaping_terms = compute_shaping_terms((base_pairs, actions), net)
                    rewards = shaping_terms_to_rewards(shaping_terms, terminal_rewards).squeeze(2).T
                    returns = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1]).detach()
                else:
                    returns = terminal_rewards.repeat(1, 9)

                action_prob_ratios = action_probs / action_probs.detach()
                ppo_terms = torch.min(action_prob_ratios * returns,
                                      torch.clamp(action_prob_ratios, 1 - PPO_EPS, 1 + PPO_EPS) * returns)
                policy_loss = - torch.sum(ppo_terms) / BATCH_SIZE
                policy_loss.backward()

                optimizer.step()

                # On final step, perform beam search
                if j == NB_BATCHES - 1:
                    beam_state = beam_env.reset(instances=instance)
                    _, measures, successes = beam_search_decode(beam_state, net, beam_width=9, device=DEVICE)
                    if successes.any():
                        successful_searches += 1

        end = perf_counter()
        print(f"Run name: {RUN_NAME}")
        print(f"Model type: {MODEL_TYPE}")
        print(f"Average time per solution: {(end - start)/NB_TESTS} seconds for {NB_TESTS} problems")
        print(f"Average optimality gap for successful solns: {sum_optimality_gaps/successful_searches}")
        print(f"Success rate: {successful_searches/NB_TESTS}")
