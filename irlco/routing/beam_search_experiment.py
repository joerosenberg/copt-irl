import time

from irlco.routing.train import load_pickled_data
from irlco.pointer_transformer import TwinDecoderPointerTransformer
from irlco.routing.policy import beam_search_decode
from irlco.routing.env import generate_valid_instances
import torch
import wandb

if __name__ == '__main__':
    MODEL_TYPE = 'no_shaped'  # or 'no_shaped' or 'ail_shared'

    if MODEL_TYPE == 'ail_separate':
        RUN_NAME = 'drawn-planet-268'
        RUN_ID = '2qjzuvj0'
        STEP = 2100
    elif MODEL_TYPE == 'no_shaped':
        RUN_NAME = 'copper-lion-267'
        RUN_ID = '108xpzcc'
        STEP = 1750
    elif MODEL_TYPE == 'ail_shared':
        RUN_NAME = 'volcanic-rain-265'
        RUN_ID = '3jfcmcp1'
        STEP = 2150
    else:
        RUN_NAME = 'earnest-music-260'
        RUN_ID = '13g0ijcq'
        STEP = 475

    PATH = f'./saved_models/{RUN_NAME}/{RUN_NAME}_step_{str(STEP)}_model'

    TEST_DATA_PATH = './data/test_data_config.yaml'
    PICKLED_TEST_DATA_PATH = './data/pickle/test_data.pkl'

    USE_RANDOM_DATA = False

    NB_TEST_BATCHES = 5
    TEST_BATCH_SIZE = 256

    DEVICE = torch.device('cuda')

    # Get model hyperparameters and load model in evaluation mode
    api = wandb.Api()
    run = api.run(f'routing/{RUN_ID}')
    p = run.config
    net = TwinDecoderPointerTransformer(d_input=4,
                                        d_model=p['embedding_dim'],
                                        dim_feedforward=p['ff_dim'],
                                        nhead=p['nb_heads'],
                                        dropout=p['dropout'],
                                        num_encoder_layers=p['nb_encoder_layers'],
                                        num_decoder_layers=p['nb_decoder_layers'],
                                        shared_encoder=p['shared_ail_encoder']).cuda()
    net.load_state_dict(torch.load(PATH))
    net.eval()

    # Load test data
    test_data = load_pickled_data(TEST_DATA_PATH, PICKLED_TEST_DATA_PATH)

    # Run tests for episode lengths 3 to 9
    with torch.no_grad():
        for episode_length in range(3, 10):
            if USE_RANDOM_DATA:
                nb_successes = 0

                for i in range(NB_TEST_BATCHES):
                    instances = generate_valid_instances(episode_length, TEST_BATCH_SIZE)
                    empty_actions = torch.zeros((TEST_BATCH_SIZE, 0), dtype=torch.long, device=DEVICE)
                    _, _, successes = beam_search_decode((instances, empty_actions), net, beam_width=episode_length,
                                                         device=DEVICE)
                    nb_successes += successes.float().sum()

                success_rate = nb_successes / (NB_TEST_BATCHES * TEST_BATCH_SIZE)
                normalised_success_rate = success_rate / (1 - 0.17)  # Assuming approx 17% of instances have no solution
                # for size 9
                print(f'Success rate for problems of size {episode_length} was {success_rate}.')
                print(f'Normalised success rate for problems of size {episode_length} was {normalised_success_rate}.')

            else:
                sum_of_optimality_gaps = 0
                nb_successes = 0
                start = time.perf_counter()

                for i in range(NB_TEST_BATCHES):
                    instances, solutions, solution_measures = test_data.get_batch(episode_length, TEST_BATCH_SIZE,
                                                                                  DEVICE)
                    empty_actions = torch.zeros((TEST_BATCH_SIZE, 0), dtype=torch.long, device=DEVICE)
                    _, measures, successes = beam_search_decode((instances, empty_actions), net,
                                                                beam_width=episode_length, device=DEVICE)

                    nb_successes += successes.float().sum()
                    optimality_gaps_of_successes = (1 - solution_measures / measures).masked_fill(
                        torch.logical_not(successes), 0)
                    sum_of_optimality_gaps += optimality_gaps_of_successes.sum()

                end = time.perf_counter()
                torch.cuda.synchronize()

                success_rate = nb_successes / (NB_TEST_BATCHES * TEST_BATCH_SIZE)
                mean_optimality_gap_of_successes = sum_of_optimality_gaps / nb_successes
                print(f'Success rate for problems of size {episode_length} was {success_rate}.')
                print(f'Mean optimality gap for successful solutions for problems of size {episode_length}',
                      f'was {mean_optimality_gap_of_successes}.')
                print(f'Produced {NB_TEST_BATCHES * TEST_BATCH_SIZE} solutions to problems of size {episode_length}',
                      f'in {end - start} seconds.')
                print(f'Took an average of {(end-start)/(NB_TEST_BATCHES * TEST_BATCH_SIZE)} seconds per problem.')
