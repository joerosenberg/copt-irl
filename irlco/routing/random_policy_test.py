# Sanity check to see how well a random policy performs on the test data
from irlco.routing.policy import evaluate_terminal_states
from irlco.routing.train import load_pickled_data
from irlco.routing.env import BatchCircuitRoutingEnv
import torch


if __name__ == '__main__':
    test_data = load_pickled_data('./data/irl_data_config.yaml', './data/pickle/irl_data.pkl')
    test_env = BatchCircuitRoutingEnv(256, 6, 9)

    while True:
        for test_episode_length in range(6, 10):
            instances, solutions, test_measures = test_data.get_batch(test_episode_length, 256, torch.device('cuda'))
            test_states = test_env.reset(instances=instances)

            # Generate random actions
            actions = torch.zeros((256, test_episode_length), dtype=torch.long, device=torch.device('cuda'))
            for i in range(256):
                actions[i, :] = torch.randperm(test_episode_length)
    
            measures, successes = evaluate_terminal_states((instances, actions), device=torch.device('cuda'))

            print(f'success rate for length {test_episode_length}: {successes.float().mean()}')
