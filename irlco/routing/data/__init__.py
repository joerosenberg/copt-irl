from torch.utils.data import Dataset
import yaml
import torch
from pathlib import Path


class CircuitSolutionDataset(Dataset):
    def __init__(self, config_path: str):
        # Load config
        p = Path(config_path)
        self.data_dir_path = p.parent
        config_file = open(config_path, 'r')
        self.config = yaml.load_all(config_file)

        # Create empty lists to hold data from the config files
        self.instances = []
        self.solutions = []
        self.measures = []
        self.indices = {}
        self.episode_indices = {}
        self.nb_episodes_stored = {}

        # Tracks index for each entry we add
        index = 0

        # Fill in empty lists with data from files specified in the config:
        for i, config_entry in enumerate(self.config):
            # Read data from corresponding data file
            data_file = open(self.data_dir_path / Path(config_entry['output_file']), 'r')
            data = yaml.load_all(data_file)

            # Create empty tensors to hold data for this file:
            nb_solutions = config_entry['nb_instances'] * config_entry['nb_top_solutions']
            instance_size = config_entry['instance_size']
            # Initialise so dimensions match (seq length, batch size, nb_features) for transformer
            self.instances.append(torch.zeros(instance_size, nb_solutions, 4))
            self.solutions.append(torch.zeros(instance_size, nb_solutions, dtype=torch.long))
            self.measures.append(torch.zeros(nb_solutions, 1))

            self.episode_indices[instance_size] = i

            # Write data from this file into empty tensors & simultaneously create indices for each entry
            for j, data_entry in enumerate(data):
                self.instances[i][:, j, :] = torch.FloatTensor(data_entry['instance'])
                self.solutions[i][:, j] = torch.LongTensor(data_entry['order'])
                self.measures[i][j, 0] = float(data_entry['measure'])
                self.indices[index] = (i, j)
                index += 1
                self.nb_episodes_stored[instance_size] = j + 1

    def __len__(self):
        # Read lengths of each data file from the config file and sum them to obtain total length
        length = sum([config_entry['nb_instances'] * config_entry['nb_top_solutions'] for config_entry in self.config])
        return length

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns: Tuple of instance, solution (as connection order) and measure.

        """
        assert 0 <= index < self.__len__()
        i, j = self.indices[index]
        return self.instances[i][:, j, :], self.solutions[i][:, j], self.measures[i][j, :]

    def get_batch(self, episode_length, batch_size, device):
        # Get data for episodes of requested length:
        episode_index = self.episode_indices[episode_length]
        dataset_size = self.nb_episodes_stored[episode_length]
        instances = self.instances[episode_index]
        solutions = self.solutions[episode_index]
        measures = self.measures[episode_index]
        # Sample indices
        batch_indices = torch.randint(0, dataset_size, [batch_size])
        return instances[:, batch_indices, :].to(device), solutions[:, batch_indices].to(device), \
               measures[batch_indices, :].to(device)
