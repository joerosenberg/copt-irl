import copt
import yaml
from typing import List, Dict
from tqdm import tqdm
import os.path


def top_solutions(problem_instance, n_top: int) -> List[Dict]:
    """
    Finds the n_top best solutions using brute force search.
    Args:
        problem_instance: Description of the problem instance.
        n_top: Number of solutions to return.

    Returns:
        List of the n_top best solutions to the problem.

    """
    # Brute force all possible solutions - bruteForce returns a list ordered by measure
    all_solutions = copt.bruteForce(problem_instance)
    # Get best n_top solutions
    best_solns = all_solutions[:n_top]
    # If any of the best solutions are failures, return nothing
    for i in range(len(best_solns)):
        if best_solns[i]['success'] == 0:
            return []
    # Otherwise, return the best solutions
    return all_solutions[:n_top]


def generate_data(config_path):
    config_file = open(config_path)
    configs = yaml.load_all(config_file)

    for config in configs:
        instance_size = config['instance_size']
        nb_instances = config['nb_instances']
        nb_top_solutions = config['nb_top_solutions']
        output_file = config['output_file']

        # Check if file exists already - if it does, don't generate it again
        if os.path.exists(output_file):
            continue
        else:
            out = open(output_file, 'w')

        for i in tqdm(range(nb_instances)):
            problem = copt.getProblem(instance_size)
            best_solns = top_solutions(problem, nb_top_solutions)
            # Get rid of path data since we don't need it, and add instance description to each dict
            for soln in best_solns:
                del soln['pathData']
                soln['instance'] = problem
            yaml.dump_all(best_solns, out)


if __name__ == '__main__':
    generate_data("irl_data_config.yaml")
    generate_data("test_data_config.yaml")
