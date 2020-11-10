import copt
import yaml
from typing import List, Dict
from tqdm import tqdm
import os.path
from pathlib import Path
import re
from multiprocessing import Process, Queue


def top_solutions(problem_instance, n_top: int) -> List[Dict]:
    """
    Finds the n_top best solutions using brute force search.
    Args:
        problem_instance: Description of the problem instance as a tuple of base pairs.
        n_top: Number of solutions to return.

    Returns:
        List of the n_top best solutions to the problem.

    """
    # Brute force all possible solutions - bruteForce returns a list ordered by measure
    all_solutions = copt.bruteForce(problem_instance)
    # Get best n_top solutions
    best_solns = all_solutions[:n_top]
    # If any of the best solutions are failures, retry with a new problem instance
    for i in range(len(best_solns)):
        if best_solns[i]['success'] == 0:
            return []
    # Otherwise, return the best solutions
    return all_solutions[:n_top]


def top_solutions_mp(problem_instance, n_top: int, queue: Queue):
    """
    Wraps top_solutions so it can be run in a separate process.

    Args:
        problem_instance: Description of the problem instance as a tuple of base pairs.
        n_top: Number of solutions to return.
        queue: multiprocessing.Queue to write the top solutions to.

    """
    result = top_solutions(problem_instance, n_top)
    queue.put(result)


def generate_data(config_path):
    """

    Args:
        config_path:

    Returns:

    """
    config_file = open(config_path)
    configs = yaml.load_all(config_file)
    queue = Queue()

    for config in configs:
        instance_size = config['instance_size']
        nb_instances = config['nb_instances']
        nb_top_solutions = config['nb_top_solutions']
        output_file = config['output_file']

        # Check if file exists already - if it does, don't generate it again
        if os.path.exists(output_file):
            continue
        else:
            with open(output_file, 'w') as out:
                for i in tqdm(range(nb_instances)):
                    best_solns = []
                    # Keep generating problems until we get successful connections
                    while not best_solns:
                        problem = copt.getProblem(instance_size)
                        # Start new process to work around copt.bruteForce() memory leak
                        process = Process(target=top_solutions_mp, args=(problem, nb_top_solutions, queue))
                        process.start()
                        best_solns = queue.get()
                        process.join()
                    # Get rid of path data since we don't need it, and add instance description to each dict
                    for soln in best_solns:
                        del soln['pathData']
                        soln['instance'] = problem
                    if i > 0:
                        out.write('---\n')
                    yaml.dump_all(best_solns, out)


def fix_separators(config_path):
    """
    Fixes separators for the data files listed in config_path.
    Args:
        config_path: Location of the config file used to generate the data files.

    Returns: Nothing

    """
    regex = r"(success: 1\nfailedConnections)"
    subst = "success: 1\\n---\\nfailedConnections"

    parent_dir = Path(config_path).parent
    for entry in yaml.load_all(open(config_path, 'r')):
        data_file = open(parent_dir / Path(entry['output_file']), 'rw')
        content = data_file.read()
        fixed_content = re.sub(regex, subst, content)
        data_file.write(fixed_content)


if __name__ == '__main__':
    generate_data("irl_data_config.yaml")
    generate_data("test_data_config.yaml")
