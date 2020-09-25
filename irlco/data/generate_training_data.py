import copt
import yaml


def top_solutions(problem_instance: ProblemInstance, n_top: int) -> List[Dict]:
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
    best_solns = all_solutions[:n_top]
    # If any of the best solutions are failures, return nothing
    for i in range(n_top):
        if best_solns[i]['success'] == 0:
            return []
    # Otherwise, return the best solutions
    return all_solutions[:n_top]


if __name__ == '__main__':
    config_file = open("generator_config.yaml")
    configs = yaml.load_all(config_file)

    for config in configs:
        instance_size = config['instance_size']
        nb_instances = config['nb_instances']
        nb_top_solutions = config['nb_top_solutions']
        output_file = config['output_file']

        out = open(output_file, 'w')

        for i in range(nb_instances):
            problem = copt.getProblem(instance_size)
            best_solns = top_solutions(problem, nb_top_solutions)
            yaml.dump_all(best_solns, out)
