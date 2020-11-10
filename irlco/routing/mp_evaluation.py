from multiprocessing import Pool
import copt


def mp_evaluate(problems, orderings, nb_workers=12):
    """
    Evaluates multiple solutions in parallel.
    Args:
        problems: List of tuples of base pairs.
        orderings: List of lists of connection orderings.
        nb_workers: Number of CPU workers to use when evaluating. Should be less than the number of CPUs on your system.

    Returns: List of tuples (measure, success) for each solution.

    """
    with Pool(nb_workers) as pool:
        evaluations = pool.map(evaluate_unpack, zip(problems, orderings))
    return evaluations


def evaluate_unpack(x):
    """
    Evaluates a single problem and connection ordering.
    Args:
        x: Tuple (problem, ordering) where problem is a tuple of base pairs and ordering is a list of nodes to connect.

    Returns: measure and success of the given problem and ordering.

    """
    evaluation = copt.evaluate(x[0], x[1])
    return evaluation['measure'], evaluation['success']
