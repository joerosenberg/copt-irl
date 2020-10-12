from multiprocessing import Pool
import copt


def mp_evaluate(problems, orderings):
    with Pool(12) as pool:
        evaluations = pool.map(evaluate_unpack, zip(problems, orderings))
    return evaluations


def evaluate_unpack(x):
    evaluation = copt.evaluate(x[0], x[1])
    return evaluation['measure'], evaluation['success']
