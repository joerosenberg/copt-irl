import copt
from typing import List, Tuple, Dict
from irlco.types import ProblemInstance, Solution
import irlco.solution_pb2 as pb

def serialize_solution(problem_instance: ProblemInstance, solution: Dict) -> bytes:
    """
    Serializes a (problem instance, solution) pair according to the protocol given in solution.proto so that it can be
    written to disk.
    Args:
        problem_instance: Problem instance given as sequence of base pairs.
        solution: Solution in the format given by copt.bruteForce.

    Returns: Serialized form of the inputs.

    """
    entry = pb.Solution()

    # Add problem description
    for i in range(len(problem_instance)):
        base_pair = entry.instance.add()
        base_pair.start.x, base_pair.start.y, base_pair.end.x, base_pair.end.y = problem_instance[i]

    entry.order.extend(solution['order'])
    entry.success = bool(solution['success'])
    entry.measure = solution['measure']
    entry.nRouted = solution['nRouted']
    # entry.pathData = ...
    entry.failedConnections.extend(solution['failedConnections'])

    return entry.SerializeToString()


def read_serialized_solution(serialized_soln: str) -> Tuple[ProblemInstance, Dict]:
    entry = pb.Solution()
    entry.ParseFromString(serialized_soln)

    # Read base pairs
    instance = [(0, 0, 0, 0)] * len(entry.instance)
    for i, base_pair in enumerate(entry.instance):
        instance[i] = (base_pair.start.x, base_pair.start.y, base_pair.end.x, base_pair.end.y)

    # Read solution
    solution = {
        'order': entry.order[:],
        'success': entry.success,
        'measure': entry.measure,
        'nRouted': entry.nRouted,
        'failedConnections': entry.failedConnections[:]
    }

    return instance, solution


if __name__ == '__main__':
    problem = copt.getProblem(3)
    soln = copt.bruteForce(problem)[0]
    print(problem)
    print(soln)
    decoded_problem, decoded_soln = read_serialized_solution(serialize_solution(problem, soln))
    print(serialize_solution(problem, soln))
    print(decoded_problem)
    print(decoded_soln)