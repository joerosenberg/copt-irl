import irlco.copt_env
import numpy as np
import copt

# Create a new unconnected state
problem_instance = copt.getProblem(6)
state = irlco.copt_env.CircuitRoutingState(problem_instance)
assert state.get_problem_size() == 6
print(state.get_connected_base_pairs())
assert state.get_connected_base_pairs() == []
assert state.get_unconnected_base_pairs() == list(range(6))
assert not state.is_terminal()
assert not state.is_connected(5)

# Connect a base pair
state = state.connect_base_pair(5)
assert state.get_problem_size() == 6
assert state.get_connected_base_pairs() == [5]
assert state.get_unconnected_base_pairs() == [0, 1, 2, 3, 4]
assert not state.is_terminal()
assert state.is_connected(5)

# Connect the rest of the base pairs
for i in range(5):
    state = state.connect_base_pair(i)

assert state.get_problem_size() == 6
assert state.get_connected_base_pairs() == [5, 0, 1, 2, 3, 4]
assert state.get_unconnected_base_pairs() == []
assert state.is_terminal()
print(state.evaluate())
