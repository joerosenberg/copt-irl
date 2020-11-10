import matplotlib.pyplot as plt
import numpy as np
from math import factorial

fig, axs = plt.subplots(2, 2, sharey=True)
axs = axs.flatten()
colors = ['r', 'b', 'g', 'k']

for j, problem_size in enumerate([6, 7, 8, 9]):
    nb_solutions_file = open(f'./nb_solutions_for_size_{problem_size}.csv', 'r')
    measures_file = open(f'./measures_for_size_{problem_size}.csv', 'r')

    nb_solutions = np.array([int(line) for line in nb_solutions_file.read().splitlines()])
    measures = np.array([float(line) for line in measures_file.read().splitlines()])

    stds = np.zeros_like(nb_solutions)
    ranges = np.zeros_like(nb_solutions)

    measure_index = 0
    for i, nb_solution in enumerate(nb_solutions):
        if nb_solution != 0:
            stds[i] = np.std(measures[measure_index: measure_index + nb_solution])
            ranges[i] = np.ptp(measures[measure_index: measure_index + nb_solution])
            measure_index += nb_solution

    axs[j].scatter(nb_solutions / factorial(problem_size), ranges, label=f'Problems of size {problem_size}',
                   color=colors[j], s=1.0)
    axs[j].legend()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
fig.text(0.5, 0.02, "Number of valid solutions as proportion of search space", ha='center')
fig.text(0.02, 0.5, "Range of measures for valid solutions", va='center', rotation='vertical')
fig.savefig(f'./range.png')