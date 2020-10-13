import matplotlib.pyplot as plt
import numpy as np
from math import factorial

for problem_size in [6, 7, 8, 9]:
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

    plt.figure()
    plt.scatter(nb_solutions, stds)
    plt.xlabel('Number of solutions')
    plt.ylabel('Standard deviation of measure of all solutions')
    plt.savefig(f'./variance_vs_nb_solutions_for_size_{problem_size}.png')

    plt.figure()
    plt.scatter(nb_solutions, ranges)
    plt.xlabel('Number of solutions')
    plt.ylabel('Range of measure across all solutions')
    plt.savefig(f'./range_vs_nb_solutions_for_size_{problem_size}.png')

    plt.figure()
    plt.scatter(nb_solutions / factorial(problem_size), stds)
    plt.xlabel('Proportion of valid solutions')
    plt.ylabel('Standard deviation of measure of all solutions')
    plt.savefig(f'./variance_vs_proportion_valid_for_size_{problem_size}.png')

    plt.figure()
    plt.scatter(nb_solutions / factorial(problem_size), ranges)
    plt.xlabel('Proportion of valid solutions')
    plt.ylabel('Range of measure across all solutions')
    plt.savefig(f'./range_vs_proportion_valid_for_size_{problem_size}.png')

