import matplotlib.pyplot as plt
import numpy as np
from math import factorial

fig, axs = plt.subplots(2, 2, sharey=True)
axs = axs.flatten()
colors = ['r', 'b', 'g', 'k']

for i, problem_size in enumerate([6, 7, 8, 9]):
    nb_solutions_file = open(f'./nb_solutions_for_size_{problem_size}.csv', 'r')
    nb_solutions = np.array([int(line) for line in nb_solutions_file.read().splitlines()])

    axs[i].hist(nb_solutions / factorial(problem_size), label=f'Problems of size {problem_size}', color=colors[i],
                ec='w')
    axs[i].legend()

# add hidden axis
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Number of valid solutions as proportion of search space")
plt.ylabel("Number of problem instances")
fig.savefig(f'./valid_solns_hist.png')
