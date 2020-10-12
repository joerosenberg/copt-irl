from multiprocessing import Process, Queue, freeze_support
from tqdm import tqdm
import copt


def brute_force_mp(problem, queue):
    solns = copt.bruteForce(problem)
    queue.put(solns)


if __name__ == '__main__':
    freeze_support()
    queue = Queue()
    for problem_size in [6, 7, 8, 9]:
        for i in tqdm(range(100)):
            # Generate valid problem
            while True:
                problem = copt.getProblem(problem_size)
                for j in range(problem_size - 1):
                    start_sq_dist = (problem[j][0] - problem[j+1][0])**2 + (problem[j][1] - problem[j+1][1])**2
                    end_sq_dist = (problem[j][2] - problem[j+1][2])**2 + (problem[j][3] - problem[j+1][3])**2
                    if start_sq_dist < 45**2 or end_sq_dist < 45**2:
                        print(start_sq_dist, end_sq_dist)
                        continue
                break

            process = Process(target=brute_force_mp, args=(problem, queue))
            process.start()
            solns = queue.get()
            process.join()

            with open(f'./copt_analysis/clearance45_nb_solutions_for_size_{problem_size}.csv', 'a') as nb_solns_file:
                nb_solns_file.write(f'{len(solns)}\n')

            with open(f'./copt_analysis/clearance45_measures_for_size_{problem_size}.csv', 'a') as measures_file:
                measures_file.writelines([str(soln['measure']) + '\n' for soln in solns])