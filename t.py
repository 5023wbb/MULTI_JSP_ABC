import argparse

import numpy as np
from jsp.JSPInstance import JSPInstance
from jsp.matrix_util import uni_instance_gen
from abc_.swarm import ABC
from jsp import graph_util as gu
from jsp import matrix_util as mu

parser = argparse.ArgumentParser(description='Arguments for test_learned_on_benchmark')
parser.add_argument('--Gn', type=int, default=3, help='Number of groups')
parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs on which to be loaded net are trained')
parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
parser.add_argument('--which_benchmark', type=str, default='tai', help='Which benchmark to test')
parser.add_argument('--low', type=int, default=1, help='LB of duration')
parser.add_argument('--high', type=int, default=99, help='UB of duration')
parser.add_argument('--np_seed_validation', type=int, default=200, help='Seed for numpy for validation')
parser.add_argument('--rts', type=int, default=200, help='Seed for release time')
params = parser.parse_args()

benchmark = params.which_benchmark
N_JOBS_N = params.Nn_j
N_MACHINES_N = params.Nn_m
G_N = params.Gn
LOW = params.low
HIGH = params.high
RELEASE_TIME_SEED = params.rts

if __name__ == '__main__':
    # data = uni_instance_gen(n_j=N_JOBS_N, n_m=N_MACHINES_N, low=LOW, high=HIGH)

    groups = mu.get_groups(G_N, N_JOBS_N)

    dataset = []
    # dataset.append(uni_instance_gen(n_j=N_JOBS_N, n_m=N_MACHINES_N, low=LOW, high=HIGH))

    # dataGen里的数据
    # dataLoaded = np.load('./DataGen/generatedData' + str(params.Nn_j) + '_' + str(params.Nn_m) + '_Seed' + str(
    #     params.np_seed_validation) + '.npy')

    # BenchData数据
    dataLoaded = np.load('./BenchDataNmpy/' + benchmark + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '.npy')
    # dataset = []

    for i in range(dataLoaded.shape[0]):
        dataset.append((dataLoaded[i][0], dataLoaded[i][1]))

    for i, data in enumerate(dataset):
        np.random.seed(RELEASE_TIME_SEED)
        release_time = np.random.randint(30 * N_JOBS_N, size=N_JOBS_N)
        release_time[np.random.randint(N_JOBS_N)] = 0
        JSPInstance.reset(data, groups, release_time)
        print(JSPInstance.m)
        abc = ABC(10, 10)
        best_solution = abc.optimize()
        print('OPIDS\n', best_solution.opIDsOnMchs)
        cps, mss = gu.get_cps_mss_by_group(JSPInstance.release_dur, best_solution.g_list, best_solution.opIDsOnMchs, JSPInstance.groups)
        min_ms = sum(mss)

        print('最终ms，', min_ms)
