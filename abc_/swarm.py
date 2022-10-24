from abc_.food_source import FoodSource
from jsp.JSPInstance import JSPInstance
import jsp.graph_util as gu
import jsp.matrix_util as mu

from operator import attrgetter
import random as rand
import numpy as np
import copy
import random
import time

# from numba import jit


class ABC(object):
    food_sources = []

    def __init__(
            self,
            npopulation,
            nruns,
            trials_limit=20,
            employed_bees_percentage=0.5
    ):
        super(ABC, self).__init__()
        self.npopulation = npopulation
        self.nruns = nruns
        self.trials_limit = trials_limit

        # 采蜜的
        self.employed_bees = round(npopulation * employed_bees_percentage)
        # 旁观的
        self.onlooker_bees = npopulation - self.employed_bees

    def initialize(self):
        '''
        每个雇佣蜂初始化一个解
        :return:
        '''
        # 每个雇佣蜂一个蜜源
        self.food_sources = [self.create_foodsource() for i in range(self.employed_bees)]

    def create_foodsource(self):
        '''
        初始化一个解
        :return:
        '''
        sol, g_list, opIDsOnMchs, ms = JSPInstance.generate_rand_solution()
        fitness = 1 / (1 + ms)
        return FoodSource(sol, fitness, opIDsOnMchs, g_list)

    def best_source(self):
        '''
        获取最好的解，fitness最大
        :return:
        '''
        best = max(self.food_sources, key=attrgetter('fitness'))
        return best

    def selection(self, solutions, weights):
        # 加权随机选一个
        return rand.choices(solutions, weights)[0]

    def fitness(self, weights, g_list, opIDsOnMchs, groups):
        '''
        计算fitness
        :param weights 加工时间矩阵
        :param opIDsOnMchs 机器-操作矩阵
        :param solution_g_list: 邻接表
        :param groups: 分组索引
        :return:
        '''
        cps, mss = gu.get_cps_mss_by_group(weights, g_list, opIDsOnMchs, groups)
        if cps == -1:
            return -1
        ms = sum(mss)
        fitness = 1 / (1 + ms)
        return fitness

    def probability(self, solution_fitness):
        '''根据fitness占比计算下次在该解上生成新解的概率'''
        fitness_sum = sum([fs.fitness for fs in self.food_sources])
        probability = solution_fitness.fitness / fitness_sum

        return probability

    def update_solution(self, opids):
        '''
        根据机器-工序阵更新解序列
        :param opids:
        :return:
        '''
        sol = np.arange(JSPInstance.number_of_tasks).tolist()
        sol = np.array(mu.cut_in_line(sol, opids))[:, 0].tolist()
        return sol

    def generate_solution_n7(self, current_solution_index):
        '''
        n7生成新解
        :param current_solution_index:
        :return: 新的邻接表和机器-操作阵
        '''
        # 生成新解
        source = self.food_sources[current_solution_index]
        g_list = copy.deepcopy(source.g_list)
        opids = copy.deepcopy(source.opIDsOnMchs)

        cps, mss = gu.get_cps_mss_by_group(JSPInstance.dur, source.g_list, source.opIDsOnMchs, JSPInstance.groups)

        # 找各组的关键块
        cbss = []
        for cp in cps:
            cbss.append(gu.find_block(cp, g_list))

        critical_blocks = rand.choice(cbss)  # 随机选择一组的关键路径上的关键块

        # n7
        block = np.array(rand.choice(critical_blocks))
        mch = block[0][1]
        if len(block) == 2:
            # print('就俩 翻转')
            g_list = gu.invert(block[0][0], block[1][0], g_list, JSPInstance.last_col)
            opids = gu.invert_opid(block[0][0], block[1][0], opids, mch)
        elif len(block) > 2:
            before_list = block[:, 0]
            if rand.random() < 0.5:  # 将关键块头/尾移到中间
                if rand.random() < 0.5:  # 动头
                    g_list, opids = gu.start2mid(before_list, g_list, opids, mch)
                else:  # 动尾
                    g_list, opids = gu.end2mid(before_list, g_list, opids, JSPInstance.last_col, mch)
            else:  # 将中间移到头/尾
                if rand.random() < 0.5:  # 移到头
                    g_list, opids = gu.mid2start(before_list, g_list, opids, mch)
                else:  # 移到尾
                    g_list, opids = gu.mid2end(before_list, g_list, opids, JSPInstance.last_col, mch)
        return g_list, opids

    def fi(self, current_solution_index):
        '''
        first improve 对关键块内每相邻的两道工序调换顺序，若有提高，返回进行下一次循环
        :param current_solution_index:
        :return: 最后一次提高的新邻接表和机器-操作矩阵
        '''
        source = self.food_sources[current_solution_index]
        opids = source.opIDsOnMchs.copy()
        g_list = source.g_list.copy()
        fitness_new = source.fitness

        # 对每个组进行一次fi
        for group in range(len(JSPInstance.groups)):
            for i in range(3):
                improved = True
                while improved:
                    cps, mss = gu.get_cps_mss_by_group(JSPInstance.dur, g_list, opids, JSPInstance.groups)
                    ms = sum(mss)

                    # 获取每组的关键块
                    cbss = []
                    for cp in cps:
                        cbss.append(gu.find_block(cp, g_list))
                    critical_blocks = cbss[group]

                    if len(critical_blocks) == 0: break
                    improved = False
                    for block in critical_blocks:
                        if len(block) < 2:
                            continue

                        mch = block[0][1]
                        g_list_backup = copy.deepcopy(g_list)
                        opids_backup = copy.deepcopy(opids)

                        if i == 0:
                            v1 = block[0][0]
                            v2 = block[1][0]
                            g_list = gu.invert(v1, v2, g_list, JSPInstance.last_col)
                            opids = gu.invert_opid(v1, v2, opids, mch)

                        elif i == 1:
                            v1 = block[-2][0]
                            v2 = block[-1][0]
                            g_list = gu.invert(v1, v2, g_list, JSPInstance.last_col)
                            opids = gu.invert_opid(v1, v2, opids, mch)
                        else:
                            rand = random.randint(0, len(block) - 2)
                            v1 = block[rand][0]
                            v2 = block[rand + 1][0]
                            g_list = gu.invert(v1, v2, g_list, JSPInstance.last_col)
                            opids = gu.invert_opid(v1, v2, opids, mch)

                        cps_new, mss_new = gu.get_cps_mss_by_group(JSPInstance.dur, g_list, opids, JSPInstance.groups)

                        if not mss_new == -1 and sum(mss_new) < ms:
                            ms = sum(mss_new)
                            fitness_new = 1 / (ms + 1)
                            improved = True
                            break

                        else:
                            g_list = g_list_backup.copy()
                            opids = opids_backup.copy()

        return g_list, opids, fitness_new

    def best_solution_n7(self, current_solution_g_list, new_solution_g_list, current_opids, new_opids):
        '''
        比较n7生成新解和现有解
        :param current_solution_g_list: 现有邻接表
        :param new_solution_g_list: 新邻接表
        :param current_opids: 现有机器-操作阵
        :param new_opids: 新机器-操作阵
        :return: fitness更大解 （邻接表，机器-操作阵，fitness，是否提升）
        '''
        fitness = self.fitness(JSPInstance.dur, current_solution_g_list, current_opids, JSPInstance.groups)
        fitness_new = self.fitness(JSPInstance.dur, new_solution_g_list, new_opids, JSPInstance.groups)
        if fitness_new > fitness:
            return new_solution_g_list, new_opids, fitness_new, True
        else:
            return current_solution_g_list, current_opids, fitness, False

    def set_solution_fi(self, food_source, new_g_list, new_opids, fitness_new):
        '''
        更新fi提升后的解
        :param food_source:
        :param new_g_list:
        :param new_opids:
        :param fitness_new:
        :return:
        '''

        # 在这边根据opids和g_list更新solution
        food_source.fitness = fitness_new
        food_source.g_list = new_g_list
        food_source.opIDsOnMchs = new_opids
        food_source.solution = self.update_solution(new_opids)
        food_source.trials = 0

    def set_solution_n7(self, food_source, new_g_list, new_opids, fitness_new, improved):
        '''
        更新n7的解
        :param food_source:
        :param new_g_list:
        :param new_opids:
        :param fitness_new:
        :param improved:
        :return:
        '''
        # 在这边根据opids和g_list更新solution
        if improved:
            food_source.fitness = fitness_new
            food_source.g_list = new_g_list
            food_source.opIDsOnMchs = new_opids
            food_source.solution = self.update_solution(new_opids)
            food_source.trials = 0
        else:
            food_source.trials += 1

    def employed_bees_stage(self):
        # 对每一只雇佣蜂，生成并选出当前最优解

        for i in range(self.employed_bees):
            food_source = self.food_sources[i]
            '''底下两行给fi'''
            # g_list_new, opids_new, fitness_new = self.fi(i)
            # self.set_solution(food_source, g_list_new, opids_new, fitness_new)
            '''底下三行给n7'''
            g_list_new, opids_new = self.generate_solution_n7(i)
            g_list_best, opids_best, fitness_best, improved = self.best_solution_n7(food_source.g_list, g_list_new, food_source.opIDsOnMchs, opids_new)
            # 如果没变，trials+1 尝试替换的次数，到达一定量丢弃
            self.set_solution_n7(food_source, g_list_best, opids_best, fitness_best, improved)

    def onlooker_bees_stage(self):
        # 对每一只旁观者，根据fitness权重，随机选一个解生成新解，看是否最优
        for i in range(self.onlooker_bees):
            probabilities = [self.probability(fs) for fs in self.food_sources] # 算权重
            selected_index = self.selection(range(len(self.food_sources)), probabilities)
            selected_source = self.food_sources[selected_index]
            '''底下两行给fi'''
            # g_list_new, opids_new, fitness_new = self.fi(selected_index)
            # self.set_solution_fi(selected_source, g_list_new, opids_new, fitness_new)
            '''底下三行给n7'''
            g_list_new, opids_new = self.generate_solution_n7(selected_index)
            g_list_best, opids_best, fitness_best, improved = self.best_solution_n7(selected_source.g_list, g_list_new, selected_source.opIDsOnMchs,
                                                            opids_new)
            # 如果没变，trials+1 尝试替换的次数，到达一定量丢弃
            self.set_solution_n7(selected_source, g_list_best, opids_best, fitness_best, improved)

    def scout_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            if food_source.trials > self.trials_limit:
                food_source = self.create_foodsource()

    # @jit
    def optimize(self):
        t = time.time()
        self.initialize()
        best_fs = self.best_source()
        min_ms = sum(gu.get_cps_mss_by_group(JSPInstance.dur, best_fs.g_list, best_fs.opIDsOnMchs,
                                             JSPInstance.groups)[1])
        print(min_ms)
        print('初始ms ', )
        print('initialize ', time.time() - t)

        for nrun in range(1, self.nruns+1):
            t = time.time()
            self.employed_bees_stage()
            self.onlooker_bees_stage()
            self.scout_bees_stage()
            best_fs = self.best_source()
            min_ms = sum(gu.get_cps_mss_by_group(JSPInstance.dur, best_fs.g_list, best_fs.opIDsOnMchs,
                                                 JSPInstance.groups)[1])
            print(min_ms)
            print('一轮循环，', time.time() - t)

        best_fs = self.best_source()
        print('一个解：', time.time()-t)

        return best_fs
