import random
import copy
import numpy as np

from jsp.matrix_util import getActionNbghs
import jsp.graph_util as gu


class JSPInstance:

    release_time = None  # 工件释放时间
    data = None  # data[0] = dur, data[1] = m
    groups = None  # 分组索引列表
    dur = None  # 加工时间矩阵
    release_dur = None  # 带释放时间的加工时间矩阵
    m = None  # 机器阵
    number_of_machines = None  # 机器数
    number_of_jobs = None  # 工件数
    number_of_tasks = None  # 总工序 = 机器 * 工件
    release_col = None  # 释放节点的索引
    first_col = None  # 第一道工序的索引
    last_col = None  # 最后一道工序的索引
    g_list = None  # 邻接矩阵
    opIDsOnMchs = None  # 机器-工序矩阵
    ms = None  # makespan

    @classmethod
    def reset(cls, data, groups, release_time):
        '''
        初始化分组信息，邻接表，机器-工序矩阵
        :param release_time:
        :param groups:
        :param data:
        :return:
        '''
        cls.release_time = release_time.reshape(-1,1)
        cls.data = data
        cls.m = data[-1]  # 机器约束矩阵
        cls.dur = data[0].astype(np.single)  # 加工时间约束矩阵
        cls.release_dur = np.append(cls.release_time, cls.dur, axis=1)

        cls.number_of_jobs = cls.dur.shape[0]
        cls.number_of_machines = cls.dur.shape[1]
        cls.number_of_tasks = cls.number_of_jobs * cls.number_of_machines

        cls.release_col = np.arange(start=0, stop=cls.number_of_tasks+cls.number_of_jobs, step=1).reshape(cls.number_of_jobs, -1)[:, 0]
        cls.first_col = np.arange(start=0, stop=cls.number_of_tasks+cls.number_of_jobs, step=1).reshape(cls.number_of_jobs, -1)[:, 1]
        cls.last_col = np.arange(start=0, stop=cls.number_of_tasks+cls.number_of_jobs, step=1).reshape(cls.number_of_jobs, -1)[:, -1]

        # 初始化邻接表
        g_list = []
        for i in range(cls.number_of_tasks+cls.number_of_jobs):
            succeeds = []
            if i not in cls.last_col:
                succeeds.append(i + 1)
            g_list.append(succeeds)
        cls.g_list = g_list

        # 初始化机器-操作矩阵
        cls.opIDsOnMchs = -cls.number_of_jobs * np.ones_like(cls.dur.transpose(), dtype=np.int32)

        cls.groups = groups

    @classmethod
    def generate_rand_solution(cls):
        '''
        生成随机解，循环随机获取工序，并更新邻接表和机器-工序矩阵
        :return:
        '''
        cls.reset(cls.data, cls.groups, cls.release_time)

        sol = []
        indices = [i for i in range(cls.number_of_jobs)]

        candidate = copy.deepcopy(cls.first_col)

        # 随机采取可行的工序
        for i in range(cls.number_of_tasks):
            rand_idx = random.choice(indices)
            action = candidate[rand_idx]
            sol.append(action)

            # 更新candidate
            if action not in cls.last_col:
                candidate[action // (cls.number_of_machines+1)] += 1
            # 更新邻接表和机器-工序矩阵
            cls.update_by_action(action)

            if action % (cls.number_of_machines+1) == cls.number_of_machines:  # 最后一道工序
                indices.remove(rand_idx)

        # 邻接表和opid都有了，获取各组的关键路径和分组最大的ms
        cls.cps, mss = gu.get_cps_mss_by_group(cls.release_dur, cls.g_list, cls.opIDsOnMchs, cls.groups)
        cls.ms = sum(mss)  # 总makespan取和
        return sol, cls.g_list, cls.opIDsOnMchs, cls.ms

    @classmethod
    def update_by_action(cls, action):
        '''
        主要更新邻接表和机器-操作矩阵
        :param action:
        :return:
        '''

        row = action // (cls.number_of_machines + 1)
        col = action % (cls.number_of_machines + 1)

        mch_a = cls.m[row][col-1] - 1  # 该工序操作的机器
        mch_situation = cls.opIDsOnMchs[mch_a].tolist()  # 该机器上当前已完工的工件
        idx = mch_situation.index(-cls.number_of_jobs)
        cls.opIDsOnMchs[mch_a][idx] = action

        precd, succd = getActionNbghs(action, cls.opIDsOnMchs)
        # 删除action的所有前驱
        for succeeds in cls.g_list:
            if action in succeeds:
                succeeds.remove(action)
        #  因为添加了释放节点，所以也要有释放节点->第一道工序的边
        #if action not in cls.first_col:
        cls.g_list[action - 1].insert(0, action)
        if precd != action:
            if precd == action - 1:  # 保证黑边是后继表的第一个元素
                cls.g_list[precd].insert(0, action)
            else:
                cls.g_list[precd].append(action)
        if succd != action:
            if succd == action + 1:  # 保证黑边是后继表的第一个元素
                cls.g_list[action].insert(0, succd)
            else:
                cls.g_list[action].append(succd)

        # 不插队
        # if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
        #     self.g_list[precd].remove(succd)
