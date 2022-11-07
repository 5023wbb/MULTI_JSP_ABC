import numpy as np
import random


def getActionNbghs(action, opIDsOnMchs):
    '''
    获取工序在机器矩阵上的邻居(非黑边)
    :param action: 工序
    :param opIDsOnMchs: 机器矩阵
    :return: 工序在同一机器的前驱后继(可能包括自己)
    '''
    coordAction = np.where(opIDsOnMchs == action)
    precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()
    succdTemp = opIDsOnMchs[
        coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[
            1]].item()
    succd = action if succdTemp < 0 else succdTemp
    return precd, succd


def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    '''
    随机生成加工时间阵和机器矩阵
    :param n_j: 工件数
    :param n_m: 机器数
    :param low: 加工时间下限
    :param high: 加工时间上限
    :return:
    '''
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


def cut_in_line(partial_solution, opIDsOnMchs):
    """
    根据机器-操作矩阵生成一个对应的action序列
    :param partial_solution: 未整合机器信息的action序列
    :param opIDsOnMchs: 各machine上op的顺序
    :return: 整合插队信息的action序列 [[op: mch],[op: mch]...]
    """
    for i in opIDsOnMchs:
        temp_index = len(partial_solution)
        for action in i:
            action_index = partial_solution.index(action)
            if action_index < temp_index:
                temp_index = action_index
            else:  # 有插队情况，复制opIDsOnMchs当前数组到partial_solution中
                index_list = []
                for action_ in i:
                    index_list.append(partial_solution.index(action_))
                index_list.sort()
                for j in range(len(index_list)):
                    partial_solution[index_list[j]] = i[j]
                break
    initial_sequence = np.array(partial_solution).reshape(len(opIDsOnMchs[0]), len(opIDsOnMchs))
    for i in initial_sequence:
        temp_index = len(partial_solution)
        for action in i:
            action_index = partial_solution.index(action)
            if action_index < temp_index:
                temp_index = action_index
            else:  # 违反工序约束
                index_list = []
                for action_ in i:
                    index_list.append(partial_solution.index(action_))
                index_list.sort()
                for j in range(len(index_list)):
                    partial_solution[index_list[j]] = i[j]
                break
    op_mch = []
    for action in partial_solution:
        op_mch.append([action, np.where(opIDsOnMchs == action)[0][0]])
    return op_mch


def get_groups(group_num, number_of_jobs):
    '''
    初始化分组
    :param group_num: 分组数
    :param number_of_jobs: 工件数量
    :return: 分组索引列表
    '''

    indices = [i for i in range(number_of_jobs)]
    groups = np.array_split(indices, group_num)
    return groups


def id2string(task_id, machine_num):
    job = task_id // (machine_num + 1) + 1
    operation = task_id % (machine_num + 1)
    return 'J_' + str(job) + '_O_' + str(operation)

def opids2String(opids):
    opids_string = []
    for row in opids:
        row_string = []
        for opid in row:
            row_string.append(id2string(opid, len(opids)))
        opids_string.append(row_string)
    return opids_string



if __name__ == '__main__':
    opIDsOnMchs = np.array([[7, 29, 33, 16, -6, -6],
                            [6, 18, 28, 34, 2, -6],
                            [26, 31, 14, 21, 11, 1],
                            [30, 19, 27, 13, 10, -6],
                            [25, 20, 9, 15, -6, -6],
                            [24, 12, 8, 32, 0, -6]])
    # action = 33
    # precd, succd = getActionNbghs(action, opIDsOnMchs)
    # print(precd, succd)

    print(id2string(1, 15))
    # print(opids2String(opIDsOnMchs))
