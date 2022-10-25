import copy
import numpy as np
import random as rand
import jsp.matrix_util as mu


def top_sort(g):
    '''
    拓扑排序
    :param g: 邻接表
    :return: 拓扑排序序列，有无环
    '''
    post_order = []  # 后序遍历结果存储
    cycle_flags = []  # 存储每次递归是否有环
    visited = [False] * len(g)
    on_path = [False] * len(g)
    has_cycle = False
    for i in range(len(g)):
        traverse(g, i, visited, on_path, has_cycle, post_order, cycle_flags)
    post_order.reverse()
    return post_order, True in cycle_flags


def traverse(g, s, visited, on_path, has_cycle, post_order, cycle_flags):
    '''
    拓扑排序的递归辅助函数
    '''
    if on_path[s]:
        has_cycle = True
        cycle_flags.append(has_cycle)
        print('发现环！')

    if visited[s] or has_cycle:
        return

    visited[s] = True
    on_path[s] = True
    for t in g[s]:
        traverse(g, t, visited, on_path, has_cycle, post_order, cycle_flags)
    post_order.append(s)

    on_path[s] = False


def get_critical_path_and_makespan_topSort(topStack, weights, opIDsOnMchs, g):
    '''
    根据拓扑排序栈，processing_time阵计算关键路径
    :param topStack: 拓扑排序栈
    :param weights: processing_time阵
    :return: 关键路径，对应的ms，最早开始时间序列(etv)
    '''

    etv = {}  # 最早开始时间

    for i in topStack:
        row = i // (len(weights[0]))
        col = i % (len(weights[0]))
        # 找i的前驱
        preds = []
        for jj in range(len(g)):
            if i in g[jj]:
                preds.append(jj)

        # 如果无前驱，为起点
        if len(preds) == 0:
            etv[i] = weights[row][col]
        # 有前驱，遍历前驱，取max(前驱的etv+当前节点的weight)
        else:
            max_time = -1
            for pred in preds:
                max_time = etv[pred] + weights[row][col] if max_time < etv[pred] + weights[row][
                    col] else max_time
            etv[i] = max_time

    ltv = {}  # 最晚开始时间, 所有值初始化为etv的最大值
    max_val = max(etv.values())
    for key in etv:
        ltv[key] = max_val

    for i in reversed(topStack):
        # 找i的后继
        suceeds = g[i]
        # 如果无后继，为终点
        if len(suceeds) == 0:
            continue
        # 有后继，遍历后继，取min(后继的ltv-后继节点的weight)
        else:
            min_time = max_val
            for suceed in suceeds:
                row = suceed // (len(weights[0]))
                col = suceed % (len(weights[0]))
                min_time = ltv[suceed] - weights[row][col] if min_time > ltv[suceed] - weights[row][
                    col] else min_time
            ltv[i] = min_time

    critical_path = []

    for key in etv:
        if etv[key] == ltv[key]:
            if not key % (len(opIDsOnMchs)+1) == 0:
                critical_path.append([key, np.where(opIDsOnMchs == key)[0][0]])
            # critical_path.append(key)
    return critical_path, max_val, etv


def find_block(critical_path, g_list):
    '''
    返回关键块
    :param critical_path: 关键路径
    :param g_list: 邻接表
    :return: 关键块 [[[action0, machine0], [action1, machine0]],[[action2, machine1], [action3, machine1]]]
    '''
    '''这边不是只要连在一块就行的，比如
     关键路径 13,14,16,18
     opid:  13 14 15 16 18
     关键块要分 [13, 14] 和 [16, 18]
     '''
    blocks = []
    temp = -1
    for node in critical_path:
        # 在同一机器
        if node[1] == temp:
            # 且有连边
            if node[0] in g_list[blocks[-1][-1][0]]:
                blocks[-1].append(node)
            else:
                blocks.append([node])
        else:
            blocks.append([node])
        temp = node[1]
    return blocks


def get_cps_mss_by_group(weights, g_list, opIDsOnMchs, groups):
    '''
    返回各组的关键路径和 makespan ([cp1, cp2, cp3], [ms1, ms2, ms3])
    :param weights: 加工时间矩阵
    :param g_list: 邻接表
    :param opIDsOnMchs: 机器-工序矩阵
    :param groups: 分组的索引[[1,2], [3,4]]
    :return: 各组的关键路径和 makespan ([cp1, cp2, cp3], [ms1, ms2, ms3]), 有环的话返回-1, -1
    '''
    cps = []
    mss = []
    top_stack, has_cycle = top_sort(g_list)
    if has_cycle:
        return -1, -1
    cp0, ms0, etv0 = get_critical_path_and_makespan_topSort(top_stack, weights, opIDsOnMchs, g_list)
    cps.append(cp0)
    mss.append(ms0)
    for group in groups:
        if cp0[-1][0] // (len(opIDsOnMchs[0])+1) in group:
            continue
        else:
            g = copy.deepcopy(g_list)
            max_etv = -1
            max_end = -1
            # 获取该组ms最大的最后一道工序
            for idx in group:
                end = idx * (len(opIDsOnMchs)+1) + len(opIDsOnMchs)
                if max_etv < etv0[end]:
                    max_end = end
                    max_etv = etv0[end]
            # print(max_end, max_etv)
            # print('删前：', g)
            # 删图, 将etv中大于当前max_etv的工序的前驱删除
            for k, v in etv0.items():
                if v >= max_etv and not k == max_end:
                    for suceds in g:
                        if k in suceds:
                            suceds.remove(k)
            # print('删后g：', g)
            top_stack_, has_cycle = top_sort(g)
            cp, ms, _ = get_critical_path_and_makespan_topSort(top_stack_, weights, opIDsOnMchs, g)
            cps.append(cp)
            mss.append(ms)
    return cps, mss


def get_not_black_suced(g_list, node, last_col):
    '''
    获取节点非黑边后继，如果没有返回-1
    '''
    suceds = g_list[node]
    if node in last_col:  # 如果是最后一道工序
        if len(suceds) > 0:
            return suceds[0]
        return -1
    else:
        if len(suceds) > 1:
            return suceds[1]
        return -1


def get_not_black_pred(g_list, node):
    '''
    获取节点非黑边前驱，如果没有返回-1
    :param g_list:
    :param node:
    :return:
    '''
    not_black_suced = -1
    for i in range(len(g_list)):
        if node in g_list[i] and not node == i + 1:
            not_black_suced = i
    return not_black_suced


def invert(v1, v2, g_list, last_col):
    '''
    在邻接表上翻转两道工序
    :param v1: 工序1
    :param v2: 工序2
    :param g_list: 原邻接表
    :param last_col: 最后一列工序(静态)
    :return: 翻转后的邻接表
    '''
    not_black_pred = get_not_black_pred(g_list, v1)  # 获取v1的非黑边前驱
    not_black_suced = get_not_black_suced(g_list, v2, last_col)  # 获取v2的非黑边后继

    ### v1操作
    g_list[v1].remove(v2) # v1的后继删除v2
    # 如果v2有非黑边后继
    if not not_black_suced == -1:
        g_list[v2].remove(not_black_suced) # v2的后继表删除原非黑边后继
        g_list[v1].append(not_black_suced) # v1的后继表加上该非黑边后继

    ### v2操作
    g_list[v2].append(v1)  # v2的后继加上v1
    # 如果v1有非黑边前驱
    if not not_black_pred == -1:
        g_list[not_black_pred].remove(v1)  # v1的非黑边前驱后继表删除v1
        g_list[not_black_pred].append(v2)  # v1的非黑边前驱后继表加上v2

    return g_list


def invert_opid(v1, v2, opids, mch):
    '''
    在机器-工序矩阵调换两工序位置
    :param v1: 工序1
    :param v2: 工序2
    :param opids: 机器-工序矩阵
    :param mch: 机器
    :return: 调换位置后的机器-工序矩阵
    '''
    idx1 = np.where(opids[mch] == v1)
    idx2 = np.where(opids[mch] == v2)
    opids[mch][idx1], opids[mch][idx2] = opids[mch][idx2], opids[mch][idx1]
    return opids


# n7相关操作

def start2mid(before_list, g_list, opids, mch):
    '''
    将mch上的开头工序调到中间
    :param before_list: 原来该机器上的工序排列
    :param g_list: 原邻接表
    :param opids: 原机器-工序矩阵
    :param mch: 调换工序的机器
    :return: 调换后的邻接表和机器-工序矩阵
    '''
    to_move = before_list[0]  # 头元素移到中间
    before_list = np.delete(before_list, 0)
    insert_idx = rand.randint(1, len(before_list) - 1)
    # print('把头元素'+str(to_move)+'移到中间'+str(insert_idx)+'的位置')
    after_list = np.insert(before_list, insert_idx, to_move)

    ### 原头元素操作
    not_black_pred = get_not_black_pred(g_list, to_move)
    # print('非黑边前驱:', not_black_pred)

    # 如果原头元素有非黑边前驱，该非黑边前驱的后继表删除原头元素
    if not not_black_pred == -1:
        g_list[not_black_pred].remove(to_move)
    g_list[to_move].remove(before_list[0])  # 删除该元素的后继
    g_list[to_move].append(before_list[insert_idx])  # 该元素的后继加上插入后前面元素的原后继

    ### 插入后前面元素操作
    g_list[before_list[insert_idx - 1]].append(to_move)  # 该元素插入后 前面的元素的后继加上该元素
    g_list[before_list[insert_idx - 1]].remove(before_list[insert_idx])  # 该元素插入后 前面的元素 删除原后继

    ### 新头元素操作
    if not not_black_pred == -1:  # 如果原头元素有非黑边前驱
        # 该非黑边前驱的后继加上新头元素
        g_list[not_black_pred].append(before_list[0])

    # 交换opids 整合opids的插队信息生成新解
    idx = 0
    job_queue = opids[mch]
    for i in range(len(job_queue)):
        if job_queue[i] in after_list:
            job_queue[i] = after_list[idx]
            idx = idx + 1
        if idx == len(after_list):
            break
    opids[mch] = job_queue

    return g_list, opids


def end2mid(before_list, g_list, opids, last_col, mch):
    '''
    将mch上的最后工序调到中间
    :param before_list: 原来该机器上的工序排列
    :param g_list: 原邻接表
    :param opids: 原机器-工序矩阵
    :param last_col: 最后一列工序索引(静态)
    :param mch: 调换工序的机器
    :return: 调换后的邻接表和机器-工序矩阵
    '''
    to_move = before_list[-1]
    before_list = np.delete(before_list, -1)
    insert_idx = rand.randint(1, len(before_list) - 1)
    # print('把尾元素' + str(to_move) + '移到中间' + str(insert_idx) + '的位置')
    after_list = np.insert(before_list, insert_idx, to_move)

    ### 原尾元素操作
    not_black_suced = get_not_black_suced(g_list, to_move, last_col)
    # print('非黑边后继:', not_black_suced)

    # 如果该元素有非黑边后继的其他后继，删了
    if not not_black_suced == -1:
        g_list[to_move].remove(not_black_suced)
    g_list[to_move].append(before_list[insert_idx])  # 该元素的后继加上前面元素的原后继

    ### 插入后的前面元素操作
    g_list[before_list[insert_idx - 1]].remove(before_list[insert_idx])  # 该元素插入后 前面的元素 删除原后继
    g_list[before_list[insert_idx - 1]].append(to_move)  # 该元素插入后 前面的元素的后继加上该元素

    ### 插入后的后面元素(新尾元素)操作
    g_list[before_list[-1]].remove(to_move)  # 现在尾元素的后继删除该元素
    if not not_black_suced == -1:  # 如果原尾元素有非黑边后继
        g_list[before_list[-1]].append(not_black_suced)  # 新尾元素加上该非黑边后继

    # 交换opids 整合opids的插队信息生成新解
    idx = 0
    job_queue = opids[mch]
    for i in range(len(job_queue)):
        if job_queue[i] in after_list:
            job_queue[i] = after_list[idx]
            idx = idx + 1
        if idx == len(after_list):
            break
    opids[mch] = job_queue

    return g_list, opids


def mid2start(before_list, g_list, opids, mch):
    '''
    将mch上的中间工序调到开头
    :param before_list: 原来该机器上的工序排列
    :param g_list: 原邻接表
    :param opids: 原机器-工序矩阵
    :param mch: 调换工序的机器
    :return: 调换后的邻接表和机器-工序矩阵
    '''
    to_move_index = rand.randint(1, len(before_list) - 2)
    to_move = before_list[to_move_index]
    after_list = np.delete(before_list, to_move_index)
    after_list = np.insert(after_list, 0, to_move)

    not_black_pred = get_not_black_pred(g_list, before_list[0])  # 获取原头元素的非黑边前驱和后继
    # print('非黑边前驱:', not_black_pred)

    ### 新头元素操作
    if not not_black_pred == -1:  # 如果原头元素有非黑边前驱
        g_list[not_black_pred].remove(before_list[0])  # 该非黑边前驱的后继表删除原头元素
        g_list[not_black_pred].append(to_move)  # 该非黑边前驱的后继表加上新头元素
    g_list[to_move].append(before_list[0])  # 新头元素后继表加上原头元素
    g_list[to_move].remove(before_list[to_move_index + 1])  # 该元素后继删除原后继

    # 该元素原前驱操作
    g_list[before_list[to_move_index - 1]].append(before_list[to_move_index + 1])  # 原前驱加上该元素的原后继
    g_list[before_list[to_move_index - 1]].remove(to_move)  # 原前驱删除该元素

    # 交换opids 整合opids的插队信息生成新解
    idx = 0
    job_queue = opids[mch]
    for i in range(len(job_queue)):
        if job_queue[i] in after_list:
            job_queue[i] = after_list[idx]
            idx = idx + 1
        if idx == len(after_list):
            break
    opids[mch] = job_queue

    return g_list, opids


def mid2end(before_list, g_list, opids, last_col, mch):
    '''
    将mch上的中间工序调到最后
    :param before_list: 原来该机器上的工序排列
    :param g_list: 原邻接表
    :param opids: 原机器-工序矩阵
    :param last_col: 最后一列工序索引(静态)
    :param mch: 调换工序的机器
    :return: 调换后的邻接表和机器-工序矩阵
    '''
    to_move_index = rand.randint(1, len(before_list) - 2)
    to_move = before_list[to_move_index]
    after_list = np.delete(before_list, to_move_index)
    after_list = np.append(after_list, to_move)

    not_black_suced = get_not_black_suced(g_list, before_list[-1], last_col)  # 获取原尾元素的非黑边前驱和后继
    # print('非黑边后继:', not_black_suced)

    ### 新尾元素操作
    if not not_black_suced == -1:  # 如果原尾元素存在非黑边后继
        g_list[before_list[-1]].remove(not_black_suced)  # 原尾元素后继表删除非黑边后继
        g_list[to_move].append(not_black_suced)  # 新尾元素后继表加上该非黑边后继
    g_list[to_move].remove(before_list[to_move_index + 1])  # 新尾元素后继删除原后继

    ### 原尾元素操作
    g_list[before_list[-1]].append(to_move)  # 原尾元素后继表加上该元素

    ### 原前驱操作
    g_list[before_list[to_move_index - 1]].append(before_list[to_move_index + 1])  # 原前驱加上该元素的后继
    g_list[before_list[to_move_index - 1]].remove(to_move)  # 原前驱后继表删除该元素

    # 交换opids 整合opids的插队信息生成新解
    idx = 0
    job_queue = opids[mch]
    for i in range(len(job_queue)):
        if job_queue[i] in after_list:
            job_queue[i] = after_list[idx]
            idx = idx + 1
        if idx == len(after_list):
            break
    opids[mch] = job_queue

    return g_list, opids


def adj_to_g(adj):
    '''
    邻接矩阵转化成邻接表
    :param adj:
    :return:
    '''
    g = []
    for i in range(len(adj)):
        adj[i][i] = 0
    for i in range(len(adj)):
        col = adj[:, i]
        succeeds = np.where(col == 1)[0]
        g.append(succeeds.tolist())
    return g


if __name__ == '__main__':
    g = [[1, 28], [2, 25], [3, 35], [4, 16], [5, 21], [], [7, 26], [8, 29], [9, 14], [10, 15], [11], [], [13, 7],
         [14, 27], [15], [16, 23], [17], [11], [19, 12], [20, 2], [21, 3], [22, 10], [23, 17], [], [25, 19], [26, 22],
         [27, 20], [28, 4], [29, 9], [5], [31, 18], [32, 6], [33, 0], [34, 13], [35, 1], [8]]

    processing_time = [[27, 17, 69, 43, 56, 77, ],
                       [80, 90, 15, 92, 58, 90, ],
                       [12, 7, 43, 57, 2, 8, ],
                       [52, 28, 74, 16, 2, 24, ],
                       [86, 25, 8, 23, 55, 78, ],
                       [17, 71, 36, 32, 33, 46, ]]

    opids = np.array(
        [[24, 19, 2, 35, 8, 14],
         [30, 18, 12, 7, 29, 5],
         [31, 6, 26, 20, 3, 16],
         [34, 1, 25, 22, 17, 11],
         [33, 13, 27, 4, 21, 10],
         [32, 0, 28, 9, 15, 23]]
    )

    groups = mu.get_groups(3, 6)
    print(get_cps_mss_by_group(processing_time, g, opids, groups))
