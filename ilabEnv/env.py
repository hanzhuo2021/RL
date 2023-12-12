import copy
import queue
import time

import numpy as np

from ilabEnv import task, node
from SACDiscreteTest.ga.ga import GA

def is_ready(node_load, task_cost):
    for index, value in enumerate(node_load):
        if value + task_cost[index] > 100.0:
            return False
    return True


def add_load(node_load, task_cost):
    result = []
    for index, value in enumerate(node_load):
        result.append(value + task_cost[index])
    return result


def remove_load(node_load, task_cost):
    result = []
    for index, value in enumerate(node_load):
        result.append(value - task_cost[index])
    return result


class Environment:
    def __init__(self):
        # 环境初始化时的固定值：节点列表、任务列表、任务代价列表
        self.NODE_LIST = task.get_NODE_LIST()
        self.bingfashu = 1
        # self.TASK_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        #                   27, 28, 29, 30]
        self.TASK_LIST = task.get_task_list()
        self.task_cost_map = task.init_task_cost_map()

        # 环境初始化时的变量：开始时间、排队队列、队列长度、执行队列、等待队列
        self.node_exec_priority_queue = {}
        self.node_wait_queue = {}
        for node_name in self.NODE_LIST:
            # 执行队列，字典类，每个节点一个队列（小根堆）
            self.node_exec_priority_queue[node_name] = []
            # 等待队列，字典类，每个节点一个队列（顺序队列）
            self.node_wait_queue[node_name] = []
        self.BEGIN_TIME = None
        self.queue_queue = queue.PriorityQueue()
        self.request_num = 0
        # 是否使用遗传算法
        self.fitness = 0.0
        # 输入到神经网络中的环境状态变量
        self.task_arrive_time = None
        self.task_id = None
        self.init_node_load = node.get_node_load(self.NODE_LIST)
        self.node_load = self.init_node_load.copy()
        self.node_task_cost = {}
        self.node_wait_time = {}

    def set_fitness(self, fitness):
        self.fitness = fitness

    def reset(self):
        # 环境初始化时的变量：开始时间、排队队列、队列长度、执行队列、等待队列
        self.node_exec_priority_queue = {}
        self.node_wait_queue = {}
        for node_name in self.NODE_LIST:
            # 执行队列，字典类，每个节点一个队列（小根堆）
            self.node_exec_priority_queue[node_name] = []
            # self.node_exec_priority_queue[node_name].queue.clear()
            # 等待队列，字典类，每个节点一个队列（顺序队列）
            self.node_wait_queue[node_name] = queue.Queue()
            self.node_wait_queue[node_name].queue.clear()
        self.BEGIN_TIME = time.time()
        self.queue_queue = queue.PriorityQueue()
        self.queue_queue.queue.clear()
        arrive_time_list = np.random.poisson(1000, self.bingfashu * len(self.TASK_LIST))
        # arrive_time_list.sort()
        for index, value in enumerate(arrive_time_list):
            self.queue_queue.put((value / 1000.0, self.TASK_LIST[index % len(self.TASK_LIST)]))
        self.request_num = self.queue_queue.qsize()

        # 输入到神经网络中的环境状态变量
        self.task_arrive_time = None
        self.task_id = None
        self.node_load = self.init_node_load.copy()
        self.node_task_cost = {}
        self.node_wait_time = {}

        # 更新输入到神经网络中的环境状态变量
        state = self.update_state()
        return state
        # return state / np.linalg.norm(state)

    def step(self, action):
        # 动作转换
        if action <= - 1.0 / 4:
            action = 0
        elif action <= 2.0 / 4:
            action = 1
        elif action <= 3.0 / 4:
            action = 2
        else:
            action = 3

        node_name = self.NODE_LIST[action]

        arrive_time, task_id = self.queue_queue.get()
        task_cost = self.task_cost_map[node_name][task_id]

        # 等待队列为空且资源充足时，直接将任务加入执行队列
        if self.node_wait_queue[node_name].empty() and is_ready(self.node_load[node_name], task_cost):
            # self.node_exec_priority_queue[node_name].put((arrive_time + task_cost[-1] + self.node_wait_time[node_name],
            #                                               self.node_wait_time[node_name], task_cost))
            self.node_exec_priority_queue[node_name].append((arrive_time + task_cost[-1], task_cost))
            self.node_load[node_name] = add_load(self.node_load[node_name], task_cost)

        # 判断是否结束
        done = self.queue_queue.empty()
        if done:
            # 获取奖励
            reward = self.get_reward(node_name)
        else:
            reward = self.get_reward_not_done(node_name)

        state = []
        if not done:
            # 更新环境状态
            state = self.update_state()
            # state /= np.linalg.norm(state)
        return state, reward, done

    # 标准化环境状态空间
    def update_state(self):
        # 环境状态1：根据排队队列生成的唯一id
        state = []
        # state.append(self.request_num - self.queue_queue.qsize())

        self.task_arrive_time, self.task_id = self.queue_queue.queue[0]
        # 环境状态2：队首任务到达时间、id
        # state.append(self.task_arrive_time)
        # state.append(self.task_id)

        for node_name in self.NODE_LIST:
            # 环境状态3：每个节点的负载数据
            state.extend(self.node_load[node_name])

        for node_name in self.NODE_LIST:
            self.node_task_cost[node_name] = self.task_cost_map[node_name][self.task_id]
            # # 环境状态4：任务在每个节点上运行需要的资源消耗数据
            state.extend(self.node_task_cost[node_name])

            # 环境状态4：任务在每个节点上运行时间
            # state.append(self.node_task_cost[node_name][-1])

        # for node_name in self.NODE_LIST:
        #     self.node_wait_time[node_name] = self.get_wait_time(node_name)
        #     # 环境状态5：任务在每个节点上需要等待的时间
        #     state.append(self.node_wait_time[node_name])

        return state

    # 计算任务在当前节点上的等待时间
    def get_wait_time(self, node_name):
        # 深拷贝 当前节点负载、当前任务资源消耗、当前执行队列、当前等待队列 数据
        current_node_load = copy.deepcopy(self.node_load[node_name])
        current_task_cost = copy.deepcopy(self.node_task_cost[node_name])
        current_exec_priority_queue = queue.PriorityQueue()
        current_exec_priority_queue.queue.clear()
        for item in self.node_exec_priority_queue[node_name].queue:
            current_exec_priority_queue.put(item)
        current_wait_queue = queue.Queue()
        current_wait_queue.queue.clear()
        for item in self.node_wait_queue[node_name].queue:
            current_wait_queue.put(item)

        # 等待队列为空且资源充足时，任务无需等待
        if current_wait_queue.empty() and is_ready(current_node_load, current_task_cost):
            return float()

        # 若节点资源不足，需要先清空等待队列，再计算等待时间
        wait_time = float()
        # 清空等待队列
        while not current_wait_queue.empty():
            wait_end_time, wait_wait_time, wait_task_cost = current_wait_queue.queue[0]
            # 执行队列出队
            while not is_ready(current_node_load, wait_task_cost):
                exec_end_time, exec_wait_time, exec_task_cost = current_exec_priority_queue.get()
                current_node_load = remove_load(current_node_load, exec_task_cost)
                wait_time = exec_end_time - self.task_arrive_time
            # 等待队列出队到执行队列
            current_exec_priority_queue.put(current_wait_queue.get())
            current_node_load = add_load(current_node_load, wait_task_cost)
        # 计算当前任务等待时间
        while not is_ready(current_node_load, current_task_cost):
            exec_end_time, exec_wait_time, exec_task_cost = current_exec_priority_queue.get()
            current_node_load = remove_load(current_node_load, exec_task_cost)
            wait_time = exec_end_time - self.task_arrive_time
        return wait_time

    # 获取奖励
    def get_reward(self, node_name):
        reward = [1.0, 0.3, -0.7]
        # wait_time_list = list(self.node_wait_time.values())
        # wait_time_list.sort()
        # wait_time = self.node_wait_time[node_name]
        # index = wait_time_list.index(wait_time)
        exec_time = self.node_task_cost[node_name][-1]
        # if self.is_use_GA:
        #     fitness = self.get_best_fitness()
        # else:
        #     fitness = 0.0
        # return reward[index]
        w = 0.7
        # return - (w * exec_time + (1 - w) * wait_time) / 1000
        makespan = self.get_makespan()
        return -(makespan + self.fitness) / 10
        # return -(exec_time) / 1000

    def get_reward_not_done(self, node_name):
        reward = [1.0, 0.3, -0.7]
        exec_time = self.node_task_cost[node_name][-1]
        return -(exec_time) / 1000

    # 获取最小完工时间
    def get_makespan(self):
        makespan = 0.0
        for key in self.node_exec_priority_queue.keys():
            queue = self.node_exec_priority_queue[key]
            finish_time = 0.0
            for node in queue:
                finish_time += node[1][-1]
            # finish_time为0.0的不需要参与计算，因为其没有任务
            if finish_time != 0.0:
                if finish_time > makespan:
                    makespan = finish_time
        return makespan