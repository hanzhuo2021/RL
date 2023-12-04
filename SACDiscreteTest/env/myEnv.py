from SACDiscreteTest.env import task
import sys
import numpy as np
from gym import spaces


class Environment:
    def __init__(self):
        self.NODE_LIST = ["node1", "node2", "node3", "node4"]
        # CPU(MIPS)、MEM(MB)
        self.NODE_LIST_RESOURCE = [
            [1000000, 16384],
            [3000000, 32768],
            [500000, 8192],
            [1000000, 8192],
        ]
        # 资源分配列表
        self.allocation_table = {'node1': [], 'node2': [], 'node3': [], 'node4': []}

        self.task_cost_map = task.init_task_cost_map()
        self.TASK_LIST = [i for i in range(len(self.task_cost_map['node1']))]
        self.action_space = [0, 1, 2, 3]
        self.ptr = 0
        self.observation_space = self.get_state()

    def reset(self):
        self.NODE_LIST = ["node1", "node2", "node3", "node4"]
        # CPU(MIPS)、MEM(MB)
        self.NODE_LIST_RESOURCE = [
            [1000000, 16384],
            [3000000, 32768],
            [500000, 8192],
            [1000000, 8192],
        ]
        self.ptr = 0
        self.allocation_table = {'node1': [], 'node2': [], 'node3': [], 'node4': []}
        self.TASK_LIST = [i for i in range(len(self.task_cost_map['node1']))]
        self.task_cost_map = task.init_task_cost_map()
        self.action_space = [0, 1, 2, 3]
        state = self.get_state()
        return np.array(state), {}

    def get_state(self):
        state = []
        # try:
        #     task_id = self.TASK_LIST[self.ptr]
        # except:
        #     print("错误")
        # 服务器资源
        for sublist in self.NODE_LIST_RESOURCE:
            for item in sublist:
                state.append(item)

        # 任务资源
        # for node_name in self.NODE_LIST:
        #     task = self.task_cost_map[node_name][task_id]
        #     state.extend(task)

        # 对数组进行归一化

        # state = np.sum(self.NODE_LIST_RESOURCE, axis=0)
        state = np.array(state)
        mean = state.mean(axis=0)
        std = state.std(axis=0)
        state = (state - mean) / std
        return state


    def step(self, action):
        node_name = self.NODE_LIST[action]
        task_id = self.TASK_LIST[self.ptr]
        task = self.task_cost_map[node_name][task_id]
        self.allocation(task, node_name)

        self.ptr += 1
        done = self.ptr == (len(self.TASK_LIST))
        nex_state = self.get_state()
        if done:
            reward = self.get_reward()
        else:
            reward = 0
        return np.array(nex_state), reward, done

    def allocation(self, task, node_name):
        request_cpu = task[0]
        request_mem = task[1]
        index = -1
        if node_name in self.NODE_LIST:
            index = self.NODE_LIST.index(node_name)
        else:
            print(f"子元素 '{node_name}' 不在列表中")
            return False
        task_on_node = self.allocation_table[node_name]
        task_on_node.append(task)
        node_resource = self.NODE_LIST_RESOURCE[index]
        node_resource[0] = node_resource[0] - request_cpu
        node_resource[1] = node_resource[1] - request_mem

    def get_reward(self):
        makespan = 0.0
        for key in self.allocation_table.keys():
            node = self.allocation_table[key]
            finish_time = 0.0
            for task in node:
                finish_time += task[2]

            # finish_time为0.0的不需要参与计算，因为其没有任务
            if finish_time != 0.0:
                if finish_time > makespan:
                    makespan = finish_time
        return -makespan





