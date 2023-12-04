# 初始化 taskCost : cpuCost、gpuCost、memoryCost、networkCost、execTime
import random

task_cost_map = {}


def init_task_cost_map():
    task_cost_map = {}
    taskArr1 = []
    taskArr2 = []
    taskArr3 = []
    taskArr4 = []
    for i in range(10):
        task1 = []
        task1.append(random.randint(1, 3000))
        task1.append(random.uniform(1,128))
        task1.append(random.uniform(1,1000))
        taskArr1.append(task1)

        task2 = []
        task2.append(random.randint(1, 3000))
        task2.append(random.uniform(1,128))
        task2.append(random.uniform(1,1000))
        taskArr2.append(task2)

        task3 = []
        task3.append(random.randint(1, 3000))
        task3.append(random.uniform(1, 128))
        task3.append(random.uniform(1, 1000))
        taskArr3.append(task3)

        task4 = []
        task4.append(random.randint(1, 3000))
        task4.append(random.uniform(1, 128))
        task4.append(random.uniform(1, 1000))
        taskArr4.append(task4)

    task_cost_map['node1'] = taskArr1
    task_cost_map['node2'] = taskArr2
    task_cost_map['node3'] = taskArr3
    task_cost_map['node4'] = taskArr4

    return task_cost_map

# 输出内容：二维数组，每行为在各个节点上消耗的资源
# [[4.0010285, 0.0, 2.69980363, 0.0, 12.68250000],
#  [4.0010285, 0.0, 2.69980363, 0.0, 12.68250000],
#  [4.0010285, 0.0, 2.69980363, 0.0, 12.68250000]]
# def get_task_info(task_id):
#     task_info = {}
#     for key in task_cost_map:
#         node_task_cost_map = task_cost_map[key]
#         task_cost = node_task_cost_map[task_id]
#         task_info.extend(task_cost)
#     return task_info
