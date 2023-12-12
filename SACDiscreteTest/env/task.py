# 初始化 taskCost : cpuCost、gpuCost、memoryCost、networkCost、execTime
import random

task_cost_map = {}

task_length = 50
# vm_list = ['node1', 'node2', 'node3', 'node4']
# vm_list_resource = [{'node1': [1000000, 16384]}, {'node2': [3000000, 32768]}, {'node3': [500000, 8192]}, {'node4': [1000000, 8192]}]
vm_list = []
# vm_list_resource = [{'node1': [1000000, 16384]}, {'node2': [2000000, 32768]}, {'node3': [3000000, 32768]}, {'node4': [400000000, 32768]}, {'node5': [5000000, 32768]}, {'node6': [6000000, 32768]}]
vm_list_resource = [{'node1': [4000, 9000]}, {'node2': [4000, 9000]}, {'node3': [4000, 9000]}, {'node4': [4000, 9000]}]
for node_item in vm_list_resource:
    for key in node_item.keys():
        vm_list.append(key)
def init_task_cost_map():
    task_cost_map = {}
    random.seed(100)
    taskTotal = []
    for idx, node in enumerate(vm_list):
        task_node = []
        for i in range(task_length):
            task = []
            # task.append(random.randint(1, 100 * (idx + 1)))
            # task.append(random.uniform(1, 10 * (idx + 1)))
            # task.append(random.uniform(1, 100 * (idx + 1)))
            task.append(random.randint(1, 100))
            task.append(random.uniform(1, 100))
            task.append(random.uniform(1, 100))
            task_node.append(task)
        taskTotal.append(task_node)

    for idx, node_name in enumerate(vm_list):
        task_cost_map[node_name] = taskTotal[idx]

    return task_cost_map

def get_vm_list():
    return vm_list


def get_task_length():
    return task_length
