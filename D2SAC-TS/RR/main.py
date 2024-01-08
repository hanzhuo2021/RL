from ilabEnv.taskYiLai import *
import pandas as pd
def round_robin_scheduling(tasks, servers):
    # 字典来存储服务器-任务分配
    server_tasks = {server: [] for server in servers}

    # 以轮询方式将任务分配给服务器
    for i, task in enumerate(tasks):
        index = i % len(servers)
        server_tasks[servers[index]].append(task)

    return server_tasks

def get_makespan(assignments):
    task_node_list = init_task_cost_map()
    makespan = 0.0
    for node in assignments:
        assign_task_list = assignments[node]
        task_for_node = task_node_list[node]
        finish_time = 0.0
        for id in assign_task_list:
            task = task_for_node[id]
            finish_time += task[-1]
        if finish_time > makespan:
            makespan = finish_time
    return makespan

task_list = get_task_list()
node_list = get_NODE_LIST()
# 示例使用
assignments = round_robin_scheduling(task_list, node_list)
makespan = get_makespan(assignments)
makespan_field = {"best": [makespan], "final": [makespan]}
makespanFrame = pd.DataFrame(makespan_field)
# makespanFrame.to_csv("/opt/data/RR/makespan50.csv", index=False, sep=',')
makespanFrame.to_csv("/opt/data/RR/makespanServer10.csv", index=False, sep=',')
print(makespan)
