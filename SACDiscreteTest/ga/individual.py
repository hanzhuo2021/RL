import numpy as np

from SACDiscreteTest.ga.gene import Gene
from SACDiscreteTest.env.task import init_task_cost_map


class Individual:
    def __init__(self, task_list, vm_list):
        self.genes = []
        self.task_list = task_list
        self.vm_list = vm_list
        for task in range(task_list):
            gene = Gene(-1, -1)
            self.genes.append(gene)

    def cal_fitness(self):
        time_map = {}
        for gene in self.genes:
            task_id = gene.task
            vm_id = gene.vm
            task_cost_map = init_task_cost_map()
            task = task_cost_map[self.vm_list[vm_id]][task_id]
            if vm_id not in time_map:
                time_map[vm_id] = task[2]
            else:
                time_map[vm_id] += task[2]

        # 计算timeMap中最大的时间，即为虚拟机集群的完工时间
        max_time = 0.0
        for key in time_map.keys():
            if time_map[key] > max_time:
                max_time = time_map[key]

        return -max_time
