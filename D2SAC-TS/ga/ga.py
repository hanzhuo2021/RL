from ga.population import Population
from ilabEnv import taskYiLai
import pandas as pd
class GA:
    def __init__(self, task_list, vm_list, init_task_cost_map):
        self.task_list = task_list
        self.vm_list = vm_list
        self.init_task_cost_map = init_task_cost_map
        self.population = Population(self.task_list, self.vm_list, init_task_cost_map)
        self.population.init_population()

    def run(self):
        for i in range(100):
            print("第" + str(i) + "次迭代")
            self.population.select()
            self.population.cross()
            self.population.gene_variation()

            # print([indi['label'].cal_fitness() for indi in self.population.get_top_n_individuals(10)])
        # print([indi['value'] for indi in self.population.get_top_n_individuals(10)])

    def get_top_n_individuals(self, n):
        sorted_population, sorted_fitnesses = self.population.get_top_n_individuals(n)
        print([indi.cal_fitness() for indi in sorted_population])
        return sorted_population


if __name__ == '__main__':
    NODE_LIST = taskYiLai.get_NODE_LIST()
    TASK_LIST = taskYiLai.get_task_list()
    init_task_cost_map = taskYiLai.init_task_cost_map()
    ga = GA(TASK_LIST, NODE_LIST, init_task_cost_map)
    ga.run()
    best_individual = ga.get_top_n_individuals(1)
    makespan = -best_individual[0].cal_fitness()
    makespan_field = {"best": [makespan], "final": [makespan]}
    makespanFrame = pd.DataFrame(makespan_field)
    # makespanFrame.to_csv("/opt/data/GA/makespan50.csv", index=False, sep=',')
    makespanFrame.to_csv("/opt/data/GA/makespanServer10.csv", index=False, sep=',')
    print(makespan)