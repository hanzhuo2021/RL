from SACDiscreteTest.ga.population import  Population
from SACDiscreteTest.env.task import *
class GA:
    def __init__(self, task_list, vm_list):
        self.task_list = task_list
        self.vm_list = vm_list
        self.population = Population(self.task_list, self.vm_list)
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
    NODE_LIST = ["node1", "node2", "node3", "node4"]
    TASK_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ga = GA(len(TASK_LIST), NODE_LIST)
    ga.run()
    ga.get_top_n_individuals(10)