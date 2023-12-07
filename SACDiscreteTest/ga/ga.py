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
        print([indi['value'] for indi in self.population.get_top_n_individuals(10)])
        return self.population.get_top_n_individuals(n)


if __name__ == '__main__':
    ga = GA(get_task_length(), get_vm_list())
    ga.run()
    ga.get_top_n_individuals(10)