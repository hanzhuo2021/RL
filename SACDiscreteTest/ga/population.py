import random

from SACDiscreteTest.ga.individual import Individual


class Population:
    def __init__(self, task_list, vm_list):
        self.population_size = 200
        self.variation = 0.2
        self.retain = 2
        self.cross_rate = 0.5
        self.max_variation_num = 1
        self.task_list = task_list
        self.vm_list = vm_list
        self.individuals = []

    def init_population(self):
        for i in range(self.population_size):
            individual = Individual(self.task_list, self.vm_list)
            genes = individual.genes
            for idx in range(self.task_list):
                gene = genes[idx]
                gene.task = idx
                gene.vm = random.randint(0, len(self.vm_list) - 1)
                individual.genes[idx] = gene

            self.individuals.append(individual)

    def get_top_n_individuals(self, n):
        temp = [{'label': individual, 'value': individual.cal_fitness()} for individual in self.individuals]
        sorted_individual = sorted(temp, key=lambda x: x['value'], reverse=True)
        # sorted_individual = [individual['label'] for individual in sorted_individual]
        return sorted_individual[:n]

    def select(self):
        individuals = [ind['label'] for ind in self.get_top_n_individuals(len(self.individuals))]
        new_population = []
        # 复制最优个体
        for i in range(self.retain):
            new_population.append(individuals[i])

        total_fitness = 0.0
        for individual in individuals[self.retain:]:
            total_fitness += individual.cal_fitness()

        cumulative_p = 0.0
        for individual in individuals[self.retain: self.population_size]:
            selected = None
            while selected is None:
                num = random.randint(self.retain, len(individuals) - 1)
                cur_individual = individuals[num]
                p = cur_individual.cal_fitness() / total_fitness
                cumulative_p += p
                val = random.random()
                if val <= cumulative_p:
                    selected = cur_individual

            new_population.append(selected)

        self.individuals = new_population

    def cross(self):
        new_individuals = []
        for individual in self.individuals:
            rate = random.random()
            if rate <= self.cross_rate:
                i = random.randint(0, len(self.individuals) - 1)
                target = self.individuals[i]
                genes1 = individual.genes
                genes2 = target.genes
                index = random.randint(0, len(genes1) - 1)
                new_individual = Individual(self.task_list, self.vm_list)
                new_genes = new_individual.genes
                for idx, gene in enumerate(new_genes):
                    if idx <= index:
                        new_genes[idx] = genes1[idx]
                    else:
                        new_genes[idx] = genes2[idx]
                new_individual.genes = new_genes
                new_individuals.append(new_individual)
        self.individuals += new_individuals

    def gene_variation(self):
        for individual in self.individuals:
            rate = random.random()
            if rate <= self.variation:
                count = random.randint(0, self.max_variation_num)
                for i in range(count):
                    genes = individual.genes
                    index = random.randint(0, len(genes) - 1)
                    genes[index].vm = random.randint(0, len(self.vm_list) - 1)
