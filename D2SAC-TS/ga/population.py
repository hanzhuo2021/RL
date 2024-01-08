import random

from ga.individual import Individual


class Population:
    def __init__(self, task_list, vm_list, init_task_cost_map):
        self.population_size = 100
        self.variation = 0.2
        self.retain = 2
        self.cross_rate = 0.5
        self.max_variation_num = 1
        self.task_list = task_list
        self.vm_list = vm_list
        self.individuals = []
        self.init_task_cost_map = init_task_cost_map

    def init_population(self):
        for i in range(self.population_size):
            individual = Individual(self.task_list, self.vm_list, self.init_task_cost_map)
            genes = individual.genes
            for idx in range(len(self.task_list)):
                gene = genes[idx]
                gene.task = idx
                gene.vm = random.randint(0, len(self.vm_list) - 1)
                individual.genes[idx] = gene

            self.individuals.append(individual)

    def get_top_n_individuals(self, n):
        # temp = [{'label': individual, 'value': individual.cal_fitness()} for individual in self.individuals]
        # sorted_individual = sorted(temp, key=lambda x: x['value'], reverse=True)
        # # sorted_individual = [individual['label'] for individual in sorted_individual]
        # return sorted_individual[:n]

        paired_population = list(zip(self.individuals, [individual.cal_fitness() for individual in self.individuals]))
        paired_population.sort(key=lambda x: x[1], reverse=True)

        # 分离排序后的个体和适应度
        sorted_population, sorted_fitnesses = zip(*paired_population)
        return list(sorted_population[:n]), list(sorted_fitnesses[:n])

    def select(self):
        # 选择最优个体
        sorted_population, sorted_fitnesses = self.get_top_n_individuals(len(self.individuals))
        elite_population = list(sorted_population[:self.retain])

        # 对剩余个体进行轮盘赌选择
        remaining_population = list(sorted_population[self.retain:])
        remaining_fitnesses = list(sorted_fitnesses[self.retain:])
        total_fitness = sum(remaining_fitnesses)
        selection_probs = [f / total_fitness for f in remaining_fitnesses]

        selected_indices = random.choices(range(len(remaining_population)), weights=selection_probs,
                                          k=self.population_size - self.retain)
        selected_population = [remaining_population[i] for i in selected_indices]

        # 合并最优个体和选出的个体
        new_population = elite_population + selected_population

        self.individuals = new_population

        # individuals = [ind['label'] for ind in self.get_top_n_individuals(len(self.individuals))]
        # new_population = []
        # # 复制最优个体
        # for i in range(self.retain):
        #     new_population.append(individuals[i])
        #
        # total_fitness = 0.0
        # for individual in individuals[self.retain:]:
        #     total_fitness += individual.cal_fitness()
        #
        # cumulative_p = 0.0
        # for individual in individuals[self.retain: self.population_size]:
        #     selected = None
        #     while selected is None:
        #         num = random.randint(self.retain, len(individuals) - 1)
        #         cur_individual = individuals[num]
        #         p = cur_individual.cal_fitness() / total_fitness
        #         cumulative_p += p
        #         val = random.random()
        #         if val <= cumulative_p:
        #             selected = cur_individual
        #
        #     new_population.append(selected)
        #
        # self.individuals = new_population

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
                new_individual = Individual(self.task_list, self.vm_list, self.init_task_cost_map)
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
