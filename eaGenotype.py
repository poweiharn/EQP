import random
import math
import copy

class eaGenotype():
    def __init__(self):
        self.gene = None
        self.fitness = None

    def randomInitialization(self, length, sensors):
        # TODO: Add random initialization of fixed-length binary gene
        self.gene = []
        for i in range(length-1,-1,-1):
            self.gene.append(random.choices(sensors, weights = None, k = (2**i) * (2**i)))
        pass

    def recombine(self, mate, init_args, **kwargs):
        child = self.__class__()
        child.gene = copy.deepcopy(self.gene)
        # Get subtree from mate
        mate_gene = copy.deepcopy(mate.gene)
        seq_length = init_args['seq_length']

        for i in range(seq_length):
            c_g = child.gene[i]
            m_g = mate_gene[i]
            # perform 1-point crossover
            if len(c_g) - 1 > 0:
                crossover_point = random.randint(1, len(c_g) - 1)
                for i in range(crossover_point, len(c_g)):
                    c_g[i] = m_g[i]
        return child

    def mutate(self, init_args, **kwargs):
        # TODO: mutate gene of copy
        sensors = list(init_args['leaf_nodes'])

        child = self.__class__()
        child.gene = copy.deepcopy(self.gene)

        if 'mutation_rate' in kwargs:
            mutation_rate = kwargs['mutation_rate']
        else:
            mutation_rate = 0.5

        seq_length = init_args['seq_length']
        for i in range(seq_length):
            for gene_index in range(len(child.gene[i])):
                random_value = random.random()
                if random_value < mutation_rate:
                    g = child.gene[i][gene_index]
                    new_sensor = [x for x in sensors if x != g]

                    child.gene[i][gene_index] = random.choice(new_sensor)

        return child

    def print(self):
        # TODO: return a string representation of self.gene
        #       (see assignment description doc for more info)
        string = ""
        for g in self.gene:
            string = string + str(g) + '\n'

        return string

    @classmethod
    def initialization(cls, mu, init_args, **kwargs):
        population = [cls() for _ in range(mu)]
        seq_length = init_args['seq_length']
        sensors = list(init_args['leaf_nodes'])
        for i in range(len(population)):
            population[i].randomInitialization(seq_length, sensors)
        return population