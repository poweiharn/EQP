import random
import copy


# Parent selection functions---------------------------------------------------
def uniform_random_selection(population, n, **kwargs):
    # TODO: select n individuals uniform randomly
    return random.choices(population, k=n)


def k_tournament_with_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments with replacement to select n individuals
    parent = []
    for _ in range(n):
        # Select k unique individuals from the population
        k_individuals = random.sample(population, k=k)
        # Select the best individual from the k individuals
        best_fit_individual = max(k_individuals, key=lambda individual: individual.fitness)
        parent.append(best_fit_individual)
    return parent


def fitness_proportionate_selection(population, n, **kwargs):
    # TODO: select n individuals using fitness proportionate selection
    modified_fitness_list = []
    min_fitness = min([individual.fitness for individual in population])
    for individual in population:
        # Temporarily modified fitness: fitness - min_fitness*1.5
        modified_fitness = individual.fitness - min_fitness * 1.5
        modified_fitness_list.append(modified_fitness)
    modified_fitness_sum = sum(modified_fitness_list)
    fitness_proportionate = [modified_fitness / modified_fitness_sum for modified_fitness in modified_fitness_list]
    return random.choices(population, weights=fitness_proportionate, k=n)


# Survival selection functions-------------------------------------------------
def truncation(population, n, **kwargs):
    # TODO: perform truncation selection to select n individuals
    # Population is sorted by fitness, the n most fit individuals are selected to survive.
    sorted_population_by_fitness = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    return sorted_population_by_fitness[0:n]


def k_tournament_without_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments without replacement to select n individuals
    #		Note: an individual should never be cloned from surviving twice!
    survivors = []
    clone_population = population.copy()
    for _ in range(n):
        # Select k unique individuals from the population
        k_individuals = random.sample(clone_population, k=k)
        # Select the best individual from the k individuals
        best_fit_individual = max(k_individuals, key=lambda individual: individual.fitness)
        survivors.append(best_fit_individual)
        # Remove the best individual from the cloned population to avoid surviving twice.
        clone_population.remove(best_fit_individual)
    return survivors


# Yellow deliverable parent selection function---------------------------------
def stochastic_universal_sampling(population, n, **kwargs):
    # Recall that yellow deliverables are required for students in the grad
    # section but bonus for those in the undergrad section.
    # TODO: select n individuals using stochastic universal sampling
    parent = []
    modified_fitness_list = []
    min_fitness = min([individual.fitness for individual in population])
    for individual in population:
        # Temporarily modified fitness: fitness - min_fitness*1.5
        modified_fitness = individual.fitness - min_fitness * 1.5
        modified_fitness_list.append(modified_fitness)
    modified_fitness_sum = sum(modified_fitness_list)
    fitness_proportionate = [modified_fitness / modified_fitness_sum for modified_fitness in modified_fitness_list]
    # n equally spaced pointers
    step = sum(fitness_proportionate) / n
    random_offset = random.uniform(0, step)
    pointer = random_offset
    for _ in range(n):
        for i in range(len(fitness_proportionate)):
            if pointer >= sum(fitness_proportionate[0:i]) and pointer <= sum(fitness_proportionate[0:i + 1]):
                parent.append(population[i])
                break
        pointer = pointer + step
    return parent
