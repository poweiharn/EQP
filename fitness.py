import random
import simulation
from pooling_opt_count import quadtree_op_count, ea_op_count


def run_simulation(individual, train_set, val_set, test_set, configs, **kwargs):
    fitness_score = 0

    if kwargs['fitness_metric'] == "best_op":
        fitness_score = -quadtree_op_count(individual.gene)
        data = {}
    elif kwargs['fitness_metric'] == "ea_op":
        fitness_score = -ea_op_count(individual.gene)
        data = {}
    else:
        sim = simulation.PoolingSimulation(configs, train_set, val_set, test_set, **kwargs)
        data = sim.run(individual)

        if kwargs['fitness_metric'] == "best_acc":
            fitness_score = data["best_val_acc"]
        elif kwargs['fitness_metric'] == "gen_gap":
            fitness_score = -data["gen_gap"]
        elif kwargs['fitness_metric'] == "best_acc&op":
            fitness_score = data["best_val_acc"] - (data["op_count"]/10000)
        elif kwargs['fitness_metric'] == "best_acc&op&gen":
            fitness_score = data["best_val_acc"] - data["gen_gap"] - (data["op_count"]/10000)

    return fitness_score, data




