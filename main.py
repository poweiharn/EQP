import os
import pickle
import random
import statistics

import torch
import torch.optim as optim

from utils import readConfig, checkConfig
from fitness import run_simulation
from selection import *
from qtreeGenotype import *
from eaGenotype import *
from models.get_network import get_network
from outputLog import Output_log
from geneticProgramming import geneticProgrammingPopulation

from configs import configs
from dataset_loader import get_data_set


def evaluate_population(population, run_stat,  train_set, val_set, test_set, nn_config, **fitness_kwargs):
    for individual in population:
        run_stat["eval_num"] = run_stat["eval_num"] + 1
        configs.name = 'r'+str(run_stat["run_num"]) +'g' + str(run_stat["generation"]) + 'e' +str(run_stat["eval_num"])
        individual.name = configs.name
        individual.fitness, individual.sim_data = run_simulation(individual, train_set, val_set, test_set, nn_config, **fitness_kwargs)


def main():

    # Read config
    config = readConfig('./configs/config_ea.txt', globalVars=globals(), localVars=locals())
    #config = readConfig('./configs/config.txt', globalVars=globals(), localVars=locals())
    checkConfig(config)
    # Neural Network configs
    configs.add_args(config['fitness_kwargs'])
    configs.seed = config['EA_configs']['seed']
    configs.epoch = config['fitness_kwargs']['epoch'] + config['fitness_kwargs']['warmup_epoch']
    configs.training_init()
    configs.path_init()

    # Load Dataset
    train_loader, val_loader, test_loader = get_data_set(configs)

    # Set initial model weights
    #C__copy = copy.deepcopy(configs)
    #C__copy.pooling = 'M'
    #net = get_network(C__copy)
    #optimizer = optim.SGD(params=net.parameters(), lr=C__copy.lr, momentum=0.9, weight_decay=5e-4)
    #torch.save(net.state_dict(), "{}/init_nn_model.pt".format(C__copy.result_log_dir))
    #torch.save(optimizer.state_dict(), "{}/init_optimizer.pt".format(C__copy.result_log_dir))


    #random.seed(config['EA_configs']['seed'])
    r_tree_gp = geneticProgrammingPopulation(**config['EA_configs'], **config)
    rt_gp_log = Output_log(config['EA_configs']['run'], 'log')

    for run in range(config['EA_configs']['run']):
        print('----------run:' + str(run+1) + '-------')
        # Initialization
        run_stat = {
            "run_num": run+1,
            "generation": 0,
            "eval_num":0
        }
        r_tree_gp = geneticProgrammingPopulation(**config['EA_configs'], **config)
        # Fitness evaluation

        evaluate_population(r_tree_gp.population, run_stat, train_loader, val_loader, test_loader, configs, **config['fitness_kwargs'])
        r_tree_gp.evaluations = len(r_tree_gp.population)
        # Save statistics
        rt_gp_log.calculate_stat(run, run_stat["generation"], r_tree_gp.evaluations, r_tree_gp.population, True)

        while r_tree_gp.evaluations < config['EA_configs']['number_evaluations']:

            run_stat["generation"] = run_stat["generation"] + 1
            # Child generation
            children = r_tree_gp.generate_children()
            # Children fitness evaluation
            evaluate_population(children, run_stat, train_loader, val_loader, test_loader, configs, **config['fitness_kwargs'])
            r_tree_gp.evaluations += len(children)
            # Children are added to the population with the parents
            r_tree_gp.population += children
            # Survival selection
            r_tree_gp.survival()
            # Save statistics
            rt_gp_log.calculate_stat(run, run_stat["generation"], r_tree_gp.evaluations, r_tree_gp.population, False)

            print(f'Average fitness of population: {statistics.mean([individual.fitness for individual in r_tree_gp.population])}')
            print(f'Best fitness in population: {max([individual.fitness for individual in r_tree_gp.population])}')
            print(f'Number of fitness evaluations: {r_tree_gp.evaluations}')

    rt_gp_log.calculate_best_of_all_time()

    rt_gp_log.f_log.close()
    rt_gp_log.export_best_run_fitness()
    rt_gp_log.export_statistics()

    # Export Best Individual
    best_individual = rt_gp_log.best_individual_of_all_time
    print('Best Fitness:', best_individual.fitness)
    print('Best Individual:')
    print(best_individual.print())

    best_individual.config = config
    with open('log/'+best_individual.name+'.pkl', 'wb') as out_file:
        pickle.dump(best_individual, out_file)


if __name__ == "__main__":
    main()