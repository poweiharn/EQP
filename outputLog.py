from math import inf
import statistics

class Output_log():
    def __init__(self, total_run_number, file_path):
        self.path = file_path
        self.total_run_number = total_run_number

        self.best_fitness_of_all_runs = []
        self.best_fitness_of_all_time = -inf
        self.best_individual_of_all_time = None

        self.run_statistics = {}
        self.best_run_individuals = {}

        self.best_generation_individuals = {}

        for i in range(1, total_run_number + 1):
            self.run_statistics["Average_fitness_" + str(i)] = []
            self.run_statistics["Best_fitness_" + str(i)] = []
            self.best_run_individuals[str(i)] = None
            self.best_run_individuals["Best_fitness_run_" + str(i)] = -inf
            self.best_generation_individuals["run_"+str(i)] = {}

        self.f_log = open(self.path+"/search_log.txt", "w")

    def calculate_stat(self, run_num, generation_num, evaluation_num, population, init_write):
        average_fitness = statistics.mean([individual.fitness for individual in population])
        best_fitness = max([individual.fitness for individual in population])
        best_fit_individual = max(population, key=lambda individual: individual.fitness)
        self.best_generation_individuals["run_" + str(run_num+1)]["gen_"+str(generation_num)] = best_fit_individual

        if best_fitness > self.best_run_individuals["Best_fitness_run_" + str(run_num+1)]:
            self.best_run_individuals["Best_fitness_run_" + str(run_num+1)] = best_fitness
            self.best_run_individuals[str(run_num+1)] = best_fit_individual

        if init_write:
            self.f_log.write('----------run:' + str(run_num+1) + '-------\n')
            self.f_log.write(f'{"Generation".ljust(15, " ")}'f'{"Evaluations".ljust(15, " ")}'
                        f'   {"Average Fitness".ljust(20, " ")}'
                        f'   {"Best Fitness".ljust(20, " ")}'
                        f'   {"Best Generation Individual".ljust(20, " ")}\n')
        self.f_log.write(f'{str(generation_num).ljust(15, " ")}'f'{str(evaluation_num).ljust(15, " ")}'
                    f'   {str(average_fitness).ljust(20, " ")}'
                    f'   {str(best_fitness).ljust(20, " ")}'
                    f'   {best_fit_individual.name.ljust(20, " ")}\n')

        self.run_statistics["Average_fitness_" + str(run_num + 1)].append(average_fitness)
        self.run_statistics["Best_fitness_" + str(run_num + 1)].append(best_fitness)

    def calculate_best_of_all_time(self):
        for i in range(1, self.total_run_number + 1):
            if self.best_run_individuals["Best_fitness_run_" + str(i)] > self.best_fitness_of_all_time:
                self.best_fitness_of_all_time = self.best_run_individuals["Best_fitness_run_" + str(i)]
                self.best_individual_of_all_time = self.best_run_individuals[str(i)]

        self.f_log.write('\n')
        self.f_log.write('Best Fitness: '+ str(self.best_individual_of_all_time.fitness)+'\n')
        self.f_log.write('Best Individual: '+ self.best_individual_of_all_time.name +'\n')
        self.f_log.write(self.best_individual_of_all_time.print())


    def export_best_run_fitness(self):
        with open(self.path+"/best_run_fitness.txt", 'w') as f:
            [f.write(f'{self.best_run_individuals["Best_fitness_run_" + str(i)]}\n') for i in range(1, self.total_run_number + 1)]

    def export_statistics(self):
        # open file for writing
        f = open(self.path+"/statistics.txt", "w")
        f.write(str(self.run_statistics))
        f.close()



