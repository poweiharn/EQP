import random
import copy

class geneticProgrammingPopulation():
	def __init__(self, individual_class, mu, num_children, mutation_rate,
				 parent_selection, survival_selection,fitness_kwargs=dict(),
				 initialization_kwargs=dict(), parent_selection_kwargs=dict(),
				 recombination_kwargs = dict(), mutation_kwargs = dict(),
				 survival_selection_kwargs=dict(), **kwargs):
		self.mu = mu
		self.num_children = num_children
		self.mutation_rate = mutation_rate
		self.parent_selection = parent_selection
		self.survival_selection = survival_selection
		self.fitness_kwargs = fitness_kwargs
		self.initialization_kwargs = initialization_kwargs
		self.parent_selection_kwargs = parent_selection_kwargs
		self.recombination_kwargs = recombination_kwargs
		self.mutation_kwargs = mutation_kwargs
		self.survival_selection_kwargs = survival_selection_kwargs

		self.population = individual_class.initialization(self.mu, initialization_kwargs, **fitness_kwargs)

	def generate_children(self):
		children = list()
		# TODO: Select parents
		# hint: self.parent_selection(self.population, **self.parent_selection_kwargs)

		# TODO: Generate children by either recombining two parents OR
		#		generating a mutated copy of a single parent
		for _ in range(self.num_children):
			if random.random() > self.mutation_rate:
				parents = self.parent_selection(self.population, 2, **self.parent_selection_kwargs)
				child = copy.deepcopy(parents[0].recombine(parents[1], self.initialization_kwargs,**self.fitness_kwargs))
			else:
				parents = self.parent_selection(self.population, 1, **self.parent_selection_kwargs)
				child = parents[0].mutate(self.initialization_kwargs, **self.fitness_kwargs)
			children.append(child)

		return children


	def survival(self):
		self.population = self.survival_selection(self.population, self.mu, **self.survival_selection_kwargs)