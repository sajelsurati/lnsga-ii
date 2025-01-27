import numpy as np

class Dnsga_II:

    def __init__(self, objective_list, min_values, max_values, population_size):
        self.min_values = min_values
        self.max_values = max_values
        self.objective_list = objective_list
        self.population_size = population_size
        self.generation_num = 0

    def initialize_solutions_discrete(self):
        self.population = np.empty((self.population_size, len(self.objective_list)))
        for i in range(len(self.min_values)):
            objective_values = np.random.randint(self.min_values[i], self.max_values[i], (self.population_size,)) #initialize solutions to different values in the given ranges
            # print(objective_values)
            self.population[:,i] = objective_values
        print('population\n',self.population)


    def evaluate_fitness(self):
        objective_values = np.empty((self.population_size, len(self.objective_list)))
        for i in range(len( self.objective_list)):
            for j in range(len(self.population)):
                objective_values[j,i] = self.objective_list[i](self.population[j]) #sets each row in the objective_value array to the objective values of the solutions
        print('values\n',objective_values)
        return objective_values
    
    def dominates(self, solution_1, solution_2):
        print('sol1\n',solution_1)
        print('sol2\n',solution_2)
        print('greater\n',np.any(solution_1 > solution_2))
        print('geq\n',np.all(solution_1 >= solution_2))
        return (np.any(solution_1 > solution_2) and np.all(solution_1 >= solution_2)) #geq for all values, and exists some value that is strictly greater

    def non_dominated_sorting(self, values):
        dominance_count = np.empty((self.population_size))


    def nsga_ii_discrete(self):
        self.initialize_solutions_discrete()
        self.evaluate_fitness()
