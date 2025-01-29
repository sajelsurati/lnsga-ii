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
        front_assignments = np.zeros((self.population_size,))
        front_assignments -= 1
        dominance_list = np.zeros((self.population_size, self.population_size))
        for i in range(len(self.population)): #for each objective value, check if it dominates other values
            # print('testing',self.population[i], values[i])
            # print(np.all(values[i] >= values, axis = 1))
            # print(np.any(values[i] > values, axis = 1))
            # print(np.all(values[i] >= values, axis = 1) & np.any(values[i] > values, axis = 1))
            dominance_boolean = np.all(values[i] >= values, axis = 1) & np.any(values[i] > values, axis = 1)
            # dominance_count += dominance_boolean.astype(int) #add 1 to dominance count for each dominated value
            dominance_list[:,i] += dominance_boolean.astype(int)
        front_count = 0
        while (np.any(front_assignments < 0)):
            # print(dominance_list)
            dominance_sum = np.sum(dominance_list, axis=1)
            # print(dominance_sum)
            solutions_to_add = np.logical_and(dominance_sum == 0, front_assignments == -1)
            front_assignments[solutions_to_add] = front_count
            # print('sum\n',dominance_sum.astype(bool))
            # dominance_list = np.delete(dominance_list,(dominance_sum == 0), axis=1)
            # dominance_list = np.delete(dominance_list,(dominance_sum == 0), axis=0)
            # print('list\n',dominance_list)
            # print(np.expand_dims(solutions_to_add, 1).shape)
            # print(np.expand_dims(solutions_to_add, 1))
            trues = np.full((self.population_size,), True)
            front_count+=1
            dominance_list[np.ix_(trues, solutions_to_add)] = 0
            # print('fronts',front_assignments)
        return front_assignments
            
    def calculate_crowding_distance(self, values):
        crowding_distance = np.zeros((self.population_size,))
        for i in range(len(self.objective_list)):
            
            sorted_rows = np.argsort(values[:,i])
            crowding_distance[[sorted_rows[0], sorted_rows[-1]]] = np.inf
            print(crowding_distance)
            print('sorted',sorted_rows)


    def nsga_ii_discrete(self):
        self.initialize_solutions_discrete()
        objective_values = self.evaluate_fitness()
        self.non_dominated_sorting(objective_values)
        self.calculate_crowding_distance(objective_values)
