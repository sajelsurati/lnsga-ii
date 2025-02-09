import numpy as np

class Dnsga_II:

    def __init__(self, objective_list, min_values, max_values, population_size, mutation_percent):
        self.min_values = min_values
        self.max_values = max_values
        self.objective_list = objective_list
        self.population_size = population_size
        self.generation_num = 0
        self.mutation_percent = mutation_percent

    def initialize_solutions_discrete(self):
        self.population = np.empty((self.population_size, len(self.objective_list)))
        for i in range(len(self.min_values)):
            #initialize solutions to different values in the given ranges
            objective_values = np.random.randint(self.min_values[i], self.max_values[i], (self.population_size,))
            self.population[:,i] = objective_values
        print('population\n',self.population)


    def evaluate_fitness(self):
        objective_values = np.empty((self.population_size, len(self.objective_list)))

        #sets each row in the objective_value array to the objective values of the solutions
        for i in range(len( self.objective_list)):
            for j in range(len(self.population)):
                objective_values[j,i] = self.objective_list[i](self.population[j])
        print('values\n',objective_values)
        return objective_values

    def non_dominated_sorting(self, values):
        front_assignments = np.zeros((self.population_size,))
        #ensuring all necessary fronts are calculated
        front_assignments -= 1
        #an nxn matrix that counts how many values each solution dominates
        #column represents what rows are dominated
        dominance_list = np.zeros((self.population_size, self.population_size))
        #for each objective value, check if it dominates other values
        for i in range(len(self.population)):
            # all values must be geq and there must exist a strictly greater than value
            dominance_boolean = (np.all(values[i] >= values, axis = 1) & 
                np.any(values[i] > values, axis = 1))
            dominance_list[:,i] += dominance_boolean.astype(int)
        front_count = 0
        #until all solutions have assigned fronts
        while (np.any(front_assignments < 0)):
            #sums row-wise, where the non-dominated should have a sum of 0
            dominance_sum = np.sum(dominance_list, axis=1)
            #if dominance sum is 0 and the front hasn't been assigned yet
            solutions_to_add = np.logical_and(dominance_sum == 0, front_assignments == -1)
            front_assignments[solutions_to_add] = front_count
            #removing the non-dominated solution columns by changing those columns to 0s
            trues = np.full((self.population_size,), True)
            dominance_list[np.ix_(trues, solutions_to_add)] = 0
            front_count+=1
        return front_assignments
            
    def calculate_crowding_distance(self, values):
        crowding_distance = np.zeros((self.population_size,))
        for i in range(len(self.objective_list)):
            #sort by ith objective values
            sorted_rows = np.argsort(values[:,i])
            #boundary solutions have an infinite distance (least crowded)
            crowding_distance[[sorted_rows[0], sorted_rows[-1]]] = np.inf
            #non-boundary solutions have crowding value based on equation given in paper
            crowding_distance[sorted_rows[1:-1]] += ((values[sorted_rows[2:],i]-values[sorted_rows[:-2],i])/
                (values[[sorted_rows[-1]],i]-values[[sorted_rows[0]],i]))
        return(crowding_distance)

    def mutation(self, child):
        mutation = np.random.random()
        # cross_over_point = 
    
    
    def crossover(self, parent_1, parent_2):
        #select a random crossover point
        cross_over_point =  np.random.randint(0, len(self.objective_list))
        print(cross_over_point)
        child = np.zeros((len(self.objective_list),))
        print(child)
        #choose objective values up until crossover point from parent 1
        child[:cross_over_point] = parent_1[:cross_over_point]
        #choose objective values after crossover point from parent 2
        child[cross_over_point:] = parent_2[:cross_over_point]
        return child

    def binary_tournament(self, fronts, crowding):
        #generate 2 arrays of parents for binary tournament of length 2*population size
        binary_1 = np.random.randint(0,self.population_size, (2*self.population_size,))
        binary_2 = np.random.randint(0,self.population_size, (2*self.population_size,))

        #array for remaining parents from binary tournament
        parents = np.zeros((2*self.population_size, len(self.objective_list)))

        #where front of 1st solution is better than the 2nd
        binary_1_smaller = fronts[binary_1] < fronts[binary_2]
        #where front of 2nd solution is better than the 1st
        binary_2_smaller = fronts[binary_2] < fronts[binary_1]
        #fronts are the same, so 1st solution has geq crowding distance
        binary_1_crowding = (fronts[binary_1] == fronts[binary_2]) & (crowding[binary_1] >= crowding[binary_2])
        #fronts are the same, so 2nd solution has better crowding distance
        binary_2_crowding = (fronts[binary_1] == fronts[binary_2]) & (crowding[binary_2] > crowding[binary_1])

        #combining all the above masks into 1 array
        parents[np.where(binary_1_smaller),:] = self.population[binary_1[np.where(binary_1_smaller)],:]
        parents[np.where(binary_2_smaller),:] = self.population[binary_2[np.where(binary_2_smaller)],:]
        parents[np.where(binary_1_crowding),:] = self.population[binary_1[np.where(binary_1_crowding)],:]
        parents[np.where(binary_2_crowding),:] = self.population[binary_2[np.where(binary_2_crowding)],:]

        #setting odd rows as parent 1 and even rows as parent 2
        parents_1 = parents[::2,:]
        parents_2 = parents[1::2,:]
        return parents_1, parents_2

    def generate_children(self, fronts, crowding):
        #binary tournament phase:
        parents_1, parents_2 = self.binary_tournament(fronts, crowding)

        children = np.empty((len(self.population), len(self.objective_list)))

        for i in range(len(parents_1)):
            child = self.mutation(self.crossover(parents_1[i,:], parents_2[i,:]))
            children[i,:] = child
        return children



    def nsga_ii_discrete(self):
        self.initialize_solutions_discrete()
        objective_values = self.evaluate_fitness()
        fronts = self.non_dominated_sorting(objective_values)
        crowding_values = self.calculate_crowding_distance(objective_values)
        self.generate_children(fronts, crowding_values)