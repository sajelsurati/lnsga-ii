import numpy as np
from Solution import Solution
import copy

class New_Dnsga2:

    # def pt is the number of solutions propogated forward
    def __init__(self, objective_list, constraints, num_vars, min_values, max_values, previous_solution, population_size=100, mutation_percent=.3, 
                 dynamic_parent_percent=.1, dynamic_type='a', zeta = .4, detection_cutoff = 0, nt = 10, tau_t = 5,
                 pt = 5, beginning_lookback = 0, normalize = True):
        self.min_values = min_values
        self.max_values = max_values
        self.objective_list = objective_list
        self.objective_num = len(objective_list)
        self.num_vars = num_vars
        self.population_size = population_size
        self.generation_num = 0
        self.mutation_percent = mutation_percent
        self.constraints = constraints
        self.parent_percent = dynamic_parent_percent
        self.dynamic_type = dynamic_type
        self.zeta = zeta
        self.detection_cutoff = detection_cutoff
        self.nt = nt
        self.tau_t = tau_t
        self.pt = pt
        self.normalize = normalize
        self.prev_sol = previous_solution

        self.solutions = [] #list of solution objects

        for _ in range(self.population_size):
            self.solutions.append(Solution())
        self.solutions = np.array(self.solutions,dtype=Solution)

    def fulfills_constraints(self, population, t):
        constraint_values = np.full((len(population),), True)
        for constraint in self.constraints:
            fulfills = np.apply_along_axis(constraint, 1, population, t)
            constraint_values = np.logical_and(constraint_values, fulfills)
        return constraint_values


    def initialize_solutions(self, recalculate=False, t=0.0, previous_solution=None, keep_solutions=False):
        if type(previous_solution) == list:
            previous_solution = {round(-1/self.nt,5) : previous_solution}
        if not previous_solution is None:
            for i in range(self.population_size):
                # self.solutions[i].solution_dict = self.solutions[i].solution_dict.copy()
                if keep_solutions:
                    curr_sol =  self.solutions[i].solution_dict[t].copy()
                    self.solutions[i].solution_dict = previous_solution.copy()
                    self.solutions[i].solution_dict[t] = curr_sol
                else:
                    self.solutions[i].solution_dict = previous_solution.copy()
        
        if keep_solutions:
            return

        if recalculate:
            replacement_size = min(1, int(self.population_size*self.zeta))
            replacement_arr = np.random.choice(np.arange(self.population_size), (replacement_size,))
            if self.dynamic_type == 'a':
                fulfills_constraints = np.full((replacement_size,), False)
                new_solutions = np.empty((replacement_size, self.num_vars))
                while (np.any(np.logical_not(fulfills_constraints))):
                    for i in range(self.num_vars):
                        #initialize solutions to different values in the given ranges
                        solution_values = np.random.uniform(self.min_values[i], self.max_values[i], 
                                                            (len(np.where(np.logical_not(fulfills_constraints))[0]),))
                        #below is just for integers
                        new_solutions[np.where(np.logical_not(fulfills_constraints)),i] = solution_values
                    #checks whether constraints are fulfilled
                    fulfills_constraints = self.fulfills_constraints(solution_values, 0)
                    if self.normalize:
                        new_solutions = new_solutions/new_solutions.sum(axis=1)
                        
            else:
                new_solutions = self.population[np.random.choice(np.arange(self.population_size), (replacement_size,))]
                if self.normalize:
                    new_solutions = new_solutions/new_solutions.sum(axis=1)
            self.population[replacement_arr] = new_solutions
            for i in range(self.population_size):
                self.solutions[i].solution_dict[t] = self.population[i,:]
            return

        self.population = np.empty((self.population_size, self.num_vars))
        fulfills_constraints = np.full((self.population_size,), False)
        #while the constraints haven't been fulfilled yet
        while (np.any(np.logical_not(fulfills_constraints))):
            for i in range(self.num_vars):
                #initialize solutions to different values in the given ranges

                solution_values = np.random.uniform(self.min_values[i], self.max_values[i], 
                                                            (len(np.where(np.logical_not(fulfills_constraints))[0]),))
                self.population[np.where(np.logical_not(fulfills_constraints)),i] = solution_values
            #checks whether constraints are fulfilled
            fulfills_constraints = self.fulfills_constraints(self.population, 0)
        print(self.population)
        self.population = self.population/np.expand_dims(self.population.sum(axis=1), axis=1)
        print(self.population)
        for i in range(self.population_size):
            self.solutions[i].solution_dict[t] = self.population[i,:]

        return

    def evaluate_fitness(self, t, population=None, solutions=None):
        if population is None:
            population = self.population
            solutions = self.solutions
        else:
            population = population
            solutions = solutions
        objective_values = np.empty((len(population), self.objective_num))

        #sets each row in the objective_value array to the objective values of the solutions
        for i in range(len( self.objective_list)):
            for j in range(len(population)):
                objective_values[j,i] = self.objective_list[i](solutions[j].solution_dict[t],
                                                                t, solutions[j].solution_dict[round(t-1/self.nt,5)])
                print(objective_values[j,i], 'for solution',population[j])
        return objective_values

    def non_dominated_sorting(self, values, population=None):
        if population is None:
            population = self.population
        else:
            population = population
        
        front_assignments = np.zeros((len(population),))
        #ensuring all necessary fronts are calculated
        front_assignments -= 1
        #an nxn matrix that counts how many values each solution dominates
        #column represents what rows are dominated
        dominance_list = np.zeros((len(population), len(population)))
        #for each objective value, check if it dominates other values
        for i in range(len(population)):
            # all values must be geq and there must exist a strictly greater than value
            dominance_boolean = (np.all(values[i] <= values, axis = 1) & 
                np.any(values[i] < values, axis = 1))
            dominance_list[:,i] += dominance_boolean.astype(int)
        front_count = 0
        #until all solutions have assigned fronts
        while (np.any(front_assignments < 0)):
            #sums row-wise, where the non-dominated should have a sum of 0
            dominance_sum = np.sum(dominance_list, axis=1)
            #if dominance sum is 0 and the front hasn't been assigned yet
            solutions_to_add = np.logical_and(dominance_sum == 0, front_assignments == -1)
            #if none can be added, relax standards until at least one solution can belong to the next front
            while not (np.any(solutions_to_add)):
                dominance_sum -= 1
                solutions_to_add = np.logical_and(dominance_sum == 0, front_assignments == -1)
            front_assignments[solutions_to_add] = front_count
            #removing the non-dominated solution columns by changing those columns to 0s
            trues = np.full((self.population_size,), True)
            dominance_list[np.ix_(trues, solutions_to_add)] = 0
            front_count+=1
        return front_assignments
            
    def calculate_crowding_distance(self, values, population=None):
        if population is None:
            population = self.population
        else:
            population = population
        crowding_distance = np.zeros((len(population),))
        for i in range(self.objective_num):
            #sort by ith objective values
            sorted_rows = np.argsort(values[:,i])
            #boundary solutions have an infinite distance (least crowded)
            if (values[[sorted_rows[0]],i] == values[[sorted_rows[-1]],i]):
                crowding_distance[sorted_rows] += 0
            #non-boundary solutions have crowding value based on equation given in paper
            else:
                crowding_distance[[sorted_rows[0], sorted_rows[-1]]] = np.inf
                crowding_distance[sorted_rows[1:-1]] += ((values[sorted_rows[2:],i]-values[sorted_rows[:-2],i])/
                    (values[[sorted_rows[-1]],i]-values[[sorted_rows[0]],i]))
        return crowding_distance

    def mutation(self, child, t):
        mutate = np.random.random((self.num_vars,))
        fulfills = np.full((self.population_size,), False) # working on this for now.
        for i in range(self.num_vars):
            if mutate[i] < self.mutation_percent:
                fulfills = self.fulfills_constraints([child], t)
                value = np.random.uniform(self.min_values[i], self.max_values[i])
                child[i] = value

        return child
    
    
    def crossover(self, parent_1, parent_2):
        #select a random crossover point
        cross_over_point =  np.random.randint(0, self.num_vars)
        child = np.zeros((self.num_vars,))
        #choose objective values up until crossover point from parent 1
        child[:cross_over_point] = parent_1[:cross_over_point]
        #choose objective values after crossover point from parent 2
        child[cross_over_point:] = parent_2[cross_over_point:]
        return child

    def binary_tournament(self, fronts, crowding):
        #generate 2 arrays of parents for binary tournament of length 2*population size
        binary_1 = np.random.randint(0,self.population_size, (2*self.population_size,))
        binary_2 = np.random.randint(0,self.population_size, (2*self.population_size,))

        #array for remaining parents from binary tournament
        parents = np.zeros((2*self.population_size, self.num_vars))
        solutions = np.ndarray((2*self.population_size,), Solution)

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

        solutions[np.where(binary_1_smaller)] = self.solutions[binary_1[np.where(binary_1_smaller)]]
        solutions[np.where(binary_2_smaller)] = self.solutions[binary_2[np.where(binary_2_smaller)]]
        solutions[np.where(binary_1_crowding)] = self.solutions[binary_1[np.where(binary_1_crowding)]]
        solutions[np.where(binary_2_crowding)] = self.solutions[binary_2[np.where(binary_2_crowding)]]

        #setting odd rows as parent 1 and even rows as parent 2
        parents_1 = parents[::2,:]
        parents_2 = parents[1::2,:]

        solutions_1 = solutions[::2,]
        solutions_2 = solutions[1::2]
        return parents_1, parents_2, solutions_1, solutions_2
    
    def solution_crossover(self, solution_1, solution_2, t):
        solution = Solution()
        for i in range(-1,int(t*self.nt+1)):
            rand = np.random.randint(0, 2)
            if rand == 0:
                solution.solution_dict[round(i/self.nt,5)] = solution_1.solution_dict[round(i/self.nt,5)]
            else:
                solution.solution_dict[round(i/self.nt,5)] = solution_2.solution_dict[round(i/self.nt,5)]
        return solution


    def generate_children(self, fronts, crowding, t):
        #binary tournament phase:
        parents_1, parents_2, sols_1, sols_2 = self.binary_tournament(fronts, crowding)
        children = np.empty((len(self.population), self.num_vars))
        solutions = np.ndarray((self.population_size,), Solution)
        for i in range(len(parents_1)):
            child = self.mutation(self.crossover(parents_1[i,:], parents_2[i,:]), t)
            if self.normalize:
                child = child/child.sum()
            child_sol = self.solution_crossover(sols_1[i], sols_2[i], t)
            child_sol.solution_dict[t] = child
            while (np.any(np.all(self.population == child, axis=1))):
                child = self.mutation(self.crossover(parents_1[i,:], parents_2[i,:]), t)
                if self.normalize:
                    child = child/child.sum()
                child_sol.solution_dict[t] = child
            children[i,:] = child
            solutions[i] = child_sol
        return children, solutions


    def generate_new_population(self, combined_pop, combined_sols, t):
        objective_values = self.evaluate_fitness(t, combined_pop, combined_sols)
        fronts = self.non_dominated_sorting(objective_values, combined_pop)
        crowding_values = self.calculate_crowding_distance(objective_values, combined_pop)
        sort_order = np.lexsort((-crowding_values, fronts))
        combined_pop = combined_pop[sort_order]
        combined_sols = combined_sols[sort_order]
        self.solutions = combined_sols[:self.population_size]
        self.population = combined_pop[:self.population_size]

    def select_propogate_values(self, t, solutions, population, last_iteration = False):
        objective_values = self.evaluate_fitness(t, population=population, solutions=solutions)
        fronts = self.non_dominated_sorting(objective_values, population=population)
        crowding_values = self.calculate_crowding_distance(objective_values, population=population)
        sort_order = np.lexsort((-crowding_values, fronts))
        print('sorted for t=',t)
        for i in range(len(sort_order)):
            print(solutions[sort_order][i].solution_dict)

        for i in range(len(solutions)):
            foo = copy.copy(solutions[i])
            solutions[i] = Solution()
            solutions[i].solution_dict = foo.solution_dict.copy()
        
        if last_iteration:
            return solutions[sort_order][:self.population_size]

        return solutions[sort_order][:self.pt]
    
    def generation_step(self, t):
        # get objective values
        objective_values = self.evaluate_fitness(t)
        # assign fronts and crowding values
        fronts = self.non_dominated_sorting(objective_values)
        crowding_values = self.calculate_crowding_distance(objective_values)
        # generate children
        children, child_sols = self.generate_children(fronts, crowding_values, t)
        combined_population = np.concatenate((children, self.population))
        combined_sols = np.concatenate((child_sols, self.solutions))
        # generate new population by taking top solutions
        self.generate_new_population(combined_population, combined_sols, t)

    def original_iteration(self, t):
        for i in range(self.tau_t):
            self.generation_step(t)
        propogate_solutions = self.select_propogate_values(t=t, solutions=self.solutions, population=self.population)

        return propogate_solutions
    
    def future_iteration(self, t, top_solutions, last_iteration = False):
        for i in range(len(top_solutions)):
            foo = copy.copy(top_solutions[i])
            top_solutions[i] = Solution()
            top_solutions[i].solution_dict = foo.solution_dict.copy()
        
        original_population = self.population.copy()
        total_solutions = np.ndarray((self.pt*self.population_size,), 
                                     Solution)
        total_population = np.ndarray((self.pt*self.population_size,self.num_vars))
        print('top solutions for',t, 'len', len(top_solutions))
        for i in range(len(top_solutions)):
            print(top_solutions[i].solution_dict)
        print('\n')
        
        #For each propogated solution
        for i in range(len(top_solutions)):
            self.population = original_population
            self.solutions = self.solutions.copy()
            self.initialize_solutions(t=t,previous_solution=top_solutions[i].solution_dict.copy(), 
                                      keep_solutions=True)
            for j in range(self.tau_t):
                self.generation_step(t)

            for j in range(len(self.solutions)):
                foo = copy.copy(self.solutions[j])
                total_solutions[i*self.population_size+j] = Solution()
                total_solutions[i*self.population_size+j].solution_dict = foo.solution_dict.copy()
        for i in range(self.pt*self.population_size):
            total_population[i] = total_solutions[i].solution_dict[t]
        total_obj_values = self.evaluate_fitness(t, population=total_population, solutions=total_solutions)
        if last_iteration:
            return self.select_propogate_values(t=t, solutions=total_solutions,population=total_population,
                                                                       last_iteration=last_iteration)
        propogate_solutions = self.select_propogate_values(t=t, solutions=total_solutions, population=total_population)
            
        return propogate_solutions

    def run_newdnsga2(self, num_loops):
        self.initialize_solutions(previous_solution=self.prev_sol)
        propogation_sols = self.original_iteration(t=0.0)
        for step in range(1, self.nt): # because there will be nt-1 iterations (not including the 1st iteration)
            t = round(1/self.nt * step,5)
            print('time',t)
            self.initialize_solutions(recalculate=True, t=t)
            propogation_sols = self.future_iteration(t, propogation_sols)
        for i in range(len(propogation_sols)):
            propogation_sols[i].solution_dict[round(-1/self.nt,5)] = self.solutions[i].solution_dict[round(1-1/self.nt,5)]
        for loop in range(1, num_loops):
            for step in range(self.nt): # because there will be nt-1 iterations (not including the 1st iteration)
                t = round(1/self.nt * step,5)
                print('time',t)
                self.initialize_solutions(recalculate=True, t=t)
                if step == self.nt - 1 and loop == num_loops-1:
                    self.solutions = self.future_iteration(t, propogation_sols, last_iteration=True)
                propogation_sols = self.future_iteration(t, propogation_sols)
            for i in range(len(propogation_sols)):
                propogation_sols[i].solution_dict[round(-1/self.nt,5)] = self.solutions[i].solution_dict[round(1-1/self.nt,5)]
        values = []
        for step in range(self.nt):
            t = round(1/self.nt * step,5)
            for i in range(self.population_size):
                self.population[i] = self.solutions[i].solution_dict[t]
                self.solutions[i].solution_dict.pop(self.solutions[i].solution_dict[round(-1/self.nt,5)], None)
            obj_vals = self.evaluate_fitness(t=t)
            values.append(obj_vals)
        return values, self.solutions