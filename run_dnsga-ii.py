from Dnsga2 import Dnsga_II
import matplotlib.pyplot as plt

def objective1(x):
    return 2*x[0]

def objective2(x):
    return -(x[0]-1)**2 - x[1]**2

def equals_constraint(x):
    return (x[0] + x[1] == 0)

dnsga = Dnsga_II(objective_list=[objective1, objective2], constraints=[equals_constraint], 
                 min_values=[-1, -10], max_values=[9, 6], population_size=4, mutation_percent= .4)
values = dnsga.nsga_ii_discrete(num_iterations=1)
print(values)
plt.scatter(values[:,0], values[:,1])
plt.show()
