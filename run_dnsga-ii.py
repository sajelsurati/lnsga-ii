from Dnsga2 import Dnsga_II_discrete
import matplotlib.pyplot as plt

def objective1(x, t):
    return 2*x[0]

def objective2(x, t):
    return -(x[0]-1)**2 - x[1]**2

def equals_constraint(x, t):
    return (x[0] + x[1] == 0)

dnsga = Dnsga_II_discrete(objective_list=[objective1, objective2], constraints=[equals_constraint], 
                 min_values=[-1, -10], max_values=[9, 6], population_size=10, mutation_percent= .4)
values = dnsga.dnsga_ii_discrete(num_iterations=2)
print(values)
plt.scatter(values[:,0], values[:,1])
plt.show()
