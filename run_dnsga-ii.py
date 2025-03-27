from Dnsga2 import Dnsga_II_discrete
import matplotlib.pyplot as plt
import numpy as np

def objective1(x, t):
    return 2*x[0]

def objective2(x, t):
    return -(x[0]-1)**2 - x[1]**2

def equals_constraint(x, t):
    return (x[0] + x[1] == 0)



def zdt1_f1(x, t):
    return x[0]

def zdt_f2(x, t):
    g = 1 + (9/(len(x)-1))*(np.sum(x)-x[0])
    h = 1-np.sqrt(x[0]/g)
    return (g*h)


def df1_f1(x,t):
    return x[0]

def df1_f2(x,t):
    print("t=",t)
    G = np.abs(np.sin(0.5*np.pi*t))
    H = .75*np.sin(0.5*np.pi*t) + 1.25
    g = 1 + np.sum(np.power((np.array(x)[1:] - G),2))
    return g*(1-np.power((x[0]/g), H))

# def df1_g(x,t):



# dnsga = Dnsga_II_discrete(objective_list=[objective1, objective2], constraints=[], 
#                  num_vars=2, min_values=[-1, -10], max_values=[9, 6], population_size=100, mutation_percent= .4)


# # FOR ZDT1
# dnsga = Dnsga_II_discrete(objective_list=[zdt1_f1, zdt_f2], constraints=[], 
#                  num_vars=2, min_values=([0]*30), max_values=([1]*30), population_size=100, mutation_percent= .4)

dnsga = Dnsga_II_discrete(objective_list=[df1_f1, df1_f2], constraints=[], 
                 num_vars=2, min_values=([0]*2), max_values=([1]*2), population_size=100, mutation_percent= .4, tau_t=10)

solutions = dnsga.dnsga_ii_discrete(num_iterations=500)
# print(solutions)
for sol in solutions:
    plt.scatter(sol[:,0], sol[:,1])
plt.ylim((0,1))
plt.show()
