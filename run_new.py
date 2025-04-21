from NewDnsga2 import New_Dnsga2
import matplotlib.pyplot as plt
import numpy as np

def objective1(x, t):
    return 2*x[0]

def objective2(x, t):
    return -(x[0]-1)**2 - x[1]**2

def equals_constraint(x, t):
    return (x[0] + x[1] == 0)



def zdt1_f1(x, t, prev):
    return x[0]

def zdt_f2(x, t, prev):
    g = 1 + (9/(len(x)-1))*(np.sum(x)-x[0])
    h = 1-np.sqrt(x[0]/g)
    return (g*h)



def df1_f1(x,t, prev):
    return x[0]

def df1_f2(x,t, prev):
    G = np.abs(np.sin(0.5*np.pi*t))
    H = .75*np.sin(0.5*np.pi*t) + 1.25
    g = 1 + np.sum(np.power((np.array(x)[1:] - G),2))
    return g*(1-np.power((x[0]/g), H))


def test1_f1(x, t, prev):
    return x[0]

def test1_f2(x, t, prev):
    return x[0] + prev[0]


def test2_f1(x, t, prev):
    return x[0]

def test2_f2(x, t, prev):
    return x[0] * prev[0]

# def df1_g(x,t):



# dnsga = Dnsga_II_discrete(objective_list=[objective1, objective2], constraints=[], 
#                  num_vars=2, min_values=[-1, -10], max_values=[9, 6], population_size=100, mutation_percent= .4)


# # FOR ZDT1
# dnsga = Dnsga_II_discrete(objective_list=[zdt1_f1, zdt_f2], constraints=[], 
#                  num_vars=2, min_values=([0]*30), max_values=([1]*30), population_size=100, mutation_percent= .4)

# dnsga = New_Dnsga2(objective_list=[df1_f1, df1_f2], constraints=[], 
#                  num_vars=2, min_values=([0]*2), max_values=([1]*2), population_size=20, mutation_percent= .4, tau_t=50,
#                  previous_solution=[0,0], pt=5)


dnsga = New_Dnsga2(objective_list=[test1_f1, test1_f2], constraints=[], 
                 num_vars=1, min_values=([0]*1), max_values=([1]*1), population_size=20, mutation_percent= .4, tau_t=50,
                 previous_solution=[0], pt=20)


solutions = dnsga.run_newdnsga2()
# print(solutions)
print('solutions:',len(solutions))
for sol in solutions:
    plt.scatter(sol[:,0], sol[:,1])
plt.ylim((0,1))
plt.show()


# dnsga = New_Dnsga2(objective_list=[df1_f1, df1_f2], constraints=[], 
#                  num_vars=2, min_values=([0]*2), max_values=([1]*2), population_size=100, mutation_percent= .4, tau_t=50)
# solutions = dnsga.run_newdnsga2(num_iterations=500)
# print('solutions:',len(solutions))
# for sol in solutions:
#     plt.scatter(sol[:,0], sol[:,1])
# plt.ylim((0,1))
# plt.show()
# print(len(sol))