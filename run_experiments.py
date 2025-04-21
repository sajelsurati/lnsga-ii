from Dnsga2 import Dnsga_II_discrete
import matplotlib.pyplot as plt
import numpy as np

def f_1(x, t):
    return x[0]

def f_2(x,t):
    return x[0]

def f_3(x,t):
    return x[0]+x[1]

def f_4(x,t):
    return x[0]*x[1]

def f_5(x,t):
    g = 1 + (9/(len(x)-1))*(np.sum(x)-x[0])
    h = 1-np.sqrt(x[0]/g)
    return g*h

def f_5_with_past(x,t):
    g = 1 + (9/(len(x)-2))*(np.sum(x)-x[0] - x[-1])
    h = 1-np.sqrt(x[0]/g)
    return g*h+x[-1]

dnsga = Dnsga_II_discrete(objective_list=[f_1, f_5], constraints=[], 
                 num_vars=2, min_values=([0]*1), max_values=([1]*1), population_size=100, mutation_percent= .4, tau_t=50)
values_1, solutions_1 = dnsga.dnsga_ii_discrete(num_iterations=500)
for val in values_1:
    plt.scatter(val[:,0], val[:,1])
plt.ylim((0,1))
plt.show()

# dnsga = Dnsga_II_discrete(objective_list=[f_1, f_3], constraints=[], 
#                  num_vars=2, min_values=([0]*2), max_values=([1]*2), population_size=100, mutation_percent= .4, tau_t=50)
# values_2, solutions_2 = dnsga.dnsga_ii_discrete(num_iterations=500)
# print('solutions:',len(solutions_1))
# for val in values_2:
#     plt.scatter(val[:,0], val[:,1])
# plt.ylim((0,1))
# plt.title('Addition')
# # plt.show()

dnsga = Dnsga_II_discrete(objective_list=[f_1, f_5_with_past], constraints=[], 
                 num_vars=3, min_values=([0]*2), max_values=([1]*2), population_size=100, mutation_percent= .4, tau_t=50)
values_2, solutions_2 = dnsga.dnsga_ii_discrete(num_iterations=500)
for val in values_2:
    plt.scatter(val[:,0], val[:,1])
plt.ylim((0,1))
plt.title('Multiplication')
plt.show()

equals = []
print('sols2\n',solutions_2)
print(np.array(solutions_1).shape, np.array(solutions_2).shape)

for i in range(len(solutions_1[0])):
    found = False
    for j in range(len(solutions_2[0])):
        print(solutions_1[0][i][0])
        print(solutions_2[0][j][1])
        if abs(solutions_1[0][i][0] - solutions_2[0][j][1]) < .001:
            found = True
    if found:
        equals.append(True)
    else:
        equals.append(False)

print(equals)
print("total true:",np.sum(equals))
print("total false:",np.sum(np.logical_not(equals)))