import numpy as np
from NewDnsga2 import New_Dnsga2
import matplotlib.pyplot as plt
import sys
from Solution import Solution

def calculate_cost(x_now, x_past):
    # x0 is beans x1 is malting barley x2 is oat
    # barley is 1.7*EY - STN - N_pc, bean credit is 10
    # oat is 1.3*EY - STEN - N_pc, bean credit is 10

    #in BU/ACRE EY for bean is 2167 (calculated from past 6 years of yields) 
    # for barley is 68
    # for oat is 69.1

    # Price of ammonia fertilizer is 1318 per ton
    x_now = x_now / np.sum(x_now)
    x_past = x_past / np.sum(x_past)
    if np.any(x_now[x_now > 1]):
        sys.exit()
    acres_per_crop = x_now
    cost_alphas = [0.05, 1.7, 1.3]

    yields = np.array([22, 68, 69.1])

    total = (cost_alphas*acres_per_crop).dot(yields.T)
    total = total - 10*x_past[0]
    return total*.659 # multiply total fertilizer needed by price

def crop_profit(x_now, x_past):
    # print(x_now,t)
    # all prices are measured in per bushel
    oat_prices = [3.05, 3.66, 4.33, 4.53, 2.56, 2.63, 2.49, 2.15, 2.02, 2.01]
    barley_prices = [6.56, 7, 7.07, 5.14, 4.53, 4.4, 4.29, 4.37, 4.4, 5.37]

    bean_price = 37.8*22*x_now[0]
    barley_price = np.mean(barley_prices)*68*x_now[1]
    oat_price = np.mean(oat_prices)*69.1*x_now[2]
    total = bean_price+barley_price+oat_price - calculate_cost(x_now, x_past)
    return -total

def shannon_entropy(x_now, x_past):
    log = np.log(x_now)
    px = x_now*log
    return np.sum(px)


dnsga = New_Dnsga2(objective_list=[crop_profit], constraints=[], 
                 num_vars=3, min_values=([0]*3), max_values=([1]*3), population_size=10, mutation_percent= .4, tau_t=50,
                 previous_solution=[0,1,0], pt=5, nt=4)
values_1, solutions_1 = dnsga.run_newdnsga2(num_loops = 10)
# for val in values_1:
#     plt.scatter(val[:,0], val[:,1])
# plt.ylim((0,1))
# plt.show()