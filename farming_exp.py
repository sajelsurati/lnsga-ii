import numpy as np
from NewDnsga2 import New_Dnsga2
import matplotlib.pyplot as plt
import sys
from Solution import Solution

NUM_ACRES = 40

def calculate_cost(x_now, x_past):
    # x0 is soybean x1 is malting barley x2 is oat
    # barley is 1.7*EY - STN - N_pc, soybean credit is 20
    # oat is 1.3*EY - STEN - N_pc, soybean credit is 20

    #in BU/ACRE EY for soybean is 48.25 (calculated from past 10 years of yields) 
    # for barley is 68
    # for oat is 69.1

    # Price of ammonia fertilizer is 1318 per ton
    x_now = x_now / np.sum(x_now)
    x_past = x_past / np.sum(x_past)
    if np.any(x_now[x_now > 1]):
        sys.exit()
    acres_per_crop = x_now*NUM_ACRES
    cost_alphas = [0, 1.7, 1.3]
    yields = np.array([48.25, 68, 69.1])
    # if x_past[0] > 0:
    #     cost_npc = [0, 20, 20] # don't need to add anything to soybeans because 
    total = (cost_alphas*acres_per_crop).dot(yields.T)
    total = total - 20*x_past[0]*NUM_ACRES
    return total

def crop_profit(x_now, t, x_past):
    # print(x_now,t)
    # all prices are measured in per bushel
    oat_prices = [3.05, 3.66, 4.33, 4.53, 2.56, 2.63, 2.49, 2.15, 2.02, 2.01]
    barley_prices = [6.56, 7, 7.07, 5.14, 4.53, 4.4, 4.29, 4.37, 4.4, 5.37]
    soybean_prices = [10, 12.1, 14.1, 13.4, 11, 8.29, 8.4, 9.17, 9.23, 8.75]

    soy_price = np.mean(soybean_prices)*48.25*x_now[0]*NUM_ACRES
    barley_price = np.mean(barley_prices)*68*x_now[1]*NUM_ACRES
    oat_price = np.mean(oat_prices)*69.1*x_now[2]*NUM_ACRES
    total = soy_price+barley_price+oat_price - calculate_cost(x_now, x_past)
    return total


dnsga = New_Dnsga2(objective_list=[crop_profit], constraints=[], 
                 num_vars=3, min_values=([0]*3), max_values=([1]*3), population_size=100, mutation_percent= .4, tau_t=50,
                 previous_solution=[0,0, 1], pt=5, nt=4)
values_1, solutions_1 = dnsga.run_newdnsga2(num_loops = 10)
for val in values_1:
    plt.scatter(val[:,0], val[:,1])
plt.ylim((0,1))
plt.show()