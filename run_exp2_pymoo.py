import matplotlib.pyplot as plt
import numpy as np
import sys
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.problems.dyn import TimeSimulation
from pymoo.optimize import minimize
from pymoo.core.callback import CallbackCollection
from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.problems.dynamic.df import DF1
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination.robust import RobustTermination

from pymoo.visualization.video.callback_video import ObjectiveSpaceAnimation
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
# def calculate_cost(x_now, x_past):
#     # x0 is soybean x1 is malting barley x2 is oat
#     # barley is 1.7*EY - STN - N_pc, soybean credit is 20
#     # oat is 1.3*EY - STEN - N_pc, soybean credit is 20

#     #in BU/ACRE EY for soybean is 48.25 (calculated from past 10 years of yields) 
#     # for barley is 68
#     # for oat is 69.1

#     # Price of ammonia fertilizer is 1318 per ton
#     x_now = x_now / np.sum(x_now)
#     x_past = x_past / np.sum(x_past)
#     acres_per_crop = x_now*NUM_ACRES
#     cost_alphas = [0, 1.7, 1.3]
#     yields = np.array([48.25, 68, 69.1])

#     # if x_past[0] > 0:
#     #     cost_npc = [0, 20, 20] # don't need to add anything to soybeans because 
#     total = (cost_alphas*acres_per_crop).dot(yields.T)
#     total = total - 20*x_past[0]*NUM_ACRES
#     return total*.659 # multiply total fertilizer needed by price

# def crop_profit(x_now, x_past):
#     # print(x_now,t)
#     # all prices are measured in per bushel
#     oat_prices = [3.05, 3.66, 4.33, 4.53, 2.56, 2.63, 2.49, 2.15, 2.02, 2.01]
#     barley_prices = [6.56, 7, 7.07, 5.14, 4.53, 4.4, 4.29, 4.37, 4.4, 5.37]
#     soybean_prices = [10, 12.1, 14.1, 13.4, 11, 8.29, 8.4, 9.17, 9.23, 8.75]

#     soy_price = np.mean(soybean_prices)*48.25*x_now[0]*NUM_ACRES
#     barley_price = np.mean(barley_prices)*68*x_now[1]*NUM_ACRES
#     oat_price = np.mean(oat_prices)*69.1*x_now[2]*NUM_ACRES
#     total = soy_price+barley_price+oat_price - calculate_cost(x_now, x_past)
#     return -total

def shannon_entropy(x_now):
    log = np.log(x_now)
    px = x_now*log
    return np.sum(px)

# def f0(x):
#     profit = crop_profit(x[0:3], [0,0,1])
#     return profit

# def f1(x):
#     profit = crop_profit(x[3:6], x[0:3])
#     return profit

# def f2(x):
#     profit = crop_profit(x[6:9], x[3:6])
#     return profit

# def f3(x):
#     profit = crop_profit(x[9:12], x[6:9])
#     return profit



class MyProblem(ElementwiseProblem):

    # n_var=12,
    #                      n_obj=4,
    #                      n_eq_constr=4,
    #                      xl=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    #                      xu=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # n_var=3,
                        #  n_obj=1,
                        #  n_eq_constr=1,
                        #  xl=np.array([0, 0, 0 ]),
                        #  xu=np.array([1, 1, 1])
    def __init__(self):
        super().__init__( n_var=12,
                         n_obj=2,
                         n_eq_constr=4,
                         xl=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         xu=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    def _evaluate(self, x, out, *args, **kwargs):
        f0 = crop_profit(x[0:3], [0,0,1])
        f1 = crop_profit(x[3:6], x[0:3])
        f2 = crop_profit(x[6:9], x[3:6])
        f3 = crop_profit(x[9:12], x[6:9])
        f4 = shannon_entropy(x[0:3])
        f5 = shannon_entropy(x[3:6])
        f6 = shannon_entropy(x[6:9])
        f7 = shannon_entropy(x[9:12])

        h0 = 1 - np.sum(x[0:3])
        h1 = 1 - np.sum(x[3:6])
        h2 = 1 - np.sum(x[6:9])
        h3 = 1 - np.sum(x[9:12])

        # g1 = 0
        # g2 = 0

        # out["F"] = [f0]
        # out["H"] = [h0]

        out["F"] = [f0 + f1 + f2 +f3, f4 + f5 + f6 + f7]
        out["H"] = [h0, h1, h2, h3]

# class MyProblem2(ElementwiseProblem):

#     def __init__(self):
#         super().__init__(n_var=2,
#                          n_obj=2,
#                          n_ieq_constr=2,
#                          xl=np.array([0,0]),
#                          xu=np.array([1,1]))

#     def _evaluate(self, x, out, *args, **kwargs):
#         f1 = x[0]
#         f2 = x[0]-x[1]
#         # g = 1 + (9/(len(x)-2))*(np.sum(x)-x[0] - x[-1])
#         # h = 1-np.sqrt(x[0]/g)
#         # f2 = g*h + x[-1]

#         g1 = 0
#         g2 = 0

#         out["F"] = [f1, f2]
#         out["G"] = [g1, g2]

# problem1 = MyProblem()


# algorithm1 = NSGA2(
#     pop_size=100,
#     # n_offsprings=10,
#     # sampling=FloatRandomSampling(),
#     # crossover=SBX(prob=0.9, eta=15),
#     # mutation=PM(eta=20),
#     eliminate_duplicates=True
# )

# res1 = minimize(problem1,
#                algorithm1,
#                termination=('n_gen', 100),
               
#                verbose=True)

# res2 = minimize(problem2,
#                algorithm2,
#                termination=('n_gen', 100),
#                seed=1,
#                verbose=True)
problem1 = MyProblem()
algorithm1 = NSGA2(
    pop_size=100,
    # n_offsprings=10,
    # sampling=FloatRandomSampling(),
    # crossover=SBX(prob=0.9, eta=15),
    # mutation=PM(eta=20),
    eliminate_duplicates=True
)

res1 = minimize(problem1,
            algorithm1,
            termination = RobustTermination(DesignSpaceTermination(tol=0.01), period=20),
            verbose=False)

results = res1.X
print(results)

for _ in range(9):
    problem1 = MyProblem()
    algorithm1 = NSGA2(
        pop_size=100,
        # n_offsprings=10,
        # sampling=FloatRandomSampling(),
        # crossover=SBX(prob=0.9, eta=15),
        # mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res1 = minimize(problem1,
               algorithm1,
               termination = RobustTermination(DesignSpaceTermination(tol=0.00001), period=20),
               verbose=True)
    
    X1 = res1.X
    F1 = res1.F
    results = np.concatenate((results, X1))
    print(results)
print(np.array(results).shape)
print('stddev\n',np.std(results, axis=0))
print('mean\n',np.mean(results, axis=0))

# X2 = res2.X
# F2 = res2.F

xl, xu = problem1.bounds()
plt.figure(figsize=(7, 5))
# plt.scatter(F1[:, 0], F1[:, 1], facecolors='none', edgecolors='r')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.title("Design Space")

all_found = []
# print(X2)
# print(X2[0])
print(X1)
for i in range(len(X1)):
    print(X1[i,:])
# for i in range(len(X1)):
#     found = False
#     for j in range(len(X2)):
#         if abs(X1[i][0] - X2[j][1]) < .01:
#             found = True
#     all_found.append(found)
# print(all_found)
# print('true:',sum(all_found))
# print('false:',np.sum(np.logical_not(all_found)))
# plt.show()
