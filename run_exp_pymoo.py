import matplotlib.pyplot as plt
import numpy as np
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

from pymoo.visualization.video.callback_video import ObjectiveSpaceAnimation





class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([0]),
                         xu=np.array([1]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        # g = 1 + (9/(len(x)-1))*(np.sum(x)-x[0])
        # h = 1-np.sqrt(x[0]/g)
        # f2 = g*h
        f2 = x[0]

        g1 = 0
        g2 = 0

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

class MyProblem2(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=2,
                         xl=np.array([0,0]),
                         xu=np.array([1,1]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]
        f2 = x[0]-x[1]
        # g = 1 + (9/(len(x)-2))*(np.sum(x)-x[0] - x[-1])
        # h = 1-np.sqrt(x[0]/g)
        # f2 = g*h + x[-1]

        g1 = 0
        g2 = 0

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

problem1 = MyProblem()


algorithm1 = NSGA2(
    pop_size=100,
    # n_offsprings=10,
    # sampling=FloatRandomSampling(),
    # crossover=SBX(prob=0.9, eta=15),
    # mutation=PM(eta=20),
    eliminate_duplicates=True
)


problem2 = MyProblem2()


algorithm2 = NSGA2(
    pop_size=100,
    # n_offsprings=10,
    # sampling=FloatRandomSampling(),
    # crossover=SBX(prob=0.9, eta=15),
    # mutation=PM(eta=20),
    eliminate_duplicates=True
)

# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=1,
#                save_history=True,
#                verbose=True)

res1 = minimize(problem1,
               algorithm1,
               termination=('n_gen', 100),
               seed=1,
               verbose=True)

res2 = minimize(problem2,
               algorithm2,
               termination=('n_gen', 100),
               seed=1,
               verbose=True)


X1 = res1.X
F1 = res1.F

X2 = res2.X
F2 = res2.F

xl, xu = problem1.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(F1[:, 0], F1[:, 1], facecolors='none', edgecolors='r')
plt.xlim(0,1)
plt.ylim(0,1)
plt.title("Design Space")

all_found = []
print(X2)
print(X2[0])
print(X1)
print(X1[0])
for i in range(len(X1)):
    found = False
    for j in range(len(X2)):
        if abs(X1[i][0] - X2[j][1]) < .01:
            found = True
    all_found.append(found)
print(all_found)
print('true:',sum(all_found))
print('false:',np.sum(np.logical_not(all_found)))
plt.show()
