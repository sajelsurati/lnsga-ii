from Dnsga2 import Dnsga_II

def objective1(x):
    return 2*x[0]

def objective2(x):
    return -x[1]+5

dsgna = Dnsga_II([objective1, objective2], [-1, -3], [5, 6], 4)
dsgna.nsga_ii_discrete()