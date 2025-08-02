
import numpy as np
from cool_linear_solver import Variable, quick_solve, Linear_equation, Least_squares, System_of_linear_eqs, Quadratic_problem, Linear_program


M = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

p2_list = [[1, 0, 0]]

def f(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.dot(p1, np.dot(M, p2))

V = Variable
lamb = V('lamb')
p1_R = V('p1_R')
p1_P = V('p1_P')
p1_S = V('p1_S')
p1 = [p1_R, p1_P, p1_S]

sys = Linear_program()

for p2 in p2_list:
    sys.add_inequality(f(p1,p2) >= lamb)

sys.add_equality(p1_R + p1_P + p1_S == 1)
sys.set_objective(lamb)

sys.solve(toarray=False, verbose=0, bounds=(0,1))

print(sys)
print('p1_R:', sys[p1_R])
print('p1_P:', sys[p1_P])
print('p1_S:', sys[p1_S])
print('lamb:', sys[lamb])
