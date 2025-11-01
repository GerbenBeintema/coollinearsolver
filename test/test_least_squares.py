

import numpy as np
from cool_linear_solver import Variable, quick_solve, Constrained_least_squares

def test_least_squares():
    x = Variable('x')

    # eqs = [(x[0]-0.5)**2, x[0]==1]
    # sol_under = quick_solve(eqs)
    # print(sol_under[x[0]])
    # assert abs(sol_under[x[0]] - .5) < 1e-8, f'Underdetermined solution incorrect: {sol_under[x[0]]}'

    sys = Constrained_least_squares()
    sys.add_objective((x[0] + x[1] -0.5)**2)
    sys.add_equality(x[0]==1)
    # sys.add_equality(x[1]==2)
    # sys.add_inequality(x[1]>=0)
    sys.solve(verbose=2, toarray=False)
    print(sys[x[0]])
    print(sys[x[1]])

if __name__ == '__main__':
    test_least_squares()