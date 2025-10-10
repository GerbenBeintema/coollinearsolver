import numpy as np
from cool_linear_solver import Variable, quick_solve, Linear_equation, Least_squares, System_of_linear_eqs, Quadratic_problem, Linear_program
from cool_linear_solver import Variable, quick_solve, Linear_equation, Least_squares, System_of_linear_eqs, Quadratic_problem, Linear_program


def run_example(verbose=1):
    # M = np.array([
    #     [0, -1, 1],
    #     [1, 0, -1],
    #     [-1, 1, 0]
    # ])
    M = np.array([
        [1, -1],
        [-2, 2]
    ])

    p2_list = np.eye(M.shape[0])

    def f(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.dot(p1, np.dot(M, p2))

    V = Variable
    lamb = V('lamb')
    p1 = V('p1')
    p1 = [p1[i] for i in range(M.shape[0])]

    sys = Linear_program()

    sys.set_maximization_objective(lamb)
    sys.add_equations([f(p1,p2) >= lamb for p2 in p2_list])
    sys.add_equations(sum(p1) == 1)
    sys.add_equations([p1_i >= 0 for p1_i in p1])

    sys.solve(toarray=False, verbose=0)

    if verbose:
        print(sys)
        for p1_i in p1:
            print(f'{p1_i} = {sys[p1_i]}')
        print('lamb:', sys[lamb])
    return sys


if __name__ == '__main__':
    run_example()
