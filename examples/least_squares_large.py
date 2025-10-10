
from cool_linear_solver import Least_squares, Variable


def run_example(verbose=1):
    sys = Least_squares()

    a = Variable(name='a')
    b = Variable(name='b')

    sys.add_objective(a+b)
    sys.add_objective(a-b)
    sys.add_objective(a+a+4)
    sys.solve()
    if verbose:
        print('a',sys[a], 'b',sys[b])
    return sys


if __name__ == '__main__':
    run_example()

