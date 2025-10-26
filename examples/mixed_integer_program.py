
from cool_linear_solver import Least_squares, Variable, Integer, quick_solve

def run_example(verbose=1):

    eqs = []
    x = Variable(name='x')
    
    eqs.append(2*x[0] + 3*x[1] +   x[2] - 1)
    eqs.append(x[0]==Integer)
    eqs.append(x[1]==Integer)
    for i in range(3):
        eqs.append(x[i]>=0)
        eqs.append(x[i]<=1)
    
    sol = quick_solve(eqs, verbose=verbose)
    if verbose:
        print('\n=== Mixed Integer Linear Program ===')
        for i in range(3):
            print(f'x[{i}] =', sol[x[i]])
    return sol

if __name__ == '__main__':
    run_example()