
from cool_linear_solver import Variable, quick_solve


def run_example(verbose=1):
    a = Variable(name='a')
    b = Variable(name='b')

    eqs = []
    eqs.append((a+b)**2)
    eqs.append((a-b)**2)
    eqs.append((a+a+4)**2)
    
    sol = quick_solve(eqs)
    if verbose:
        print('a', sol[a], 'b', sol[b])
    return sol


if __name__ == '__main__':
    run_example()

