
from cool_linear_solver import Constrained_least_squares, Variable

sys = Constrained_least_squares()
a = Variable(name='a')
b = Variable(name='b')
sys.add_objective(a+5*b)
sys.add_objective(a-b)
sys.add_objective(a+a+4)

sys.add_inequality(a>=-1.5)
sys.add_inequality(b<=1.5)
# sys.add_inequality(b<=0)
sys.add_equality(b+2*a==-1)

sys.solve(verbose=1, toarray=True)
print('a',sys[a], 'b',sys[b])