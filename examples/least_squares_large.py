

from cool_linear_solver import Least_squares, Variable

sys = Least_squares()

a = Variable(name='a')
b = Variable(name='b')


sys.add_objective(a+b)
sys.add_objective(a-b)
sys.add_objective(a+a+4)
sys.solve()
print('a',sys[a], 'b',sys[b])
#solution https://www.wolframalpha.com/input?i=minimize+%28a%2Bb%29%5E2%2B%28a-b%29%5E2%2B%282*a%2B4%29%5E2

