
from cool_linear_solver import Variable, quick_solve

a = Variable('a')
b = Variable('b')

eq1 = a+b==2
eq2 = b-a==-1
sol = quick_solve([eq1,eq2])

print('a=',sol[a])
print('b=',sol[b])

print('eq1=', sol[eq1])
print('eq2=', sol[eq2])


T = Variable(name='T')
eq1 = T + T + T + T[1]   + 1*T[3.5]==1
eq2 = T + T     + T[1]   - 4*T[3.5]==4
eq3 = T         + T[1]/10+ 0*T[3.5]==-10.5
sol = quick_solve([eq1,eq2,eq3])
print('T=',sol[T]) #evaluated T on the solution
print('T[1]=',sol[T[1]])
print('T[3.5]=',sol[T[3.5]])
# print('T[2.5]=',sol[T[2.5]]) #this will throw an error for it was not present in the source equations
print('eq1=',sol[eq1]) #you can also evaluate expressions