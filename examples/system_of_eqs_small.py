
from cool_linear_solver import Variable, System_of_linear_eqs

a = Variable('a')
b = Variable('b')

sys = System_of_linear_eqs()

eq1 = a+b==2
sys.add_equation(eq1)
eq2 = b-a==-1
sys.add_equation(eq2)
sys.solve()

print('a=',sys[a])
print('b=',sys[b])

print('eq1=', sys[eq1])
print('eq2=', sys[eq2])


T = Variable(name='T')
print(T.coefs)
print(T)
eq1 = T + T + T + T[1]   + 1*T[3.5]==1
eq2 = T + T     + T[1]   - 4*T[3.5]==4
eq3 = T         + T[1]/10+ 0*T[3.5]==-10.5
print(eq1)
print(eq2)
print(eq3)
sys = System_of_linear_eqs()
sys.add_equation(eq1)
sys.add_equation(eq2)
sys.add_equation(eq3) #or use sys.add_equations([eq1,eq2,eq3])
sys.solve() #solve using the pushed all three equations
print('T=',sys[T]) #evaluated T on the solution
print('T[1]=',sys[T[1]])
print('T[3.5]=',sys[T[3.5]])
# print('T[2.5]=',sys[T[2.5]]) #this will throw an error for it was not present in the source equations
print('eq1=',sys[eq1]) #you can also evaluate expressions