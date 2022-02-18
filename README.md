
## cool_linear_solver: An easy going linear solver with sparse methods and minimal effort
 
usage:

```python
from cool_linear_solver import Variable, System_of_linear_eqs

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
sys.add_equation(eq3) #or use sys.add_equations([eq1,eq2,eq3]) or even `sys = quicksolver([eq1,eq2,eq3])`
sys.solve() #solve using the pushed equations
print('T=',sys[T]) #evaluated T on the solution
print('T[1]=',sys[T[1]])
print('T[3.5]=',sys[T[3.5]])
# print('T[2.5]=',sys[T[2.5]]) #this will throw an error for it was not present in the source equations
print('eq1=',sys[eq1]) #you can also evaluate expressions
```
As can be seen you can use non-integer indexes and the notation is quite simple.

### Heat Equation Solving example

It uses a sparse solver so you can used it as a PDE solver such as solving the heat equation

```python
import numpy as np
N = 200 #creates and 200**2 by 200**2 sparse matrix
Ny = Nx = N
dx, dy = 1/(Nx-1), 1/(Ny-1)
yar = np.linspace(0,1,num=Ny)
xar = np.linspace(0,1,num=Nx)

eqs = System_of_linear_eqs()
T = Variable(name='T')

for yi in range(Ny):
    for xi in range(Nx):
        x,y = xar[xi], yar[yi]
        if x==0:
            if 0.25<y<0.75:
                eqs.add_equation(T[x,y]==1)
            else:
                eqs.add_equation(T[x,y]==0)
        elif y==0 or x==1 or y==1:
            eqs.add_equation(T[x,y]==0)
        else:
            #domain:
            eqs.add_equation(T[x,y]==0.25*(T[xar[xi+1],y] + T[xar[xi-1],y] + T[x,yar[yi+1]] + T[x,yar[yi-1]]))

print(eqs.get_sparse_matrix().__repr__()) # a sparse matrix is automaticly created
eqs.solve()

Tar = []
for yi in range(Ny):
    Trow = []
    for xi in range(Nx):
        x,y = xar[xi], yar[yi]
        Trow.append(eqs[T[x,y]])
    Tar.append(Trow)
from matplotlib import pyplot as plt
plt.contourf(xar,yar,Tar)
plt.colorbar()
plt.show()
```

It can be quite slow in constructing the equations but that is not the goal of this module. 