

from cool_linear_solver import Variable, System_of_linear_eqs

T = Variable(name='T')

#it uses a space solver so you can used it as a PDE solver such as solving the heat equation
import numpy as np
N = 400 #creates and 200**2 by 200**2 sparse matrix
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

# print(eqs.get_sparse_matrix().__repr__()) # a sparse matrix is automaticly created
eqs.solve()

Tar = []
for yi in range(Ny):
    Trow = []
    for xi in range(Nx):
        x,y = xar[xi], yar[yi]
        Trow.append(eqs[T[x,y]])
    Tar.append(Trow)
# from matplotlib import pyplot as plt
# plt.contourf(xar,yar,Tar)
# plt.colorbar()
# plt.show()
