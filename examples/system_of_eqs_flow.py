
from cool_linear_solver import Variable, quick_solve

T = Variable(name='T')

#it uses a space solver so you can used it as a PDE solver such as solving the heat equation
import numpy as np
Ny = 2**6+1
Nx = 2*Ny-1
yar = np.linspace(0,1,num=Ny)
xar = np.linspace(-1,1,num=Nx)
dx, dy = xar[1]-xar[0], yar[1]-yar[0]
dxh, dyh = dx/2, dy/2

eqs = []
T = Variable(name='T')

#Variable(name='Fx')
# Fy = Variable(name='Fy')
vx = lambda x, y: 2*y*(1-x**2)
vy = lambda x, y: -2*x*(1-y**2)
lamb = 1e-5


Fx = lambda x, y: vx(x,y)*(T[x-dxh, y]+T[x+dxh, y])/2 - lamb*(T[x+dxh, y]-T[x-dxh, y])/dx
Fy = lambda x, y: vy(x,y)*(T[x, y-dyh]+T[x, y+dyh])/2 - lamb*(T[x, y+dyh]-T[x, y-dyh])/dy


for yi in range(Ny):
    for xi in range(Nx):
        x,y = xar[xi], yar[yi]
        if y==0: 
            if x<=0:
                eqs.append(T[x,y]==1+np.tanh(10*(2*x+1)))
            else:
                eqs.append(T[x,y]==T[x,y+dy])
        elif y==1 or x==-1 or x==1:
            eqs.append(T[x,y]==1-np.tanh(10))
        else: #domain:
            eqs.append(Fx(x+dxh, y) + Fy(x, y+dyh) - Fx(x-dxh, y) - Fy(x, y-dyh)==0)
        
sol = quick_solve(eqs)

Tar = [[T[x,y] for x in xar] for y in yar] #setup array 
Tar = np.array(sol[Tar]) #evaluate 

from matplotlib import pyplot as plt
plt.figure(figsize=(12,5.5))
plt.contourf(xar,yar,Tar, levels=np.linspace(0-1e-2,2,11))
plt.colorbar()
plt.show()
