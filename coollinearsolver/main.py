import random
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve as solve
from collections import defaultdict
import copy

#globals
eps_hash = 1e-7
global_back_hash = {}
n_unnamed_vars = 0

class Linear_equation(object):
    def __init__(self, coefs=[], constant=0):
        #coefs (ids, value)
        self.constant = constant
        if isinstance(coefs,defaultdict):
            self.coefs = coefs
        else:
            self.coefs = defaultdict(float)
            for idel, value in coefs: 
                self.coefs[idel] += value
    
    def __add__(self,other):
        if isinstance(other,Linear_equation):
            new_coefs = copy.copy(self.coefs)
            for id_now, value in other.coefs.items():
                new_coefs[id_now] += value
            return Linear_equation(coefs=new_coefs, constant=self.constant+other.constant)
        else:
            return Linear_equation(self.coefs,self.constant+other)
    
    def __radd__(self,other): #other  + self
        return Linear_equation(self.coefs,self.constant+other)
    
    def __sub__(self,other):
        return (self)+(-other)
    
    def __rsub__(self,other): #other - self
        z = -self
        return z.constant + other
        
    def __neg__(self):
        z = defaultdict(float, [(id_now,-value) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=z,constant=-self.constant)
    
    def __repr__(self):
        S = '+'.join([f'{el:.3}*' + global_back_hash[h] for h,el in self.coefs.items()])
        S = S.replace('+-','-').replace('1.0*','').replace('[]','') + f' == {self.constant:.3f}'
        return S
        
    def __mul__(self,other):
        return Linear_equation(coefs=defaultdict(float, [(id_now,other*value) for id_now, value in self.coefs.items()]),constant=other*self.constant)
    def __rmul__(self,other): #other*self
        return self*other
    def __truediv__(self,other):
        return self*(1/other)
    def __rtruediv__(self,other):
        assert False
    
    def __eq__(self,other):
        return self - other

class Variable(Linear_equation):
    def __init__(self,name=None):
        self.id_base = random.randint(0,10**10)
        if name is not None:
            self.name = name
        else:
            global n_unnamed_vars
            self.name = f'Var{n_unnamed_vars}'
            n_unnamed_vars += 1

        self.coefs = defaultdict(float)
        self.h = self.id_base + hash(tuple())
        self.coefs[self.h] = 1.
        global_back_hash[self.h] = f'{self.name}'
        self.constant = 0.
    
    def __getitem__(self, x):
        if isinstance(x,tuple):
            h = self.id_base + hash(tuple([int((xi+eps_hash/2)/eps_hash) for xi in x]))
            global_back_hash[h] = f'{self.name}[{x}]'.replace('(','').replace(')','')
        else:
            h = self.id_base + hash(int((x+eps_hash/2)/eps_hash))
            global_back_hash[h] = f'{self.name}[{x}]'
        return Linear_equation(coefs=[(h,1.)],constant=0)

class System_of_linear_eqs(object):
    def __init__(self):
        self.map = {} #{id,num}
        self.neqs = 0
        self.data = []
        self.eqnum = []
        self.varnum = []
        self.rhs = []
        
    def push_equation(self,eq):
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)#len(self.map)
            self.data.append(value)
            self.eqnum.append(self.neqs)
            self.varnum.append(self.map[id_now])
        self.rhs.append(-eq.constant)
        self.neqs += 1

    def push_equations(self,eqs):
        for eq in eqs:
            self.push_equation(eq)
            
    def solve(self):
        assert self.neqs==len(self.map), f'too many or too little number of equations nvars={len(self.map)}, neqs={self.neqs}'
        A = self.get_sparse_matrix()
        self.sol = solve(A, self.rhs)

    def get_sparse_matrix(self):
        return csc_matrix(arg1=(self.data, (self.eqnum,self.varnum)),shape=(self.neqs,len(self.map)))
    
    def __getitem__(self, ids):
        return sum(val*self.sol[self.map[el]] for el,val in ids.coefs.items()) + ids.constant

def quicksolve(eqs):
    sys = System_of_linear_eqs()
    sys.push_equations(eqs)
    sys.solve()
    return sys

class Linear_least_sqaures(object):
    def __init__(self):
        self.sol_sys = System_of_linear_eqs()

    def push_equation(self, eq):
        self.sol_sys.push_equation(eq)

    def solve(self):
        R = self.sol_sys.get_sparse_matrix()
        s = self.sol_sys.rhs
        self.sol_sys.sol = solve(R.T@R, R.T@s)

    def __getitem__(self, ids):
        return self.sol_sys[ids]


if __name__ == '__main__':
    sys = Linear_least_sqaures()

    a = Variable(name='a')
    b = Variable(name='b')
    sys.push_equation(a+b)
    sys.push_equation(a-b)
    sys.push_equation(a+a+4)
    sys.solve()
    print('a',sys[a], 'b',sys[b])

    #https://scaron.info/doc/qpsolvers/least-squares.html#qpsolvers.solve_ls


class Constrained_least_squares(object):
    def __init__(self):
        self.objective_sys = System_of_linear_eqs()
        self.inequality_sys = System_of_linear_eqs()
        self.inequality_sys.map = self.objective_sys.map
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.objective_sys.map

    def push_objective(self, eq):
        pass

    def push_equality(self, eq):
        pass

    def push_inequality(self, eq):
        pass




if True and __name__ == '__main__':
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
    sys.push_equation(eq1)
    sys.push_equation(eq2)
    sys.push_equation(eq3) #or use sys.push_equations([eq1,eq2,eq3])
    sys.solve() #solve using the pushed all three equations
    print('T=',sys[T]) #evaluated T on the solution
    print('T[1]=',sys[T[1]])
    print('T[3.5]=',sys[T[3.5]])
    # print('T[2.5]=',sys[T[2.5]]) #this will throw an error for it was not present in the source equations
    print('eq1=',sys[eq1]) #you can also evaluate expressions


    #it uses a space solver so you can used it as a PDE solver such as solving the heat equation
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
                    eqs.push_equation(T[x,y]==1)
                else:
                    eqs.push_equation(T[x,y]==0)
            elif y==0 or x==1 or y==1:
                eqs.push_equation(T[x,y]==0)
            else:
                #domain:
                eqs.push_equation(T[x,y]==0.25*(T[xar[xi+1],y] + T[xar[xi-1],y] + T[x,yar[yi+1]] + T[x,yar[yi-1]]))

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

    