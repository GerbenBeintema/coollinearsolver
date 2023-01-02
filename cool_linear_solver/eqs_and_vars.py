
from collections import defaultdict
import copy
import random

eps_hash = 1e-7
global_back_hash = {}
n_unnamed_vars = 0

class Linear_equation(object):
    def __init__(self, coefs=[], constant=0, is_inequality=False, valued=False):
        #coefs is a defaultdict of {id:value}
        self.valued = valued
        self.constant = constant
        self.is_inequality = is_inequality
        if isinstance(coefs,(defaultdict,dict)):
            self.coefs = defaultdict(float, {idel:value for idel, value in coefs.items() if value!=0})
        else:
            self.coefs = defaultdict(float)
            for idel, value in coefs: 
                if value!=0:
                    self.coefs[idel] += value
    
    def __add__(self,other):#self + other
        if isinstance(other,Linear_equation):
            assert self.valued==other.valued
            new_coefs = copy.copy(self.coefs)
            for id_now, value in other.coefs.items():
                new_coefs[id_now] += value
            return Linear_equation(coefs=new_coefs, constant=self.constant+other.constant, valued=self.valued)
        else:
            return Linear_equation(self.coefs,self.constant+other, valued=self.valued)
    
    def __radd__(self,other): #other  + self
        return Linear_equation(self.coefs,self.constant+other)
    
    def __sub__(self,other):
        return (self)+(-other)
    
    def __rsub__(self,other): #other - self
        z = -self
        return z + other
        
    def __neg__(self):
        z = defaultdict(float, [(id_now,-value) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=z,constant=-self.constant, valued=self.valued)
    
    def __repr__(self):
        S = '+'.join([f'{el:.3}*' + global_back_hash[h] for h,el in self.coefs.items()])
        S = S.replace('+-','-').replace('1.0*','').replace('[]','') + f' == {-self.constant:.3f}'
        if self.is_inequality:
            S = S.replace('==','<=')
        return S
        
    def __mul__(self,other):
        if isinstance(other, Linear_equation):
            assert self.valued==True and other.valued==True
            new_coefs = self.coefs.copy()
            for key, item in other.coefs.items():
                new_coefs[key] += item
            new_constant = self.constant*other.constant
            return Linear_equation(coefs=new_coefs,\
                constant=new_constant, valued=self.valued)
        else:
            return Linear_equation(coefs=defaultdict(float, [(id_now,other*value) for id_now, value in self.coefs.items()]),\
                constant=other*self.constant, valued=self.valued)
    def __rmul__(self,other): #other*self
        return self*other
    def __truediv__(self,other): #self/other
        return self*(1/other)
    def __rtruediv__(self,other): #other/self #other is a constant
        assert self.valued
        new_constant = other/self.constant
        new_coefs = defaultdict(float, [(id_now,-value*other/self.constant**2) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=new_coefs, constant=new_constant, valued=self.valued)
    
    def __pow__(self, other):
        assert not isinstance(other, Linear_equation)
        new_constant = self.constant**other
        c = other*self.constant**(other-1)
        new_coefs = defaultdict(float, [(id_now,c*value) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=new_coefs, constant=new_constant, valued=self.valued)
    
    def __eq__(self,other):
        return self - other

    def __le__(self,other):
        #self < other
        s = self - other
        s.is_inequality = True
        return s

    def __ge__(self, other):
        # self > other
        s = other - self
        s.is_inequality = True
        return s

class Variable(Linear_equation):
    def __init__(self,name=None ,value_fun=None , value_only=False):
        self.id_base = random.randint(0,10**10)
        self.is_inequality = False
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
        self.value_only = value_only

        if value_fun is not None :
            self.value_fun = value_fun if callable(value_fun) else (lambda: value_fun)
            self.constant = self.value_fun()
            self.valued = True
        else:
            self.value_fun = None
            self.constant = 0.
            self.valued = False

    def __getitem__(self, x):
        if self.value_only:
            return self.value_fun(*x) if isinstance(x,tuple) else self.value_fun(x)
        # h = hash(x)
        if isinstance(x,tuple):
            h = self.id_base + hash(tuple(int((xi+eps_hash/2)/eps_hash) for xi in x))
            if global_back_hash.get(h)==None:
                global_back_hash[h] = f'{self.name}[{x}]'.replace('(','').replace(')','')
            constant = 0 if self.value_fun is None else self.value_fun(*x)
        else:
            h = self.id_base + hash(int((x+eps_hash/2)/eps_hash))
            if global_back_hash.get(h)==None:
                global_back_hash[h] = f'{self.name}[{x}]'
            constant = 0 if self.value_fun is None else self.value_fun(x)
        
        return Linear_equation(coefs=[(h,1.)],constant=constant, valued=self.valued)



if __name__=='__main__':

    val = 0.5

    for i in range(10):

        x = Variable('x', value_fun=val)
        
        eq = x * x * x - 2 == 0
        print(eq)
        
        from cool_linear_solver.linear_solver import System_of_linear_eqs
        sys = System_of_linear_eqs()
        sys.add_equation(eq)
        print(sys.neqs, sys.rhs, sys.data)
        sys.solve()
        val = sys[x]
        print(i,val, 2**(1/3))


    value = 1
    
    for i in range(10):
        x = Variable('x', value_fun=value)

        eq = x**3
        
        from cool_linear_solver.least_squares import Least_squares
        ls = Least_squares()
        ls.add_objective(eq)
        ls.solve()
        value = ls[x]
        print('x',value)
