
from collections import defaultdict
import copy
import random

eps_hash = 1e-7
global_back_hash = {}
n_unnamed_vars = 0

class Linear_equation(object):
    def __init__(self, coefs=[], constant=0, is_inequality=False):
        #coefs (ids, value)
        self.constant = constant
        self.is_inequality = is_inequality
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
        return z + other
        
    def __neg__(self):
        z = defaultdict(float, [(id_now,-value) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=z,constant=-self.constant)
    
    def __repr__(self):
        S = '+'.join([f'{el:.3}*' + global_back_hash[h] for h,el in self.coefs.items()])
        S = S.replace('+-','-').replace('1.0*','').replace('[]','') + f' == {-self.constant:.3f}'
        if self.is_inequality:
            S = S.replace('==','<=')
        return S
        
    def __mul__(self,other):
        return Linear_equation(coefs=defaultdict(float, [(id_now,other*value) for id_now, value in self.coefs.items()]),\
            constant=other*self.constant)
    def __rmul__(self,other): #other*self
        return self*other
    def __truediv__(self,other):
        return self*(1/other)
    def __rtruediv__(self,other):
        assert False
    
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
    def __init__(self,name=None):
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
        self.constant = 0.
    
    def __getitem__(self, x):
        # h = hash(x)
        if isinstance(x,tuple):
            h = self.id_base + hash(tuple(int((xi+eps_hash/2)/eps_hash) for xi in x))
            if global_back_hash.get(h)==None:
                global_back_hash[h] = f'{self.name}[{x}]'.replace('(','').replace(')','')
        else:
            h = self.id_base + hash(int((x+eps_hash/2)/eps_hash))
            if global_back_hash.get(h)==None:
                global_back_hash[h] = f'{self.name}[{x}]'
        return Linear_equation(coefs=[(h,1.)],constant=0)