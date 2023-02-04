
from collections import defaultdict
import copy
import random
from numbers import Real

eps_hash = 1e-7
global_back_hash = {}
n_unnamed_vars = 0

class Linear_equation(object):
    def __init__(self, coefs=[], constant=0, is_equality=False, is_inequality=False):
        #coefs is a defaultdict of {hashid:value}
        self.constant = constant
        self.is_equality   = is_equality
        self.is_inequality = is_inequality
        if isinstance(coefs,(defaultdict,dict)):
            self.coefs = defaultdict(float, {idel:value for idel, value in coefs.items() if value!=0}) #copy
        else:
            self.coefs = defaultdict(float)
            for idel, value in coefs:
                if value!=0:
                    self.coefs[idel] += value
    
    def __add__(self,other):#self + other
        assert self.is_inequality==False and self.is_equality==False
        if isinstance(other, Real):
            return Linear_equation(self.coefs,self.constant+other)
        elif isinstance(other,Linear_equation):
            new_coefs = copy.copy(self.coefs)
            for id_now, value in other.coefs.items():
                new_coefs[id_now] += value
            return Linear_equation(coefs=new_coefs, constant=self.constant+other.constant)
        else:
            return NotImplemented
    def __radd__(self,other): #other  + self
        return self + other
    def __sub__(self,other):
        return (self)+(-other) 
    def __rsub__(self,other): #other - self
        return (-self) + other
    def __neg__(self):
        assert self.is_inequality==False and self.is_equality==False
        z = defaultdict(float, [(id_now,-value) for id_now, value in self.coefs.items()])
        return Linear_equation(coefs=z,constant=-self.constant)
    
    def __mul__(self,other):
        assert self.is_inequality==False and self.is_equality==False
        if isinstance(other, Linear_equation):
            lin = self*other.constant + other*self.constant - self.constant*other.constant
            A = []
            for h1, value1 in self.coefs.items():
                for h2, value2 in other.coefs.items():
                    A.append(((h1,h2),value1*value2))
            quad = defaultdict(float, A)
            return Quadratic_equation(quad, lin)
        elif isinstance(other, Real):
            coefs = defaultdict(float, [(id_now,other*value) for id_now, value in self.coefs.items()])
            return Linear_equation(coefs=coefs, constant=other*self.constant)
        else:
            return NotImplemented

    def __rmul__(self,other): #other*self
        return self*other
    def __truediv__(self,other): #self/other
        return self*(1/other)
    
    def __pow__(self, other):
        if other==1:
            return self
        elif other==2:
            return self*self
        else:
            return NotImplemented
    
    def __eq__(self,other):
        s = self - other
        s.is_equality = True
        return s
    def __le__(self,other):
        #self <= other ---> self - other <= 0 
        s = self - other
        s.is_inequality = True
        return s
    def __ge__(self, other):
        # self >= other ---> other - self <= 0
        s = other - self
        s.is_inequality = True
        return s

    def __repr__(self):
        S = ' + '.join([f'{el:.3}*' + global_back_hash[h] for h,el in self.coefs.items()])
        S = S.replace('+ -','- ').replace('1.0*','').replace('[]','') #+ f' == {-self.constant:.3f}'
        if self.is_equality:
            return S + f' == {-self.constant:.3f}'
        elif self.is_inequality:
            return S + f' <= {-self.constant:.3f}'
        else:
            if self.constant==0:
                return S
            elif self.constant>0:
                return S + f' + {self.constant:.3f}'
            else:
                return S + f'{self.constant:.3f}'

class Quadratic_equation:
    def __init__(self, quadratic_coefs, linear):
        #coefs is a defaultdict of {id:value}
        assert isinstance(linear, Linear_equation)
        self.linear = linear
        #quadratic_coefs = default_dict of floats with pairs 

        if isinstance(quadratic_coefs,(defaultdict,dict)):
            self.quadratic_coefs = defaultdict(float, {idel:value for idel, value in quadratic_coefs.items() if value!=0})
        else:
            self.quadratic_coefs = defaultdict(float)
            for idel, value in quadratic_coefs: 
                if value!=0:
                    self.coefs[idel] += value
    
    def __repr__(self):
        S = ' + '.join([f'{el:.3}*' + global_back_hash[h1] + '*' + global_back_hash[h2] for (h1,h2),el in self.quadratic_coefs.items()])
        s = str(self.linear)
        if len(s)>0:
            return S + ' + ' + s
        else:
            return S
    
    def __add__(self, other):
        if isinstance(other, (Real, Linear_equation)):
            return Quadratic_equation(self.quadratic_coefs, self.linear + other)
        elif isinstance(other, Quadratic_equation):
            lin = self.linear + other.linear
            coef_quad = copy.copy(self.quadratic_coefs)
            for (h1, h2), value in other.quadratic_coefs.items():
                coef_quad[h1,h2] += value
            return Quadratic_equation(coef_quad, lin)
        else:
            return NotImplemented
    def __radd__(self,other): #other  + self
        return self + other

    def __mul__(self, other):
        if isinstance(other, Real):
            lin = self.linear*other
            quad = defaultdict(float, [(h, value*other) for h, value in self.quadratic_coefs.items()])
            return Quadratic_equation(quad, lin)
        else:
            return NotImplemented

    def __sub__(self,other):
        return (self)+(-other)
    def __rsub__(self,other): #other - self
        return (-self) + other
    def __neg__(self):
        lin = -self.linear
        quad = defaultdict(float, [(h, -value) for h, value in self.quadratic_coefs.items()])
        return Quadratic_equation(quad, lin)
    def __rmul__(self,other): #other*self
        return self*other
    def __truediv__(self,other): #self/other
        return self*(1/other)
    
class Variable(Linear_equation):
    def __init__(self,name=None):
        self.id_base = random.randint(0,10**10)
        self.is_inequality = False
        self.is_equality = False
        if name is not None:
            self.name = name
        else:
            global n_unnamed_vars
            self.name = f'Var{n_unnamed_vars}'
            n_unnamed_vars += 1        
        h = self.id_base + hash(tuple())
        self.coefs = defaultdict(float, [(h, 1.)])
        global_back_hash[h] = f'{self.name}'
        self.constant = 0.

    def __getitem__(self, x):
        # h = hash(x)
        if isinstance(x,tuple):
            h = self.id_base + hash(tuple(int((xi+eps_hash/2)/eps_hash) for xi in x))
        else:
            h = self.id_base + hash(int((x+eps_hash/2)/eps_hash))
        
        if global_back_hash.get(h)==None:
            global_back_hash[h] = f'{self.name}[{x}]'.replace('(','').replace(')','')
        return Linear_equation(coefs=[(h,1.)], constant=0.)



if __name__=='__main__':

    # x = Variable('x')
    # y = Variable('y')

    # print(x+1)

    # print(x*2 + y)
    # print(x/2 + y)
    # print(str(x*0))
    # # print(x**2)
    # print(x*1 + x*1 - x.constant*x.constant) 

    # print(x*y+2)

    # print(x*y - 2)
    # print((x+1)*y)

    # print(x+1==0)
    # print(0==x+1)

    # print(x+1<=0)
    # print(0>=x+1)

    # print(x+1>=0)
    # print(0<=x+1)

    # print((x**2 + y**2 - (x-y)+2)*2.0)


    x = Variable('x')
    eq = x[1]+x[2]+2
    print(eq)
    eqquad = eq*eq
    print(eqquad)

    eq1 = x[1] + 2
    eq2 = x[2] - 2

    target = eq1**2 + eq2**2

    print(target)
    from cool_linear_solver.quadratic_problems import Quadratic_problem

    sys = Quadratic_problem()
    sys.add_objective(target)
    print(sys.get_sparse_matrix())
    sys.solve(toarray=True, verbose=2)
    print(sys[x[1]], sys[x[2]])

    # tmax = 5000
    # from cool_linear_solver.quadratic_problems import Quadratic_problem
    # sys = Quadratic_problem()

    # u = Variable('u')
    # x = Variable('x')
    # equalities = []
    # sys.add_equality(x[0]==0)
    # cost = 0
    # for t in range(tmax):
    #     sys.add_equality(x[t+1] == 0.8*x[t] + u[t])
    #     cost = cost + 1*u[t]**2
    # sys.add_equality(x[tmax]==10)
    # sys.add_equality(x[tmax//2]==-10)
    # # print(cost, equalities)
    # sys.add_objective(cost)
    # sys.solve(verbose=1,toarray=False, solver='osqp')

    # from matplotlib import pyplot as plt

    # plt.plot([sys[u[t]] for t in range(tmax)])
    # plt.plot([sys[x[t]] for t in range(tmax+1)])
    # plt.show()

    # val = 0.5

    # for i in range(10):

    #     x = Variable('x', value_fun=val)
        
    #     eq = x * x * x - 2 == 0
    #     print(eq)
        
    #     from cool_linear_solver.linear_solver import System_of_linear_eqs
    #     sys = System_of_linear_eqs()
    #     sys.add_equation(eq)
    #     print(sys.neqs, sys.rhs, sys.data)
    #     sys.solve()
    #     val = sys[x]
    #     print(i,val, 2**(1/3))


    # value = 1
    
    # for i in range(10):
    #     x = Variable('x', value_fun=value)

    #     eq = x**3
        
    #     from cool_linear_solver.least_squares import Least_squares
    #     ls = Least_squares()
    #     ls.add_objective(eq)
    #     ls.solve()
    #     value = ls[x]
    #     print('x',value)
