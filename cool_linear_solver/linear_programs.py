from cool_linear_solver.least_squares import Constrained_least_squares
from cool_linear_solver.linear_solver import System_of_linear_eqs
import numpy as np
class Linear_program:
    def __init__(self):
        self.objective = None
        self.inequality_sys = System_of_linear_eqs()
        self.map = self.inequality_sys.map
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.map

    def set_objective(self, eq):
        assert not eq.is_inequality #what about the map?
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)
        self.objective = eq

    def add_equality(self, eq):
        assert not eq.is_inequality
        self.equality_sys.add_equation(eq)

    def add_inequality(self, eq):
        assert eq.is_inequality
        self.inequality_sys.add_equation(eq)
    
    def solve(self, toarray=True, solver='osqp', verbose=False):
        #min q x
        # st G x <= h
        #    A x = b
        from qpsolvers import solve_qp, solve_ls
        q = np.zeros((len(self.map),),dtype=np.float64)
        for id_now, value in self.objective.coefs.items():
            q[self.map[id_now]] = value
        G = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs>0 else None
        h = np.array(self.inequality_sys.rhs, dtype=np.float64)                 if self.inequality_sys.neqs>0 else None
        A = self.equality_sys.get_sparse_matrix()   if self.equality_sys.neqs>0   else None
        b = np.array(self.equality_sys.rhs, dtype=np.float64)                   if self.equality_sys.neqs>0   else None
        if toarray:
            G = G if G is None else G.toarray()
            A = A if A is None else A.toarray()
        if verbose==1:
            prt = lambda x: None if x is None else x.shape
            print('q',q.__repr__())
            print('G',G.__repr__())
            print('h',prt(h))
            print('A',A.__repr__())
            print('b',prt(b))
        elif verbose==2:
            print('q',q)
            print('G',G)
            print('h',h)
            print('A',A)
            print('b',b)

        self.sol = \
            solve_qp(P=None, q=q, G=G, h=h, A=A, b=b, \
                lb=None, ub=None, solver=solver, \
                    initvals=None, sym_proj=False, verbose=False)
        assert self.sol is not None, 'optimization failed'

    def __getitem__(self, ids):
        from collections.abc import Iterable 
        if  isinstance(ids, Iterable):
            return [self[id] for id in ids]
        else:
            return sum(val*self.sol[self.map[el]] for el,val in ids.coefs.items()) + ids.constant

if __name__=='__main__':
    from cool_linear_solver.eqs_and_vars import Variable
    sys = Linear_program()
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    sys.set_objective(4*a + 5*b + 6*c)
    sys.add_inequality(a + b >= 11)
    sys.add_inequality(a - b <= 11)
    sys.add_equality(c - a - b == 0)
    sys.add_inequality(7*a >= 35 - 12*b)
    sys.add_inequality(7*a >= 35 - 12*b)
    sys.add_inequality(a>=0)
    sys.add_inequality(b>=0)
    sys.add_inequality(c>=0)

    sys.solve(verbose=2,solver='cvxopt')
    print('a',sys[a])
    print('b',sys[b])
    print('c',sys[c])
    