
from cool_linear_solver.linear_solver import System_of_linear_eqs
import numpy as np
from scipy.sparse.linalg import spsolve as solve

import gym

#https://scaron.info/doc/qpsolvers/least-squares.html#qpsolvers.solve_ls
class Constrained_least_squares(object):
    def __init__(self):
        self.objective_sys = System_of_linear_eqs()
        self.map = self.objective_sys.map
        self.inequality_sys = System_of_linear_eqs()
        self.inequality_sys.map = self.map
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.map

    def add_objective(self, eq):
        assert not eq.is_inequality
        self.objective_sys.add_equation(eq)

    def add_equality(self, eq):
        assert not eq.is_inequality
        self.equality_sys.add_equation(eq)

    def add_inequality(self, eq):
        assert eq.is_inequality
        self.inequality_sys.add_equation(eq)

    def solve(self, toarray=True, solver='quadprog', verbose=False, W=None, lb=None, ub=None):
        #min 1/2 |(R x - s)|^2_W
        # st G x <= h
        #    A x = b
        from qpsolvers import solve_qp, solve_ls
        R = self.objective_sys.get_sparse_matrix()
        s = np.array(self.objective_sys.rhs, dtype=np.float)
        G = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs>0 else None
        h = np.array(self.inequality_sys.rhs, dtype=np.float)                 if self.inequality_sys.neqs>0 else None
        A = self.equality_sys.get_sparse_matrix()   if self.equality_sys.neqs>0   else None
        b = np.array(self.equality_sys.rhs, dtype=np.float)                   if self.equality_sys.neqs>0   else None
        if toarray:
            R = R.toarray()
            G = G if G is None else G.toarray()
            A = A if A is None else A.toarray()
        if verbose==1:
            prt = lambda x: None if x is None else x.shape
            print('R',R.__repr__())
            print('s',prt(s))
            print('G',G.__repr__())
            print('h',prt(h))
            print('A',A.__repr__())
            print('b',prt(b))
        elif verbose==2:
            print('R',R)
            print('s',s)
            print('G',G)
            print('h',h)
            print('A',A)
            print('b',b)

        self.sol = \
            solve_ls(R, s, G=G, h=h, A=A, b=b, \
                lb=lb, ub=ub, W=W, solver=solver, initvals=None, sym_proj=False, verbose=False)
        assert self.sol is not None, 'optimization failed'

    def __getitem__(self, ids):
        return sum(val*self.sol[self.map[el]] for el,val in ids.coefs.items()) + ids.constant

class Least_squares(object):
    def __init__(self):
        self.sol_sys = System_of_linear_eqs()

    def add_objective(self, eq):
        self.sol_sys.add_equation(eq)

    def solve(self):
        R = self.sol_sys.get_sparse_matrix()
        s = self.sol_sys.rhs
        self.sol_sys.sol = solve(R.T@R, R.T@s)

    def __getitem__(self, ids):
        return self.sol_sys[ids]
