
from cool_linear_solver.linear_solver import System_of_linear_eqs
import numpy as np

class Quadratic_problem(object):
    def __init__(self):
        self.inequality_sys = System_of_linear_eqs()
        self.map = self.inequality_sys.map #from id to vector index
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.map
    
    def add_objective(self, quad):
        self.quad = quad 
        #where should this be added?
        Pdata = []
        rows_ids = []
        colums_ids = []
        for (h1, h2),value in quad.quadratic_coefs.items():
            if self.map.get(h1) is None:
                self.map[h1] = len(self.map)
            if self.map.get(h2) is None:
                self.map[h2] = len(self.map)
            Pdata.append(value)
            rows_ids.append(self.map[h1])
            colums_ids.append(self.map[h2])
        for h in quad.linear.coefs:
            if self.map.get(h) is None:
                self.map[h] = len(self.map)
        self._quaddata = (Pdata, (rows_ids, colums_ids))

    def add_equality(self, eq):
        assert not eq.is_inequality
        self.equality_sys.add_equation(eq)

    def add_inequality(self, eq):
        assert eq.is_inequality
        self.inequality_sys.add_equation(eq)
    
    def add_equations(self, eq):
        eqs = [eq] if not isinstance(eq, list) else eq
        for eq in eqs:
            if eq.is_inequality:
                self.add_inequality(eq)
            elif eq.is_equality:
                self.add_equality(eq)
            else:
                self.set_objective(eq)
    
    def get_sparse_matrix(self):
        from scipy.sparse import coo_matrix, csc_matrix
        return csc_matrix(arg1=self._quaddata, shape=[len(self.map),len(self.map)])

    def solve(self, toarray=False, solver='osqp', verbose=False, W=None, lb=None, ub=None):
        # min 0.5*x.T@P@x + q*x
        #st. G x <= h
        #    A x = b
        #    lb <= u <= ub
        from qpsolvers import solve_qp
        P = self.get_sparse_matrix()*2
        q = np.zeros((len(self.map)))
        for h, value in self.quad.linear.coefs.items():
            q[self.map[h]] = value
        G = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs>0 else None
        h = np.array(self.inequality_sys.rhs, dtype=np.float32)                 if self.inequality_sys.neqs>0 else None
        A = self.equality_sys.get_sparse_matrix()   if self.equality_sys.neqs>0   else None
        b = np.array(self.equality_sys.rhs, dtype=np.float32)                   if self.equality_sys.neqs>0   else None
        if toarray:
            P = P.toarray()
            G = G if G is None else G.toarray()
            A = A if A is None else A.toarray()
        if verbose==1:
            prt = lambda x: None if x is None else x.shape
            print('P',P.__repr__())
            print('q',prt(q))
            print('G',G.__repr__())
            print('h',prt(h))
            print('A',A.__repr__())
            print('b',prt(b))
        elif verbose==2:
            print('P',P)
            print('q',q)
            print('G',G)
            print('h',h)
            print('A',A)
            print('b',b)

        self.sol = solve_qp(P,q,G,h,A,b,lb=lb,ub=ub,solver=solver, initvals=None, verbose=False) #add initial vals?

    def __getitem__(self, ids):
        from collections.abc import Iterable 
        if  isinstance(ids, Iterable):
            return [self[id] for id in ids]
        else:
            return sum(val*self.sol[self.map[el]] for el,val in ids.coefs.items()) + ids.constant
