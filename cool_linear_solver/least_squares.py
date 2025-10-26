
from cool_linear_solver.linear_solver import System_of_linear_eqs
from cool_linear_solver.eqs_and_vars import inference, Linear_squared_equation
import numpy as np

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
        assert isinstance(eq, Linear_squared_equation)
        self.objective_sys.add_equation(eq._orig_linear)

    def add_equality(self, eq):
        assert not eq.is_inequality
        self.equality_sys.add_equation(eq)

    def add_inequality(self, eq):
        assert eq.is_inequality
        self.inequality_sys.add_equation(eq)

    def solve(self, toarray=False, solver='osqp', verbose=False, W=None, lb=None, ub=None):
        #min 1/2 |(R x - s)|^2_W
        # st G x <= h
        #    A x = b
        from qpsolvers import solve_qp
        from scipy.sparse import csc_matrix, issparse
        import numpy as _np

        R = self.objective_sys.get_sparse_matrix()
        s = _np.array(self.objective_sys.rhs, dtype=_np.float64)
        G = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs > 0 else None
        h = None if G is None else _np.array(self.inequality_sys.rhs, dtype=_np.float64)
        A = self.equality_sys.get_sparse_matrix() if self.equality_sys.neqs > 0 else None
        b = None if A is None else _np.array(self.equality_sys.rhs, dtype=_np.float64)

        if toarray:
            # allow user to inspect dense intermediates, but we'll still pass CSC to the solver
            R = R.toarray()
            G = None if G is None else (G.toarray() if hasattr(G, 'toarray') else G)
            A = None if A is None else (A.toarray() if hasattr(A, 'toarray') else A)

        if verbose:
            prt = (lambda x: None if x is None else getattr(x, 'shape', None))
            if verbose == 2:
                print('R', R)
                print('s', s)
                print('G', G)
                print('h', h)
                print('A', A)
                print('b', b)
            else:
                print('R', getattr(R, '__repr__', lambda: None)(), type(R))
                print('s', prt(s), type(s))
                print('G', getattr(G, '__repr__', lambda: None)(), type(G))
                print('h', prt(h), type(h))
                print('A', getattr(A, '__repr__', lambda: None)(), type(A))
                print('b', prt(b), type(b))

        # Convert least-squares to quadratic form: P = 2 R.T R, q = -2 R.T s
        P = (R.T @ R) * 2
        q = -2 * (R.T @ s)

        # coerce vectors and sparse matrices to solver-friendly types
        q = _np.asarray(q, dtype=_np.float64)
        h = None if h is None else _np.asarray(h, dtype=_np.float64)
        b = None if b is None else _np.asarray(b, dtype=_np.float64)

        def _as_csc(M):
            if M is None:
                return None
            return M.tocsc() if issparse(M) else csc_matrix(M)

        P = _as_csc(P)
        G = _as_csc(G)
        A = _as_csc(A)

        self.sol = solve_qp(P, q, G, h, A, b, lb=lb, ub=ub, solver=solver, initvals=None, verbose=False)
        assert self.sol is not None, 'optimization failed'

    def __getitem__(self, ids):
        return inference(self.sol, self.map, ids)

class Least_squares(object):
    def __init__(self):
        self.sol_sys = System_of_linear_eqs()

    def add_objective(self, eq):
        assert isinstance(eq, Linear_squared_equation)
        self.sol_sys.add_equation(eq._orig_linear)

    def solve(self):
        R = self.sol_sys.get_sparse_matrix()
        s = self.sol_sys.rhs
        from scipy.sparse.linalg import spsolve as solve
        self.sol_sys.sol = solve(R.T@R, R.T@s)

    def __getitem__(self, ids):
        return inference(self.sol_sys.sol, self.sol_sys.map, ids)
