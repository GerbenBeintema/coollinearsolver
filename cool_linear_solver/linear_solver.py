from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, lsmr
from cool_linear_solver.eqs_and_vars import inference

import numpy as np

class System_of_linear_eqs(object):
    def __init__(self):
        self.map = {} #{id,num}
        self.data = []
        self.eqnum = []
        self.varnum = []
        self.rhs = []
        self.eqs = []
    
    @property
    def neqs(self):
        return len(self.rhs)

    def add_equation(self,eq):
        self.eqs.append(eq)
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)
            self.data.append(value)
            self.eqnum.append(len(self.rhs))
            self.varnum.append(self.map[id_now])
        self.rhs.append(-eq.constant)

    def add_equations(self,eqs):
        for eq in eqs:
            self.add_equation(eq)
            
    def solve(self, force_solution=False):
        A = self.get_sparse_matrix().tocsc()
        b = np.array(self.rhs, dtype=np.float64)
        m, n = A.shape

        if not force_solution:
            if m > n:
                raise ValueError(f'too many equations given the number of variables nvars={n}, neqs={m}')
            if m < n:
                raise ValueError(f'too little equations given the number of variables nvars={n}, neqs={m}')

        # Preferred strategies:
        # - when force_solution is False we keep strict checks (raised above)
        # - when force_solution is True we try to produce a reasonable answer
        #   for square/over/underdetermined cases using lsmr as a robust fallback
        x = None

        # If the system is square try direct solve first (fast & exact when possible)
        if m == n and not force_solution:
            x = spsolve(A, b)
        else:
            # If force_solution requested, prefer an iterative least-squares solve which
            # behaves sensibly for over-, under- and square (including inconsistent or
            # rank-deficient cases). lsmr returns an x minimizing ||Ax-b||_2.
            try:
                x = lsmr(A, b)[0]
            except Exception:
                x = None

        # If lsmr didn't produce a usable result, fall back to AA^T route
        if x is None or np.any(np.isnan(x)) or np.any(np.isinf(x)):
            # Try the minimum-norm construction via AA^T (works when m <= n and
            # is often stable to get a finite solution), or normal-equations when
            # appropriate.
            try:
                if m <= n:
                    M = (A @ A.T).tocsc()
                    try:
                        y = spsolve(M, b)
                    except Exception:
                        y = lsmr(M, b)[0]
                    x = A.T.dot(y)
                else:
                    ATA = (A.T @ A).tocsc()
                    rhs = A.T.dot(b)
                    x = spsolve(ATA, rhs)
            except Exception:
                # Last resort: try lsmr again and accept whatever it gives
                x = lsmr(A, b)[0]

        x = np.asarray(x, dtype=np.float64).reshape((n,))
        # For force_solution we return the best-effort x even if residual>0; but
        # still guard against NaN/Inf which indicate solver failure.
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            raise ValueError('No valid solution found (NaN/Inf encountered).')
        self.sol = x

    def get_sparse_matrix(self):
        return csc_matrix(arg1=(self.data, (self.eqnum,self.varnum)),shape=(len(self.rhs),len(self.map)))
    
    def __getitem__(self, ids):
        return inference(self.sol, self.map, ids)

if __name__=='__main__':
    from cool_linear_solver import Variable
    T = Variable('T')
    sys = System_of_linear_eqs()
    sys.add_equation(T[1]+T[2]+T[3]==1)
    sys.add_equation(T[2]==1)
    sys.add_equation(T[1]+T[3] ==2.0)
    sys.solve()