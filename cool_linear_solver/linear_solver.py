from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve as solve
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
            
    def solve(self):
        if len(self.rhs)>len(self.map):
            raise ValueError(f'too many equations given the number of variables nvars={len(self.map)}, neqs={len(self.rhs)}')
        elif len(self.rhs)<len(self.map):
            raise ValueError(f'too little equations given the number of variables nvars={len(self.map)}, neqs={len(self.rhs)}')
        A = self.get_sparse_matrix()
        self.sol = solve(A, self.rhs)
        if np.any(np.isnan(self.sol)):
            raise ValueError('No solution found, the constructed matrix is probably sigular')

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