from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve as solve

import numpy as np

class System_of_linear_eqs(object):
    def __init__(self):
        self.map = {} #{id,num}
        self.data = []
        self.eqnum = []
        self.varnum = []
        self.rhs = []
    
    @property
    def neqs(self):
        return len(self.rhs)

    def add_equation(self,eq):
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)#len(self.map)
            self.data.append(value)
            self.eqnum.append(len(self.rhs))
            self.varnum.append(self.map[id_now])
        self.rhs.append(-eq.constant)

    def add_equations(self,eqs):
        for eq in eqs:
            self.add_equation(eq)
            
    def solve(self):
        assert len(self.rhs)==len(self.map), f'too many or too little number of equations nvars={len(self.map)}, neqs={len(self.rhs)}'
        A = self.get_sparse_matrix()
        self.sol = solve(A, self.rhs)

    def get_sparse_matrix(self):
        return csc_matrix(arg1=(self.data, (self.eqnum,self.varnum)),shape=(len(self.rhs),len(self.map)))
    
    def __getitem__(self, ids):
        return sum(val*self.sol[self.map[el]] for el,val in ids.coefs.items()) + ids.constant