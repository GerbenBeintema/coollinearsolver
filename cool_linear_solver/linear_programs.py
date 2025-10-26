from cool_linear_solver.eqs_and_vars import inference, Integer
from cool_linear_solver.linear_solver import System_of_linear_eqs
import numpy as np

class Linear_program:
    def __init__(self):
        self.objective = None
        self.inequality_sys = System_of_linear_eqs()
        self.map = self.inequality_sys.map
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.map

    def set_maximization_objective(self, eq):
        #max f(x) = -min -f(x)
        self.set_minimization_objective(-eq)

    def set_minimization_objective(self, eq):
        assert not eq.is_inequality and not eq.is_equality #what about the map?
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)
        assert self.objective is None, 'Objective already set'
        self.objective = eq

    def add_equality(self, eq):
        assert eq.is_equality
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
                self.set_minimization_objective(eq)
    
    def solve(self, toarray=True, verbose=False, bounds=None):
        #min q x
        # st A_ub x <= b_ub
        #    A_eq x = b_eq
        # assert bounds is not None, 'bounds must be specified like [(x,y)] for each variable or (x,y) for all variables'
        c = np.zeros((len(self.map),),dtype=np.float64)
        for id_now, value in self.objective.coefs.items():
            c[self.map[id_now]] = value
        A_ub = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs>0 else None
        b_ub = np.array(self.inequality_sys.rhs, dtype=np.float64)                 if self.inequality_sys.neqs>0 else None
        A_eq = self.equality_sys.get_sparse_matrix()   if self.equality_sys.neqs>0   else None
        b_eq = np.array(self.equality_sys.rhs, dtype=np.float64)                   if self.equality_sys.neqs>0   else None
        bounds = _extract_bounds_from_ub(A_ub, b_ub) if bounds is None else bounds
        if toarray:
            A_ub = A_ub if A_ub is None else A_ub.toarray()
            A_eq = A_eq if A_eq is None else A_eq.toarray()
        if verbose==1:
            prt = lambda x: None if x is None else x.shape
            print('c',c.__repr__())
            print('A_ub',A_ub.__repr__())
            print('b_ub',prt(b_ub))
            print('A_eq',A_eq.__repr__())
            print('b_eq',prt(b_eq))
        elif verbose==2:
            print('c',c)
            print('A_ub',A_ub)
            print('b_ub',b_ub)
            print('A_eq',A_eq)
            print('b_eq',b_eq)

        from scipy.optimize import linprog
        self.sol_full = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, \
                                method='highs', options={'disp': verbose}, bounds=bounds)
        if not self.sol_full.success:
            raise ValueError(f'Linear program failed: {self.sol_full.message}')
        self.sol = self.sol_full.x

    def __getitem__(self, ids):
        return inference(self.sol, self.map, ids)

def _extract_bounds_from_ub(A_ub, b_ub):
    bounds = [(-np.inf, np.inf)] * A_ub.shape[1]  # Initialize bounds for each variable
    if A_ub is not None and b_ub is not None:
        is_sparse = hasattr(A_ub, 'getrow')
        for i in range(A_ub.shape[0]):
            if is_sparse:
                row = A_ub.getrow(i)
                nz = row.indices
                coef = row.data[0] if len(row.data) == 1 else None
            else:
                row = A_ub[i]
                nz = np.nonzero(row)[0]
                coef = row[nz[0]] if len(nz) == 1 else None
            if len(nz) == 1 and coef is not None:
                j = nz[0]
                rhs = b_ub[i]
                lb, ub = bounds[j]
                if coef > 0:
                    ub_new = rhs / coef
                    bounds[j] = (lb, min(ub, ub_new))
                elif coef < 0:
                    lb_new = rhs / coef
                    bounds[j] = (max(lb, lb_new), ub)
    # Final checks and conversion
    for idx, (lb, ub) in enumerate(bounds):
        if lb > ub:
            raise ValueError(f'Inconsistent bounds for variable {idx}: lower bound {lb} > upper bound {ub}')
        if lb == -np.inf and ub == np.inf:
            # raise ValueError(f'Variable {idx} is unbounded.')
            pass
    # Convert infinities to None for linprog compatibility
    bounds = [(None if lb == -np.inf else lb, None if ub == np.inf else ub) for lb, ub in bounds]
    return bounds

from mip import Model, xsum, MINIMIZE, BINARY, INTEGER
class Mixed_integer_linear_program:
    def __init__(self):
        self.objective = None
        self.inequality_sys = System_of_linear_eqs()
        self.map = self.inequality_sys.map
        self.equality_sys = System_of_linear_eqs()
        self.equality_sys.map = self.map
        self.integer_vars = []
    
    def set_maximization_objective(self, eq):
        self.set_minimization_objective(-eq)

    def set_minimization_objective(self, eq):
        assert not eq.is_inequality and not eq.is_equality #what about the map?
        for id_now, value in eq.coefs.items():
            if self.map.get(id_now) is None:
                self.map[id_now] = len(self.map)
        assert self.objective is None, 'Objective already set'
        self.objective = eq
    
    def add_equality(self, eq):
        assert eq.is_equality
        self.equality_sys.add_equation(eq)
    
    def add_inequality(self, eq):
        assert eq.is_inequality
        self.inequality_sys.add_equation(eq)
    
    def add_integer(self, var): #var is the id
        if isinstance(var, Integer):
            var = var.h
        else:
            var = var == Integer
            var = var.h
        
        if self.map.get(var) is None:
            self.map[var] = len(self.map)
        self.integer_vars.append(var)
    
    def solve(self, bounds=None, verbose=False):
        A_ub = self.inequality_sys.get_sparse_matrix() if self.inequality_sys.neqs>0 else None #number of cols is len of map
        b_ub = np.array(self.inequality_sys.rhs, dtype=np.float64)                 if self.inequality_sys.neqs>0 else None
        bounds = _extract_bounds_from_ub(A_ub, b_ub) if bounds is None else bounds

        from mip import Model, xsum, MINIMIZE, BINARY, INTEGER, CONTINUOUS, OptimizationStatus
        m = Model()
        m.verbose = 1 if verbose else 0
        x = [m.add_var(var_type=INTEGER if var in self.integer_vars else CONTINUOUS, \
                       lb=bounds[id_now][0], ub=bounds[id_now][1]) for var, id_now in self.map.items()]
        m.objective = xsum([self.objective.coefs[id_now]*x[self.map[id_now]] for id_now in self.objective.coefs])
        for eq in self.inequality_sys.eqs:
            m += xsum([eq.coefs[id_now]*x[self.map[id_now]] for id_now in eq.coefs]) <= -eq.constant
        for eq in self.equality_sys.eqs:
            m += xsum([eq.coefs[id_now]*x[self.map[id_now]] for id_now in eq.coefs]) == -eq.constant
        status = m.optimize()
        if status != OptimizationStatus.OPTIMAL:
            raise ValueError(f'Mixed integer linear program failed: {status}')
        self.sol = np.array([var.x for var in x], dtype=np.float64)
    
    def __getitem__(self, ids):
        return inference(self.sol, self.map, ids)



if __name__=='__main__':
    from cool_linear_solver.eqs_and_vars import Variable
    # sys = Linear_program()
    # a = Variable('a')
    # b = Variable('b')
    # c = Variable('c')
    # sys.set_objective(4*a + 5*b + 6*c)
    # sys.add_inequality(a + b >= 11)
    # sys.add_inequality(a - b <= 11)
    # sys.add_equality(c - a - b == 0)
    # sys.add_inequality(7*a >= 35 - 12*b)
    # sys.add_inequality(7*a >= 35 - 11*b)

    # sys.solve(bounds=(0,None))
    # print('a',sys[a])
    # print('b',sys[b])
    # print('c',sys[c])
    
    
    # from mip import Model, xsum, MINIMIZE, BINARY, INTEGER, CONTINUOUS
    # m = Model()
    # x = [m.add_var(var_type=CONTINUOUS) for i in range(3)] #should be the length of map
    # m.objective = xsum([4*x[0], 5*x[1], 6*x[2]])
    # m += x[0] + x[1] >= 11
    # m += x[0] - x[1] <= 11
    # m += x[2] - x[0] - x[1] == 0
    # m += 7*x[0] >= 35 - 12*x[1]
    # m.optimize()
    # print('MIP a', x[0].x)
    # print('MIP b', x[1].x)
    # print('MIP c', x[2].x)
    x = Variable('x')
    sys = Mixed_integer_linear_program()
    sys.set_minimization_objective(4*x[0] + 5*x[1] + 6*x[2])
    sys.add_integer(x[0])
    sys.add_integer(x[1])
    sys.add_inequality(x[0] >= 0)
    sys.add_inequality(x[1] >= 0)
    sys.add_inequality(x[2] >= 0)
    sys.add_inequality(x[0] <= 10)
    sys.add_inequality(x[1] <= 10)
    sys.add_inequality(x[2] <= 10)

    sys.add_inequality(x[0] - x[1] <= 11)
    sys.add_equality(x[2] - x[0] - x[1] == 0)
    sys.add_inequality(7*x[0] >= 35 - 12*x[1])
    sys.solve()
    print('MILP x[0]',sys[x[0]])
    print('MILP x[1]',sys[x[1]])
    print('MILP x[2]',sys[x[2]])