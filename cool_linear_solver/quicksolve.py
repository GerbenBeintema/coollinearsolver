
from cool_linear_solver.eqs_and_vars import Linear_equation, Quadratic_equation, Variable, Linear_squared_equation

from cool_linear_solver.least_squares import Constrained_least_squares, Least_squares
from cool_linear_solver.quadratic_problems import Quadratic_problem
from cool_linear_solver.linear_solver import System_of_linear_eqs
import numpy as np

def quick_solve(list_of_eqs, **solver_kwargs):
    assert len(list_of_eqs)>0
    Q_obj = [] # Quadratic objectives
    L_obj = [] # Linear objectives
    Lq_obj = [] # Linear quadratic objectives
    L_ieq = [] # Linear inequalities
    L_eq = []  # Linear equalities
    for eq in list_of_eqs:
        if isinstance(eq, Linear_squared_equation):
            Lq_obj.append(eq)
        elif isinstance(eq, Quadratic_equation):
            Q_obj.append(eq)
        elif isinstance(eq, Linear_equation):
            if eq.is_equality:
                L_eq.append(eq)
            elif eq.is_inequality:
                L_ieq.append(eq)
            else:
                L_obj.append(eq)
        else:
            raise ValueError(f'{eq} is not an Quadratic or Linear equation')
    print(f'quick_solve detected: {len(Q_obj)} Quadratic objectives, {len(L_ieq)} Linear inequalities, {len(Lq_obj)} Linear squared objectives, {len(L_obj)} Linear objectives, {len(L_eq)} Linear equalities')
    # assert len(Q_obj)<=1, 'only one Quadratic term allowed'
    # assert len(L_obj)<=1, 'only one Linear objective allowed'
    # assert len(Lq_obj)<=1, 'only one Linear squared objective allowed'
    assert bool(Q_obj) + bool(L_obj) + bool(Lq_obj) <= 1, 'only one type of objective allowed'

    if len(Q_obj)==0 and len(L_ieq)==0 and len(L_obj)==0 and len(Lq_obj)==0 and len(L_eq)>0: #Linear system of equations
        sys = System_of_linear_eqs()
        sys.add_equations(L_eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_obj)==0 and len(L_ieq)==0 and len(Lq_obj)>0 and len(L_obj)==0 and len(L_eq)==0: #Least Squares
        sys = Least_squares()
        for eq in Lq_obj:
            sys.add_objective(eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_obj)==0 and len(L_obj)==0 and len(Lq_obj)>0: #Constrainted Least Squares
        sys = Constrained_least_squares()
        for eq in Lq_obj:
            sys.add_objective(eq)
        for eq in L_ieq:
            sys.add_inequality(eq)
        for eq in L_eq:
            sys.add_equality(eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_obj)>=1 and len(L_obj)==0 and len(Lq_obj)==0: #Quadratic problem
        sys = Quadratic_problem()
        sys.add_objective(sum(Q_obj))
        for eq in L_ieq:
            sys.add_inequality(eq)
        for eq in L_eq:
            sys.add_equality(eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(L_obj)>=1 and len(Q_obj)==0 and len(Lq_obj)==0: #Linear program
        from cool_linear_solver.linear_programs import Linear_program
        sys = Linear_program()
        sys.set_minimization_objective(sum(L_obj))
        for eq in L_ieq:
            sys.add_inequality(eq)
        for eq in L_eq:
            sys.add_equality(eq)
        sys.solve(**solver_kwargs)
        return sys
    else:
        raise ValueError(f'Cannot find solver for len(Q_obj)={len(Q_obj)}, len(L_ieq)={len(L_ieq)}, len(L_obj)={len(L_obj)}, len(L_eq)=={len(L_eq)}')

def _validate_quicksolve(sol, eqs, epsilon=1e-6):
    for eq in eqs:
        if isinstance(eq, Linear_equation):
            res = sol[eq]
            if eq.is_equality:
                assert abs(res)<epsilon, f'Equality {eq} not satisfied, residual={res}'
            elif eq.is_inequality:
                assert res<=epsilon, f'Inequality {eq} not satisfied, residual={res}'
            else:
                pass #objective
        elif isinstance(eq, Quadratic_equation):
            pass #objective
        else:
            raise ValueError(f'{eq} is not an Quadratic or Linear equation')

def _test_quicksolve(verbose=1):
    x = Variable('x')
    eqs = [x[0] + x[1] + 2*x[2] == 1,
            3*x[0] + x[1] + 0.5*x[2] == 2,
            x[0] + 2*x[1] + 0 == 3]
    sys = quick_solve(eqs)
    if verbose:
        print('\n=== Least system of equations ===')
        print(f'solver: {type(sys).__name__}')
        print('variables:', [x[0], x[1], x[2]])
        print('values:', [sys[xi] for xi in (x[0], x[1], x[2])])
        print('equation evaluations:', [sys[e] for e in eqs])
    assert all(abs(sys[e])<1e-6 for e in eqs), 'System of equations not satisfied'
    _validate_quicksolve(sys, eqs)

    eqs = [(x[0] + x[1] + 2*x[2]+2)**2,
            (3*x[0] + x[1] + 0.5*x[2]+2)**2,
            (x[0] + 2*x[1] + 0+4)**2,
            (x[1] + x[2]+4)**2]
    sys = quick_solve(eqs)
    if verbose:
        print('\n=== Least Squares ===')
        print(f'solver: {type(sys).__name__}')
        print('variables:', [x[0], x[1], x[2]])
        print('values:', [sys[xi] for xi in (x[0], x[1], x[2])])
        print('equation residuals:', [sys[e] for e in eqs])
    _validate_quicksolve(sys, eqs)
    R = sys.sol_sys.get_sparse_matrix()
    s = sys.sol_sys.rhs
    assert np.allclose(R.T@(R @ sys.sol_sys.sol - s), 0), 'Least squares solution does not minimize the residuals'

    eqs = [x[0] + x[1] + x[2], x[0] + 2*x[1] + 3*x[2]==4, x[1]+x[2]>=1, x[0]>=0, x[1]>=0, x[2]>=0, x[0]<=10, x[1]<=10, x[2]<=10]
    sys = quick_solve(eqs)
    if verbose:
        print('\n=== Linear Program ===')
        print(f'solver: {type(sys).__name__}')
        print('variables:', [x[0], x[1], x[2]])
        print('values:', [sys[xi] for xi in (x[0], x[1], x[2])])
        print('equation evaluations:', [sys[e] for e in eqs])
    _validate_quicksolve(sys, eqs)
    
    for toarray in (True, False):
        print(f'\n--- toarray={toarray} ---')
        eqs = [(x[0] + x[1] + 2*x[2]+2)**2,
                (3*x[0] + x[1] + 0.5*x[2]+2)**2,
                (x[0] + 2*x[1] + 0+4)**2,
                x[1] + x[2]+9==2]
        sys = quick_solve(eqs, toarray=toarray) #I'm getting a Segmentation fault if using solver='osqp' and toarray=False
        if verbose:
            print('\n=== Constrained Least Squares ===')
            print(f'solver: {type(sys).__name__}')
            print('variables:', [x[0], x[1], x[2]])
            print('values:', [sys[xi] for xi in (x[0], x[1], x[2])])
            print('equation evaluations:', [sys[e] for e in eqs])
        _validate_quicksolve(sys, eqs)

        eqs = [x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[1]+x[2], x[0]+x[1]+x[2]==1]
        sys = quick_solve(eqs, toarray=toarray)
        if verbose:
            print('\n=== Quadratic problem ===')
            print(f'solver: {type(sys).__name__}')
            print('variables:', [x[0], x[1], x[2]])
            print('values:', [sys[xi] for xi in (x[0], x[1], x[2])])
        _validate_quicksolve(sys, eqs)

if __name__=='__main__':
    _test_quicksolve()