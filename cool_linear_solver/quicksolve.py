
from cool_linear_solver.eqs_and_vars import Linear_equation, Quadratic_equation, Variable

from cool_linear_solver.least_squares import Constrained_least_squares, Least_squares
from cool_linear_solver.quadratic_problems import Quadratic_problem
from cool_linear_solver.linear_solver import System_of_linear_eqs

def quick_solve(list_of_eqs, **solver_kwargs):
    assert len(list_of_eqs)>0
    Q_exp = []
    L_exp = []
    L_ieq = []
    L_eq = []
    for eq in list_of_eqs:
        if isinstance(eq, Quadratic_equation):
            Q_exp.append(eq)
        elif isinstance(eq, Linear_equation):
            if eq.is_equality:
                L_eq.append(eq)
            elif eq.is_inequality:
                L_ieq.append(eq)
            else:
                L_exp.append(eq)
        else:
            raise ValueError(f'{eq} is not an Quadratic or Linear equation')
    assert len(Q_exp)<=1, 'only one Quadratic term allowed'

    if len(Q_exp)==0 and len(L_ieq)==0 and len(L_exp)==0 and len(L_eq)>0:#Linear system of equations
        sys = System_of_linear_eqs()
        sys.add_equations(L_eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_exp)==0 and len(L_ieq)==0 and len(L_exp)>0 and len(L_eq)==0: #Least Squares
        sys = Least_squares()
        for eq in L_exp:
            sys.add_objective(eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_exp)==0 and len(L_exp)>0: #Constrainted Least Squares
        sys = Constrained_least_squares()
        for eq in L_exp:
            sys.add_objective(eq)
        for eq in L_ieq:
            sys.add_inequality(eq)
        for eq in L_eq:
            sys.add_equality(eq)
        sys.solve(**solver_kwargs)
        return sys
    elif len(Q_exp)==1 and len(L_exp)==0: #Quadratic problem
        sys = Quadratic_problem()
        sys.add_objective(Q_exp[0])
        for eq in L_ieq:
            sys.add_inequality(eq)
        for eq in L_eq:
            sys.add_equality(eq)
        sys.solve(**solver_kwargs)
        return sys
    else:
        raise ValueError(f'Cannot find solver for len(Q_exp)={len(Q_exp)}, len(L_ieq)={len(L_ieq)}, len(L_exp)={len(L_exp)}, len(L_eq)=={len(L_eq)}')

if __name__=='__main__':
    x = Variable('x')

    eqs = [x[0] + x[1] + 2*x[2] == 1,
            3*x[0] + x[1] + 0.5*x[2] == 2,
            x[0] + 2*x[1] + 0 == 3]
    sys = quick_solve(eqs)
    print('Least system of equations:')
    print(sys, sys[x[0], x[1], x[2]], sys[eqs])

    eqs = [x[0] + x[1] + 2*x[2]+2,
            3*x[0] + x[1] + 0.5*x[2]+2,
            x[0] + 2*x[1] + 0+4, 
            x[1] + x[2]+4]
    sys = quick_solve(eqs)
    print('Least Squares:')
    print(sys, sys[x[0], x[1], x[2]], sys[eqs])
    

    eqs = [x[0] + x[1] + 2*x[2]+2,
            3*x[0] + x[1] + 0.5*x[2]+2,
            x[0] + 2*x[1] + 0+4, 
            x[1] + x[2]+9==2]
    sys = quick_solve(eqs, toarray=True) #I'm getting a Segmentation fault if using solver='osqp' and toarray=False
    print('Constrained Least Squares:')
    print(sys, sys[x[0], x[1], x[2]], sys[eqs])

    eqs = [x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[1]+x[2]]
    sys = quick_solve(eqs)
    print('Quadratic problem:')
    print(sys, sys[x[0], x[1], x[2]])


