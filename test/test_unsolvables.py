from cool_linear_solver import Variable, quick_solve
import numpy as np

def test_unsolvables():
    """Small checks for under/over/singular cases.

    Each case uses `force_solution=True` and asserts reasonable behavior:
    - underdetermined -> a solution satisfying the equations (minimum-norm)
    - redundant equations -> same as underdetermined
    - consistent overdetermined -> exact solution (if exists)
    - inconsistent equations -> least-squares compromise (checks residual)
    """

    tol = 1e-8

    x = Variable('x')

    # 1) underdetermined (not enough equations)
    eqs_not_enough = [x[0] + x[1] == 1]
    sol_not_enough = quick_solve(eqs_not_enough, force_solution=True)
    res = sol_not_enough[x[0]] + sol_not_enough[x[1]] - 1.0
    assert abs(res) < tol, f'Underdetermined residual too large: {res}'

    # 2) redundant (infinite solutions) - same equation twice
    eqs_infinite = [x[0] + x[1] == 1, x[0] + x[1] == 1]
    sol_infinite = quick_solve(eqs_infinite, force_solution=True)
    res = sol_infinite[x[0]] + sol_infinite[x[1]] - 1.0
    assert abs(res) < tol, f'Redundant residual too large: {res}'

    # 3) duplicate equation with an additional independent equation (consistent)
    eqs_dup_eq = [x[0] + x[1] == 1, x[0] + x[1] == 1, x[0] - x[1] == 2]
    sol_dup_eq = quick_solve(eqs_dup_eq, force_solution=True)
    # expected solution: x0=1.5, x1=-0.5
    assert abs(sol_dup_eq[x[0]] - 1.5) < 1e-8 and abs(sol_dup_eq[x[1]] + 0.5) < 1e-8, f'Unexpected dup_eq solution: {sol_dup_eq[x[0]]}, {sol_dup_eq[x[1]]}'

    # 4) inconsistent duplicate RHS -> least-squares compromise
    eqs_no_solution = [x[0] + x[1] == 1, x[0] + x[1] == 2]
    sol_no_solution = quick_solve(eqs_no_solution, force_solution=True)
    sum_val = sol_no_solution[x[0]] + sol_no_solution[x[1]]
    # best LS compromise for two equal-weight conflicting RHS is 1.5
    assert abs(sum_val - 1.5) < 1e-8, f'Inconsistent case did not produce expected compromise sum: {sum_val}'

    # 5) overdetermined but consistent
    eqs_over = [x[0] + x[1] == 3, 2*x[0] + 2*x[1] == 6, x[0] - x[1] == 1]
    sol_over = quick_solve(eqs_over, force_solution=True)
    assert abs(sol_over[x[0]] - 2.0) < 1e-8 and abs(sol_over[x[1]] - 1.0) < 1e-8, f'Overdetermined solution incorrect: {sol_over[x[0]]}, {sol_over[x[1]]}'

    print('unsolvables checks passed')

if __name__ == '__main__':
    test_unsolvables()