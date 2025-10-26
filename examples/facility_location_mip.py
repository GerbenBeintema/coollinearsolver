from cool_linear_solver import Variable, Integer, quick_solve


def run_example(verbose=1):
    """Small facility location example (open facilities with fixed cost, assign customers)

    Decision variables:
      - y[j] in {0,1}: open facility j
      - x[i,j] in {0,1}: assign customer i to facility j

    Objective: minimize sum_j open_cost[j]*y[j] + sum_{i,j} assign_cost[i][j]*x[i,j]
    Constraints:
      - each customer assigned to exactly one facility
      - x[i,j] <= y[j] (can only assign to open facilities)
      - binary bounds for x and y
    """
    num_facilities = 3
    num_customers = 5

    # fixed opening costs for facilities
    open_cost = [100, 120, 80]

    # assignment costs (customer x facility)
    assign_cost = [
        [20, 24, 11],
        [28, 27, 82],
        [74, 97, 71],
        [2, 55, 73],
        [46, 96, 59],
    ]

    y = Variable('y')  # facility open decision
    x = Variable('x')  # assignment decision x[i,j]

    eqs = []

    # objective: open costs + assignment costs
    obj_terms = []
    for j in range(num_facilities):
        obj_terms.append(open_cost[j] * y[j])
    for i in range(num_customers):
        for j in range(num_facilities):
            obj_terms.append(assign_cost[i][j] * x[i, j])
    eqs.append(sum(obj_terms))

    # each customer must be assigned to exactly one facility
    for i in range(num_customers):
        eqs.append(sum(x[i, j] for j in range(num_facilities)) == 1)

    # assignment implies facility open; and enforce binary domain via 0<=var<=1 and Integer marker
    for j in range(num_facilities):
        eqs.append(y[j] >= 0)
        eqs.append(y[j] <= 1)
        eqs.append(y[j] == Integer)
    for i in range(num_customers):
        for j in range(num_facilities):
            eqs.append(x[i, j] >= 0)
            eqs.append(x[i, j] <= 1)
            eqs.append(x[i, j] == Integer)
            eqs.append(x[i, j] <= y[j])

    sol = quick_solve(eqs, verbose=0)

    if verbose:
        print('\n=== Facility Location (small MILP) ===')
        print('Opened facilities:')
        for j in range(num_facilities):
            print(f'  facility {j}:', int(round(sol[y[j]])))
        print('\nAssignments (customer -> facility):')
        for i in range(num_customers):
            for j in range(num_facilities):
                if round(sol[x[i, j]]) == 1:
                    print(f'  customer {i} -> facility {j} (cost={assign_cost[i][j]})')
    
    return sol


if __name__ == '__main__':
    run_example()
