"""Traveling Salesman example (MILP)

Small TSP modeled as MILP with binary arc variables x[i,j] and MTZ
(Miller-Tucker-Zemlin) subtour elimination using continuous u variables.

Usage: call run_example(n_nodes=8, seed=0, verbose=1)
"""
from math import hypot
import numpy as np
from cool_linear_solver import Variable, Binary, quick_solve


def run_example(n_nodes=8, seed=0, verbose=1):
    if n_nodes < 2:
        raise ValueError('n_nodes must be >= 2')

    rng = np.random.RandomState(seed)
    coords = rng.rand(n_nodes, 2) * 100.0

    # Euclidean distance matrix
    dist = [[0.0]*n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                dist[i][j] = 0.0
            else:
                dist[i][j] = hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])

    # Decision variables
    x = Variable('x')  # x[i,j] binary: arc i->j
    u = Variable('u')  # MTZ continuous order variables

    eqs = []

    # Objective: minimize sum_{i!=j} dist[i][j] * x[i,j]
    obj_terms = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            obj_terms.append(dist[i][j] * x[i, j])
    eqs.append(sum(obj_terms))

    # Degree constraints: each node has exactly one outgoing and one incoming arc
    for i in range(n_nodes):
        # sum_j x[i,j] == 1
        eqs.append(sum(x[i, j] for j in range(n_nodes) if j != i) == 1)
    for j in range(n_nodes):
        # sum_i x[i,j] == 1
        eqs.append(sum(x[i, j] for i in range(n_nodes) if i != j) == 1)

    # Binary domain for arcs
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            eqs.append(x[i, j] == Binary)

    # MTZ subtour elimination (use nodes 1..n-1 for u variables)
    # Fix u[0] = 0 and 1 <= u[i] <= n_nodes-1 for i>=1
    eqs.append(u[0] == 0)
    for i in range(1, n_nodes):
        eqs.append(u[i] >= 1)
        eqs.append(u[i] <= n_nodes - 1)

    # For all i != j and i,j >=1: u[i] - u[j] + n_nodes * x[i,j] <= n_nodes - 1
    for i in range(1, n_nodes):
        for j in range(1, n_nodes):
            if i == j:
                continue
            eqs.append(u[i] - u[j] + n_nodes * x[i, j] <= n_nodes - 1)

    # Solve via quick_solve (will route to the mixed-integer solver because of Binary markers)
    sol = quick_solve(eqs, verbose=0)

    # Reconstruct tour from x values
    # Build adjacency from arcs with value ~1
    adj = {i: None for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            val = sol[x[i, j]]
            if round(val) == 1:
                adj[i] = j

    # Follow tour starting at 0
    tour = [0]
    cur = 0
    visited = {0}
    while True:
        nxt = adj[cur]
        if nxt is None:
            break
        if nxt in visited:
            break
        tour.append(nxt)
        visited.add(nxt)
        cur = nxt
        if cur == 0:
            break

    total_cost = sum(dist[i][adj[i]] for i in range(n_nodes))

    if verbose:
        print('\n=== Traveling salesman (MILP) ===')
        print('n_nodes:', n_nodes)
        print('tour:', tour)
        print(f'total cost: {total_cost:.3f}')

    # Visualization when requested
    if verbose==2:
        try:
            import matplotlib.pyplot as plt
            # coordinates
            xs = coords[:, 0]
            ys = coords[:, 1]
            # ensure tour is closed for plotting
            tour_idx = list(tour)
            if len(tour_idx) > 0 and tour_idx[0] != tour_idx[-1]:
                tour_idx = tour_idx + [tour_idx[0]]
            tx = [coords[i, 0] for i in tour_idx]
            ty = [coords[i, 1] for i in tour_idx]

            plt.figure(figsize=(6, 6))
            plt.plot(xs, ys, 'ko', label='nodes')
            plt.plot(tx, ty, '-r', linewidth=1.5, label='tour')
            for i, (xcoord, ycoord) in enumerate(coords):
                plt.text(xcoord + 0.8, ycoord + 0.8, str(i), fontsize=9)
            plt.title(f'TSP tour n={n_nodes} cost={total_cost:.2f}')
            plt.axis('equal')
            plt.legend()
            try:
                plt.show()
            except Exception:
                # headless environment: save a PNG instead
                fname = f'tsp_n{n_nodes}_seed{seed}.png'
                plt.savefig(fname, bbox_inches='tight')
                print(f'Plot saved to {fname}')
        except Exception as e:
            print('Plot skipped (matplotlib not available or failed):', e)
    return sol, tour, total_cost

def speed_test():
    # run run_example with increasing number of nodes and time it
    import time
    n_now = 2
    while True:
        start_time = time.time()
        run_example(n_nodes=n_now, seed=1, verbose=0)
        elapsed = time.time() - start_time
        print(f'n_nodes={n_now} solved in {elapsed:.2f} seconds')
        n_now += 2  # increase the size for next test
        if elapsed > 60:  # stop if it takes more than 60 seconds
            break


if __name__ == '__main__':
    run_example(8, seed=0)
