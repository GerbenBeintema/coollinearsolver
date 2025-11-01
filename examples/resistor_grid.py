"""Compute effective resistance of an N x N grid of 1-ohm resistors.

Constructs node-voltage unknowns V[i,j] for a square grid. All interior
nodes satisfy Kirchhoff's current law (sum of currents to neighbours = 0).
Two adjacent central nodes are fixed to 1V and 0V respectively. Currents on
each edge are computed as voltage differences (R=1), and the effective
resistance between the two central nodes is R = (V_diff) / I_total.

Usage: run the script directly. It accepts an optional --N argument.
"""
from cool_linear_solver import Variable, quick_solve
import argparse
import numpy as np
import matplotlib.pyplot as plt

def build_and_solve(N):
    V = Variable('V')
    eqs = []

    def neighbors(i, j):
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < N and 0 <= nj < N:
                yield (ni, nj)

    # choose two adjacent center nodes: (mid, mid-1) and (mid, mid)
    mid = N // 2
    plus = (mid-1, mid-2)
    minus = (mid, mid)

    for i in range(N):
        for j in range(N):
            if (i,j) == plus or (i,j) == minus:
                # fixed potentials; handled below
                continue
            if i==0: # BC von Neumann type
                eqs.append(V[i,j] == V[i+1,j])
                # eqs.append(V[i,j] == 0.5)
            elif i==N-1:
                eqs.append(V[i,j] == V[i-1,j])
                # eqs.append(V[i,j] == 0.5)
            elif j==0:
                eqs.append(V[i,j] == V[i,j+1])
                # eqs.append(V[i,j] == 0.5)
            elif j==N-1:
                eqs.append(V[i,j] == V[i,j-1])
                # eqs.append(V[i,j] == 0.5)
            else: # build KCL for every node
                eqs.append(4*V[i,j] == V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1])


    # fix potentials on the two central adjacent nodes
    eqs.append(V[plus] == 1.0)
    eqs.append(V[minus] == 0.0)

    # don't forward example verbosity into solver internals (some solvers
    # don't accept a `verbose` kwarg on their `solve()` method)
    sys = quick_solve(eqs)

    # extract node voltages into a 2D list for convenience
    volt = [[sys[V[i,j]] for j in range(N)] for i in range(N)]

    # compute currents out of the + node (sum of V_plus - V_neighbor)
    I_total = 0.0
    for ni, nj in neighbors(*plus):
        I_total += (volt[plus[0]][plus[1]] - volt[ni][nj])

    # voltage difference is 1.0 (by construction)
    R_eff = 1.0 / I_total if I_total != 0 else float('inf')

    return volt, I_total, R_eff


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=7, help='grid size N (N x N)')
    args = p.parse_args()

    volt, I_total, R_eff = build_and_solve(args.N)
    print(f'Grid {args.N} x {args.N}: total current from + node = {I_total:.8f} A')
    print(f'Effective resistance between the two center nodes: R = {R_eff:.8f} ohm')

    # optional: print central area voltages
    mid = args.N // 2
    print('\nCentral voltages (3x3 window):')
    for i in range(mid-1, mid+2):
        row = []
        for j in range(mid-1, mid+2):
            if 0 <= i < args.N and 0 <= j < args.N:
                row.append(f'{volt[i][j]:.6f}')
            else:
                row.append('   -   ')
        print('  '.join(row))
    
    # 3d plot of the voltages:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(0, args.N, 1)
    Y = np.arange(0, args.N, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.array(volt)
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X index')
    ax.set_ylabel('Y index')
    ax.set_zlabel('Voltage (V)')
    plt.title('Voltage Distribution in Resistor Grid')
    plt.show()

def acc_with_n(n_max=100):
    plt.figure()

    ref_sol = 4/np.pi - 0.5
    N_list = np.geomspace(7, n_max, num=30).astype(int)
    R_eff_list = []
    from tqdm import tqdm
    import time
    for N in tqdm(N_list):
        start_time = time.time()
        volt, I_total, R_eff = build_and_solve(N)
        R_eff_list.append(R_eff)
        print(f'Grid {N}x{N}: R_eff = {R_eff:.8f} ohm, R error = {R_eff - ref_sol:.8f} ohm, time taken: {time.time() - start_time:.2f} s')
    
    plt.loglog(N_list, abs(np.array(R_eff_list) - ref_sol), marker='o')
    plt.xlabel('Grid size N')
    plt.ylabel('Effective Resistance R_eff - Reference Solution (ohm)')
    plt.title('Effective Resistance of N x N Resistor Grid')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()
    acc_with_n(n_max=100)
