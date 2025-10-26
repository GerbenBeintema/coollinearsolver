from cool_linear_solver import Variable, quick_solve


def run_example(verbose=1):
    """Solve a 2D resistor grid (Kirchhoff current law).

    Left boundary is held at V=1, right boundary at V=0. A unit current is
    injected at the center node. The code builds linear equations for each
    grid node and solves for node voltages.
    """
    import numpy as np

    Nx = Ny = 400
    xar = np.linspace(0, 1, Nx)
    yar = np.linspace(0, 1, Ny)

    V = Variable(name='V')
    eqs = []

    # injection: unit current at center
    I = np.zeros((Ny, Nx), dtype=float)
    I[Ny // 2, Nx // 2] = 1.0

    for yi in range(Ny):
        for xi in range(Nx):
            x, y = xar[xi], yar[yi]
            # Dirichlet boundaries: left=1, right=0, top/bottom insulated (Neumann ~ handled by copying neighbor)
            if xi == 0:
                eqs.append(V[x, y] == 1.0)
            elif xi == Nx - 1:
                eqs.append(V[x, y] == 0.0)
            elif yi == 0 or yi == Ny - 1:
                # treat top/bottom as fixed Neumann by replacing missing neighbor with same node
                # which results in fewer neighbors; implement discrete Laplacian with available neighbors
                neigh = []
                if xi + 1 < Nx:
                    neigh.append(V[xar[xi + 1], y])
                if xi - 1 >= 0:
                    neigh.append(V[xar[xi - 1], y])
                if yi + 1 < Ny:
                    neigh.append(V[x, yar[yi + 1]])
                if yi - 1 >= 0:
                    neigh.append(V[x, yar[yi - 1]])
                nnb = len(neigh)
                # nnb*V_i - sum(neigh) == I
                eqs.append(nnb * V[x, y] - sum(neigh) == I[yi, xi])
            else:
                # interior node: 4*V_i - sum(neighbors) = I
                neigh = [V[xar[xi + 1], y], V[xar[xi - 1], y], V[x, yar[yi + 1]], V[x, yar[yi - 1]]]
                eqs.append(4 * V[x, y] - sum(neigh) == I[yi, xi])

    sol = quick_solve(eqs)
    if verbose:
        M = sol.get_sparse_matrix()
        print('matrix shape:', M.shape, 'nnz:', M.nnz)

    # collect solution into array
    Vmat = np.zeros((Ny, Nx), dtype=float)
    for yi in range(Ny):
        for xi in range(Nx):
            Vmat[yi, xi] = sol[V[xar[xi], yar[yi]]]

    if verbose:
        try:
            from matplotlib import pyplot as plt
            plt.imshow(Vmat, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
            plt.colorbar(label='Voltage')
            plt.title('Resistor grid voltages')
            plt.show()
        except Exception:
            # plotting is optional in headless/test environments
            pass

    return sol


if __name__ == '__main__':
    run_example()
