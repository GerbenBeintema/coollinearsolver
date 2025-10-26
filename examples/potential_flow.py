from cool_linear_solver import Variable, quick_solve


def run_example(verbose=1):
    """Solve a 2D streamfunction Poisson problem to illustrate potential flow.

    We solve Laplacian(psi) = -Gamma * delta(x-x0, y-y0) on a rectangular grid
    with Dirichlet boundary conditions that impose a uniform flow (psi = U*y)
    on the boundaries. The source term models a single point vortex at the
    domain center. The solution's gradient gives the velocity field.
    """
    import numpy as np

    Nx = Ny = 80
    xar = np.linspace(0, 1, Nx)
    yar = np.linspace(0, 1, Ny)
    dx = xar[1] - xar[0]
    dy = yar[1] - yar[0]

    Psi = Variable(name='psi')
    eqs = []

    # Uniform flow speed and vortex strength
    U = 1.0
    Gamma = 5.0

    # RHS source: approximate delta at center cell
    I = np.zeros((Ny, Nx), dtype=float)
    cy, cx = Ny // 2, Nx // 2
    I[cy, cx] = -Gamma / (dx * dy)  # RHS = -Gamma * delta approximated

    for yi in range(Ny):
        for xi in range(Nx):
            x, y = xar[xi], yar[yi]
            # Dirichlet boundaries: impose psi = U*y to represent uniform flow
            if xi == 0 or xi == Nx - 1 or yi == 0 or yi == Ny - 1:
                eqs.append(Psi[x, y] == U * y)
            else:
                neigh = [Psi[xar[xi + 1], y], Psi[xar[xi - 1], y], Psi[x, yar[yi + 1]], Psi[x, yar[yi - 1]]]
                # discrete Laplacian: (sum neigh - 4 psi)/dx^2 ~ laplacian
                # rearrange to 4*psi - sum(neigh) = -I * dx^2 (we scale RHS accordingly)
                eqs.append(4 * Psi[x, y] - sum(neigh) == I[yi, xi] * (dx * dx))

    sol = quick_solve(eqs)

    # collect solution into array
    Psimat = np.zeros((Ny, Nx), dtype=float)
    for yi in range(Ny):
        for xi in range(Nx):
            Psimat[yi, xi] = sol[Psi[xar[xi], yar[yi]]]

    if verbose:
        M = sol.get_sparse_matrix()
        print('matrix shape:', M.shape, 'nnz:', M.nnz)

        try:
            from matplotlib import pyplot as plt

            # compute velocity components: u = dpsi/dy, v = -dpsi/dx
            uy, ux = np.gradient(Psimat, yar, xar)
            u = uy  # dpsi/dy
            v = -ux  # -dpsi/dx

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes[0]
            im = ax.imshow(Psimat, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu')
            ax.set_title('Streamfunction (psi)')
            fig.colorbar(im, ax=ax)

            ax = axes[1]
            skip = max(Nx // 20, 1)
            X, Y = np.meshgrid(xar, yar)
            ax.streamplot(X, Y, u, v, density=1.2)
            ax.set_title('Velocity streamlines')

            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    return sol


if __name__ == '__main__':
    run_example()
