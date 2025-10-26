# cool_linear_solver
 
An easy going (contrained) linear solver with sparse methods and minimal effort

## Installation

```bash
pip install cool-linear-solver
```

## 1) System of linear equations

```python
from cool_linear_solver import Variable, quick_solve

x = Variable('x')
eqs = [x[0] + x[1] == 2, x[1] - x[0] == 3] #x[0] creates a new variable automatically you can also use thing like x['a'] 
sol = quick_solve(eqs)
print('x0, x1 =', sol[x[0]], sol[x[1]])
```

## 2) Solving the heat equation 

This is a compact demonstration of using indexed `Variable` objects to build a small finite-difference-like system. This module uses sparse matrixes so the memory size increases only linearlly with the number of non-zero element in the system of equation.

```python
import numpy as np
from cool_linear_solver import Variable, quick_solve

Nx = 50
Ny = 50
T = Variable('T')

x_coords = np.linspace(0, 1, Nx)
y_coords = np.linspace(0, 1, Ny)
dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]

eqs = []
for x in x_coords:
    for y in y_coords:
        # Simple Dirichlet boundary: set boundaries to 1 on x==0 and 0 elsewere on boundary
        # Interior to Laplace discretization
        if x == 0:
            eqs.append(T[x, y] == 1)
        elif y == 0 or y == 1 or x==1:
            eqs.append(T[x, y] == 0)
        else:
            eqs.append(T[x-dx, y] + T[x+dx, y] + T[x, y-dy] + T[x, y+dy] - 4*T[x, y] == 0) #floats can be used

sol = quick_solve(eqs)
# collect solution into array
Tarr = np.array([[sol[T[x, y]] for x in x_coords] for y in y_coords])

# 3d surface plot of result
from matplotlib import pyplot as plt
X, Y = np.meshgrid(x_coords, y_coords)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Tarr, cmap='viridis', edgecolor='none')
ax.view_init(elev=30, azim=-60)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('T')
plt.colorbar(surf, ax=ax, shrink=0.6)
plt.tight_layout(); plt.show()
```

## 3) Least squares (unconstrained)

Create a least-squares objective by passing squared linear expressions.

```python
from cool_linear_solver import Variable, quick_solve

x = Variable('x')
eqs = [ (x[0] + x[1] - 3)**2, (2*x[0] - x[1] + 1)**2 ]
sol = quick_solve(eqs)
print('solution:', sol[x[0]], sol[x[1]])
```

## 4) Constrained least squares

Add linear equalities / inequalities together with squared objectives.

```python
from cool_linear_solver import Variable, quick_solve

x = Variable('x')
eqs = [ (x[0] + x[1] - 3)**2, (2*x[0] - x[1] + 1)**2, x[0] + x[1] == 2, x[0] >= 0 ]
sol = quick_solve(eqs)
print('solution:', sol[x[0]], sol[x[1]])
```

## 5) Quadratic problem

Minimize a quadratic objective (here: sum of squares plus linear term) with a linear equality.

```python
from cool_linear_solver import Variable, quick_solve

x = Variable('x')
eqs = [ x[0]*x[0] + x[1]*x[1] + x[0] + x[1], x[0] + x[1] == 1 ]
sol = quick_solve(eqs)
print('solution:', sol[x[0]], sol[x[1]])
```

## 6) Linear program (LP)

Small LP: minimize a linear cost with linear constraints.

```python
from cool_linear_solver import Variable, quick_solve

x = Variable('x')
eqs = [ x[0] + 2*x[1], x[0] + x[1] == 3, x[0] >= 0, x[1] >= 0 ]
sol = quick_solve(eqs)
print('x0, x1 =', sol[x[0]], sol[x[1]])
```

## 7) Mixed-integer program (MIP)

Mark variables as integer or binary using the `Integer` and `Binary` markers. `quick_solve` will route the problem to the MILP backend.

```python
from cool_linear_solver import Variable, quick_solve, Integer, Binary

x = Variable('x')
eqs = [ 2*x[0] - 3*x[1] + x[2], x[0] == Integer, x[1] == Binary, x[0] >= 0, x[1] >= 0, x[2] >= 0, x[0] <= 5, x[1] <= 1 ]
sol = quick_solve(eqs)
print('x0, x1, x2 =', sol[x[0]], sol[x[1]], sol[x[2]])
```
