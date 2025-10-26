### Quick orientation for AI coding agents

This repo implements a tiny, dependency-light library for constructing and solving
linear, least-squares, constrained least-squares and simple quadratic problems
using sparse matrices. Keep changes small and preserve the existing public API
surface in `cool_linear_solver/__init__.py`.

Key files to read and reference:
- `cool_linear_solver/eqs_and_vars.py` — Variable and expression builder. Variables
  generate internal hashed ids and support non-integer indexing (e.g. `T[3.5]`).
- `cool_linear_solver/linear_solver.py` — sparse linear system builder and solver
  using `scipy.sparse` and `scipy.sparse.linalg.spsolve`.
- `cool_linear_solver/least_squares.py` — implements unconstrained and
  constrained least-squares. The constrained solver relies on `qpsolvers.solve_ls`
  (optional `osqp` extra in pyproject).
- `cool_linear_solver/quadratic_problems.py` — quadratic objective -> `qpsolvers.solve_qp`.
- `cool_linear_solver/quicksolve.py` — router that inspects expression types and
  dispatches to the appropriate solver. Many example usages live in `examples/`.

Important patterns and gotchas (use these as rules when editing or adding code):
- Expression types: code distinguishes Linear_equation vs Quadratic_equation vs Variable.
  `quick_solve` inspects `eq.is_equality`/`is_inequality` and counts quadratic terms.
- Variable indexing: `Variable[name][index]` creates new hashed ids stored in
  a global `global_back_hash` — do not change the hashing scheme without testing
  `examples/system_of_eqs_*` and `examples/con_least_squares_small.py`.
- Sparse shapes: `System_of_linear_eqs` builds COO-like lists then constructs
  a CSC matrix with shape `(n_eqs, n_vars)`; solvers expect matching shapes —
  mismatched `neqs` vs `nvars` raises errors in `solve()`.
- qpsolvers/osqp: constrained/nonlinear routines call `qpsolvers` and may segfault
  with certain sparse/dtype combinations; `quicksolve` sometimes uses `toarray=True`
  in examples to avoid runtime issues — preserve optional `toarray` flag and
  prefer passing `toarray=True` in PR tests when touching qpsolvers calls.

Developer workflows and commands
- Install dev dependencies: the project uses `poetry` in `pyproject.toml`.
  Minimal runtime deps: `numpy`, `scipy`, `qpsolvers` (+`osqp` extra), `matplotlib`.
- Run examples manually (recommended to reproduce behavior): open a Python REPL
  or run e.g. `python -m examples.system_of_eqs_small` (examples are plain scripts).
- When editing solver code, run quick smoke tests by executing small examples in
  `examples/` that exercise the changed code (e.g. `system_of_eqs_small.py`,
  `con_least_squares_small.py`). Prefer small grid sizes to keep runtimes short.

Testing and validation guidance for agents
- Keep changes minimal and add a focused example under `examples/` if you add
  new public behavior. Use numpy random seeds or tiny deterministic problems.
- When changing matrix construction or solver calls, test both dense (`toarray=True`)
  and sparse paths to avoid OSQP/qpsolvers segfaults.

Style and conventions
- API surface is thin and procedural. Preserve function names like `quick_solve`,
  `Variable`, `Least_squares`, `Constrained_least_squares`, `System_of_linear_eqs`.
- Prefer small, self-contained edits; avoid changing the global hashing or the
  `global_back_hash` format without a compelling reason and broad test updates.

If you need clarification from a human
- Ask about intended numeric stability expectations (dense vs sparse) and whether
  `toarray` should be the default for the constrained solvers.

Files/examples to run when verifying changes:
- `examples/system_of_eqs_small.py` — linear system end-to-end
- `examples/con_least_squares_small.py` — constrained least squares
- `examples/least_squares_small.py` — least squares

When you finish, post the changed files and a one-line summary of the validation
commands you ran and their output.
