import os
import importlib
import pytest

# List of example modules to test. Use module path under examples (dot notation).
# Run all examples (including heavy ones).
EXAMPLES = [
    'examples.least_squares_small',
    'examples.con_least_squares_small',
    'examples.system_of_eqs_small',
    'examples.gramian',
    'examples.nash',
    'examples.three-number-game',
    'examples.raising',
    'examples.system_of_eqs_flow',
    'examples.least_squares_large',
    'examples.system_of_eqs_large',
    'examples.facility_location_mip',
    'examples.mixed_integer_program',
]


@pytest.mark.parametrize('modname', EXAMPLES)
def test_example_runs(modname):
    """Import the example module and call run_example(verbose=0).

    The examples were refactored to expose run_example(verbose=1). We call with
    verbose=0 to avoid plots/prints. The test asserts the function runs without
    raising an exception and returns a non-None value.
    """
    mod = importlib.import_module(modname)
    assert hasattr(mod, 'run_example'), f'{modname} missing run_example'
    try:
        res = mod.run_example(verbose=0)
    except FileNotFoundError as e:
        pytest.skip(f'missing data file for example {modname}: {e}')
        return
    assert res is not None


def test_quicksolve_runs():
    """Run the internal quicksolve smoke test (_test_quicksolve) to ensure it executes."""
    from cool_linear_solver import quicksolve
    # _test_quicksolve prints by default; run with verbose=0 to keep test output quiet
    quicksolve._test_quicksolve(verbose=0)
