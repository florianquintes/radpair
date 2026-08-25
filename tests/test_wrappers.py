"""Tests for :mod:`radpair._wrappers`."""

from types import SimpleNamespace

import numpy as np

from radpair._wrappers import function_benchmark, multicore, timer

# ---------------------------------------------------------------------------
# timer
# ---------------------------------------------------------------------------


class TestTimer:
    """Tests for the :func:`timer` decorator."""

    def test_returns_original_result(self, capsys):
        @timer
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_prints_runtime(self, capsys):
        @timer
        def dummy():
            return 42

        dummy()
        captured = capsys.readouterr()
        assert "runtime" in captured.out.lower()
        assert "dummy" in captured.out

    def test_preserves_kwargs(self, capsys):
        @timer
        def greet(name, greeting="hello"):
            return f"{greeting}, {name}"

        result = greet("world", greeting="hi")
        assert result == "hi, world"


# ---------------------------------------------------------------------------
# function_benchmark
# ---------------------------------------------------------------------------


class TestFunctionBenchmark:
    """Tests for the :func:`function_benchmark` decorator.

    Note: ``function_benchmark`` takes ``func`` as its first positional
    argument (it is not a decorator factory with parenthesised kwargs).
    """

    def test_does_not_return_original_result(self, capsys):
        def square(x):
            return x * x

        benchmarked = function_benchmark(square, niter=5)
        result = benchmarked(3)
        assert result is None

    def test_prints_statistics(self, capsys):
        def dummy(x):
            return x

        benchmarked = function_benchmark(dummy, niter=5)
        benchmarked(42)
        captured = capsys.readouterr()
        assert "benchmark" in captured.out.lower()
        assert "average" in captured.out.lower()
        assert "best" in captured.out.lower()
        assert "worst" in captured.out.lower()

    def test_runs_niter_times(self, capsys):
        call_count = [0]

        def counted(x):
            call_count[0] += 1
            return x

        benchmarked = function_benchmark(counted, niter=10)
        benchmarked(0)
        assert call_count[0] == 10


# ---------------------------------------------------------------------------
# multicore
# ---------------------------------------------------------------------------


def _identity_simulation(sys, exp, simopt):
    """Trivial simulation that returns the field axis — no eprbase needed."""
    return 1 * exp.B_z


class TestMulticore:
    """Tests for the :func:`multicore` decorator."""

    @staticmethod
    def _make_inputs(n_points=100, cpu_cores=2):
        sys = SimpleNamespace()
        exp = SimpleNamespace(
            B_z=np.linspace(320, 370, n_points),
            magnetic_field=np.linspace(320, 370, n_points),
        )
        simopt = SimpleNamespace(cpu_cores=cpu_cores)
        return sys, exp, simopt

    def test_output_matches_single_core(self):
        """Concatenated slices reconstruct the original field axis."""
        sys, exp, simopt = self._make_inputs(n_points=100, cpu_cores=2)
        wrapped = multicore(_identity_simulation)
        result = wrapped(sys, exp, simopt)
        np.testing.assert_array_equal(result, exp.B_z)

    def test_cpu_cores_zero_autodetects(self):
        """cpu_cores=0 resolves to cpu_count()."""
        from multiprocessing import cpu_count

        sys, exp, simopt = self._make_inputs(n_points=100, cpu_cores=0)
        wrapped = multicore(_identity_simulation)
        result = wrapped(sys, exp, simopt)
        assert simopt.cpu_cores == cpu_count()
        np.testing.assert_array_equal(result, exp.B_z)

    def test_single_core(self):
        """cpu_cores=1 is a degenerate but valid case."""
        sys, exp, simopt = self._make_inputs(n_points=50, cpu_cores=1)
        wrapped = multicore(_identity_simulation)
        result = wrapped(sys, exp, simopt)
        np.testing.assert_array_equal(result, exp.B_z)

    def test_uneven_split(self):
        """Field points not evenly divisible across cores still work."""
        sys, exp, simopt = self._make_inputs(n_points=101, cpu_cores=3)
        wrapped = multicore(_identity_simulation)
        result = wrapped(sys, exp, simopt)
        assert result.shape == (101,)
        np.testing.assert_array_equal(result, exp.B_z)
