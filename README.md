# JF Tools

Collection of small useful helper tools for Python by Johannes Feist.

## Short Iterative Lanczos Backend Selection

The `jftools.short_iterative_lanczos` solver now supports backend auto-dispatch for static Hamiltonians:

- `cython` when the Cython extension is built and importable.
- `reference` for callable Hamiltonians and unsupported formats.

Select backend behavior with the `backend` function argument:

- `auto` (default): choose the best available backend for the Hamiltonian type.
- `python`: force the original Python implementation.
- `cython`: force Cython backend (requires compiled extension).

If an explicit backend does not match the Hamiltonian type (or is unavailable), the solver raises an error.

Example:

```python
prop = jftools.short_iterative_lanczos.lanczos_timeprop(H, maxsteps=14, target_convg=1e-12, backend="cython")
```

### Build Cython Extension

The Cython implementation is in [jftools/short_iterative_lanczos_cython.pyx](jftools/short_iterative_lanczos_cython.pyx).

It is built automatically during normal installation via `meson-python`.

Recommended (uv-only) workflow:

```bash
uv sync --group dev
uv run python -m pip install -e . --no-build-isolation
```

For a non-editable local install through uv:

```bash
uv run python -m pip install .
```

After building, `auto` prefers `cython` for static dense and CSR Hamiltonians.

## QuTiP Compatibility

Short iterative Lanczos targets modern QuTiP (5.x):

- QuTiP Hamiltonians are converted through `H.data.as_scipy()` when available.
- When the Cython extension is built, the QuTiP 5 sparse data path uses direct Cython cimports of QuTiP's data-layer CSR matvec implementation.
- QuTiP state outputs preserve `dims` and state shape.
- Tests skip QuTiP-specific checks if QuTiP is not installed.

## Benchmark

Run the standalone benchmark script:

```bash
uv run python dev/bench_short_iterative_lanczos.py
```

Example with explicit size sweep and CSV export:

```bash
uv run python dev/bench_short_iterative_lanczos.py --sizes 500 1000 2000 5000 --repeats 3 --csv dev/bench_short_iterative_lanczos.csv
```

The benchmark reports, for each case:

- selected auto backend,
- warmup timing (first call, includes JIT compile cost),
- steady-state timing (average repeated calls),
- steady-state speedup and relative error versus the reference backend.

For dense Cython cases, you can also request a per-phase timing breakdown:

```bash
uv run python dev/bench_short_iterative_lanczos.py --sizes 500 --profile-dense
```

This adds one profiled auto-backend pass for dense cases and prints the time spent in:

- dense matvec,
- orthogonalization and basis updates,
- coefficient and eigensolve work,
- state reconstruction,
- total profiled runtime.

Useful options:

- `--sizes`: one or more system sizes.
- `--repeats`: number of steady-state repetitions.
- `--n-times`: number of time points in `ts`.
- `--t-final`: final propagation time.
- `--csv`: path for CSV output.
- `--profile-dense`: print a dense-case phase breakdown for the Cython backend.
