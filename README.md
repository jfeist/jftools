# JF Tools

Collection of small useful helper tools for Python by Johannes Feist.

## Short Iterative Lanczos Backend Selection

The `jftools.short_iterative_lanczos` solver now assumes the Numba backend is installed as part of the package dependencies.

- `auto` (default) selects `numba`.
- `numba` forces the Numba implementation.
- `python` forces the original reference implementation.

Select backend behavior with the `backend` function argument:

- `auto` (default): select the Numba backend.
- `python`: force the original Python implementation.
- `numba`: force the Numba backend.

Example:

```python
prop = jftools.short_iterative_lanczos.lanczos_timeprop(H, maxsteps=14, target_convg=1e-12, backend="numba")
```

## Installation

`jftools` is a pure Python package again. There is no compiled extension build step.

Recommended (uv-only) workflow:

```bash
uv sync --group dev
uv run python -m pip install -e .
```

For a non-editable local install through uv:

```bash
uv run python -m pip install .
```

## QuTiP Compatibility

Short iterative Lanczos targets modern QuTiP (5.x):

- QuTiP Hamiltonians are converted through `H.data.as_scipy()` when available.
- QuTiP state outputs preserve `dims` and state shape.

## Publishing

Before publishing a new version:

1. Update `__version__` in [jftools/__init__.py](jftools/__init__.py).
2. Commit the version bump.
3. Build locally.

For a local packaging sanity check on your current machine only:

```bash
uv build
```
