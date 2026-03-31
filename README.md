# JF Tools

Collection of small useful helper tools for Python by Johannes Feist.

## Short Iterative Lanczos Backend Selection

The `jftools.short_iterative_lanczos` solver now supports backend auto-dispatch for static Hamiltonians:

- `cython` when the Cython extension is built and importable.
- `python` for callable Hamiltonians and unsupported formats.

Select backend behavior with the `backend` function argument:

- `auto` (default): choose the best available backend.
- `python`: force the original Python implementation.
- `cython`: force Cython backend (requires compiled extension).

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

After building, `auto` prefers `cython`.

## QuTiP Compatibility

Short iterative Lanczos targets modern QuTiP (5.x):

- QuTiP Hamiltonians are converted through `H.data.as_scipy()` when available.
- QuTiP state outputs preserve `dims` and state shape.

## Publishing

Releases are intended to be published through GitHub Actions rather than from a local machine.

Before publishing a new version:

1. Update the version in [meson.build](meson.build) (the single source of truth; [pyproject.toml](pyproject.toml) picks it up automatically).
2. Commit and push the version bump.
3. Optionally run the `Test release (TestPyPI)` workflow to verify the full wheel matrix before publishing.
4. Create a GitHub Release or run the `Publish` workflow manually.

The publish workflow builds:

- wheels for CPython 3.12, 3.13, and 3.14,
- Linux `x86_64` and `aarch64`,
- macOS `x86_64` and `arm64`,
- Windows `AMD64`,
- plus an sdist.

For a dry run against TestPyPI, use the `Test release (TestPyPI)` workflow.

For a local packaging sanity check on your current machine only:

```bash
uv build
```
