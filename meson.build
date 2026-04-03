project(
  'jftools',
  ['cpp', 'cython'],
  version: run_command('cat', 'jftools/_version.txt', check: true).stdout().strip(),
  meson_version: '>=1.4.0',
  default_options: ['warning_level=2', 'cython_language=cpp']
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

numpy_include = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

scipy_include = run_command(
  py,
  ['-c', 'import pathlib, scipy; print(pathlib.Path(scipy.__file__).resolve().parent)'],
  check: true
).stdout().strip()

py.extension_module(
  'short_iterative_lanczos_cython',
  'jftools/short_iterative_lanczos_cython.pyx',
  subdir: 'jftools',
  dependencies: py_dep,
  install: true,
  cpp_args: [
    '-I' + numpy_include,
  ],
  cython_args: [
    '-3',
    '-I' + scipy_include,
  ],
)

py.install_sources(
  [
    'jftools/__init__.py',
    'jftools/_version.txt',
    'jftools/fedvr.py',
    'jftools/interpolate.py',
    'jftools/ipynbimport.py',
    'jftools/myjit.py',
    'jftools/plotting.py',
    'jftools/shade_color.py',
    'jftools/short_iterative_lanczos.py',
      'jftools/short_iterative_lanczos_numba.py',
    'jftools/tictoc.py',
    'jftools/unroll_phase.py',
  ],
  subdir: 'jftools'
)
