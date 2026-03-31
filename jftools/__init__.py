"""Collection of small useful helper tools for Python by Johannes Feist."""

from importlib.resources import files

__version__ = files(__package__).joinpath("_version.txt").read_text(encoding="utf-8").strip()

__all__ = ["shade_color", "tic", "toc", "ipynbimport_install", "unroll_phase", "interp_cmplx", "plotcolored", "fedvr", "short_iterative_lanczos"]

from .shade_color import shade_color
from .tictoc import tic, toc
from .ipynbimport import install as ipynbimport_install
from .unroll_phase import unroll_phase
from .interpolate import interp_cmplx
from .plotting import plotcolored
from . import fedvr
from . import short_iterative_lanczos
