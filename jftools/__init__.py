"""Collection of small useful helper tools for Python by Johannes Feist."""

__version__ = '0.1'

__all__ = ['shade_color','tic','toc','ipynbimport_install']

from .shade_color import shade_color
from .tictoc import tic, toc
from .ipynbimport import install as ipynbimport_install

