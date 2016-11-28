from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import exp, angle
from . import unroll_phase

arg = lambda x: unroll_phase(angle(x))

def interp_cmplx(x,y,*args,absarg=True,interpolator=InterpolatedUnivariateSpline,**kwargs):
    if absarg:
        return interp_cmplx_absarg(interpolator,x,y,*args,**kwargs)
    else:
        return interp_cmplx_reim  (interpolator,x,y,*args,**kwargs)

class interp_cmplx_absarg:
    def __init__(self,interpolator,x,y,*args,**kwargs):
        self.abs = interpolator(x,abs(y),*args,**kwargs)
        self.arg = interpolator(x,arg(y),*args,**kwargs)
    def __call__(self,*args,**kwargs):
        return self.abs(*args,**kwargs)*exp(1j*self.arg(*args,**kwargs))

class interp_cmplx_reim:
    def __init__(self,interpolator,x,y,*args,**kwargs):
        self.real = interpolator(x,y.real,*args,**kwargs)
        self.imag = interpolator(x,y.imag,*args,**kwargs)
    def __call__(self,*args,**kwargs):
        return self.real(*args,**kwargs) + 1j*self.imag(*args,**kwargs)
