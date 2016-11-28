try:
    from numba import jit
except:
    import warnings
    warnings.warn('jftools: not using numba - can accelerate some computations!')
    # in principle, jit can be used without options as well, which would make it trickier to deal with
    # but we only use the second form, so no problem
    def jit(*args,**kwargs):
        def g(f):
            return f
        return g
