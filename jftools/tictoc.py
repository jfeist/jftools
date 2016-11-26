from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    tt = time() - _tstart_stack.pop()
    if fmt:
        print (fmt % tt)
    return tt
