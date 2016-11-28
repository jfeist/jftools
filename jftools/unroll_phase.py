from .myjit import jit
import numpy as np

def unroll_phase(phases):
    return _unroll_phase(np.asarray(phases))

@jit(nopython=True)
def _unroll_phase(phases):
    TWOPI = 2*np.pi
    # acc_phase keeps track of how many phase jumps by TWOPI we have accumulated
    acc_phase = 0.
    for ii in range(1,len(phases)):
        # the phase difference between the previous and current step,
        # including a possible phase jump of +-TWOPI
        phase_diff = phases[ii] - phases[ii-1] - acc_phase
        if abs(phase_diff-TWOPI)<abs(phase_diff):
            # if +TWOPI
            acc_phase  += TWOPI
            phase_diff -= TWOPI
        elif abs(phase_diff+TWOPI)<abs(phase_diff):
            # if -TWOPI
            acc_phase  -= TWOPI
            phase_diff += TWOPI
        phases[ii] = phases[ii-1] + phase_diff
    return phases
