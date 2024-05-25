from .myjit import jit
import numpy as np


def unroll_phase(phases):
    # first convert to numpy array
    phases = np.asarray(phases)
    # then make sure it is at least float64
    phases = np.asarray(phases, dtype=np.promote_types(phases.dtype, np.float64))
    return _unroll_phase(phases)


@jit(nopython=True)
def _unroll_phase(phases):
    TWOPI = 2 * np.pi
    # acc_phase keeps track of how many phase jumps by TWOPI we have accumulated
    acc_phase = 0.0
    for ii in range(1, len(phases)):
        # the phase difference between the previous and current step,
        # including a possible phase jump of +-TWOPI
        phase_diff = phases[ii] - phases[ii - 1] - acc_phase
        if abs(phase_diff - TWOPI) < abs(phase_diff):
            # if +TWOPI
            acc_phase += TWOPI
            phase_diff -= TWOPI
        elif abs(phase_diff + TWOPI) < abs(phase_diff):
            # if -TWOPI
            acc_phase -= TWOPI
            phase_diff += TWOPI
        phases[ii] = phases[ii - 1] + phase_diff
    return phases
