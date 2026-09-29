"""Calibrate EVENT_GAIN_BY_H so the event frame's intensity statistics are comparable across
resolutions.

Why this is needed: focal length scales with H, so per-point image speed (hence event magnitude)
scales DOWN as H shrinks -- while pixel area scales as 1/H^2, so MORE edge points accumulate into
each pixel, pushing values UP. The two effects do not cancel, and the nuisance DR (salt_val,
horizon_mag) is expressed in ABSOLUTE magnitude units. So if the signal level moves with H, the
effective SNR of the task moves too, and a 32x32 policy would be solving a quietly different
problem than the 64x64 one.

We match the MEDIAN NONZERO uint8 value of the delivered observation (the same quantity the Isaac
bridge calibrates its count->uint8 gain against, target 51).

Usage: JAX_PLATFORMS=cpu python scripts/calibrate_event_gain.py [task] [episodes]
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proxy'))
from skycourse import SkyCourse
import quad_course_env as QC

BASE_TASK = sys.argv[1] if len(sys.argv) > 1 else 'magzone'
EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TICKS = 260          # past the startup ramp


def stats(task, gain_override=None):
    if gain_override is not None:
        QC.EVENT_GAIN_BY_H[_H_OF[task]] = gain_override
    e = SkyCourse(task=task, seed=0)
    d = e.act_space['action'].shape[0]
    rng = np.random.RandomState(0)
    nz, occ, sat = [], [], []
    for ep in range(EPISODES):
        o = e.step({'reset': True, 'action': np.zeros(d, np.float32)})
        for t in range(TICKS):
            a = rng.uniform(-1, 1, d).astype(np.float32)
            a[0] = abs(a[0])                      # bias forward so it flies down the corridor
            o = e.step({'reset': False, 'action': a})
            if t < 120:
                continue                          # skip the ramp
            img = np.asarray(o['image']).astype(np.float32)
            v = img[img > 0]
            if v.size:
                nz.append(np.median(v))
            occ.append((img > 0).mean())
            sat.append((img >= 255).mean())
            if bool(o['is_last']):
                break
    return (float(np.mean(nz)) if nz else 0.0,
            float(np.mean(occ)) * 100, float(np.mean(sat)) * 100, e.H)


_H_OF = {}
tasks = [BASE_TASK, BASE_TASK + 'h32']
for t in tasks:
    _H_OF[t] = SkyCourse(task=t, seed=0).H

print(f'task family: {BASE_TASK}   ({EPISODES} eps, post-ramp ticks)')
print(f'{"task":16s} {"H":>4s} {"gain":>6s} {"median nonzero":>15s} {"occupancy %":>12s} {"saturated %":>12s}')
base = None
for t in tasks:
    H = _H_OF[t]
    g = QC.steer_gain(H)
    med, occ, sat, HH = stats(t)
    print(f'{t:16s} {HH:4d} {g:6.2f} {med:15.1f} {occ:12.2f} {sat:12.2f}')
    if HH == 64:
        base = med

if base:
    H32 = _H_OF[tasks[1]]
    print(f'\nsweeping gain at H={H32} to match the H=64 median nonzero of {base:.1f} ...')
    best, bestg = None, None
    for g in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
        med, occ, sat, _ = stats(tasks[1], gain_override=g)
        flag = ''
        if best is None or abs(med - base) < abs(best - base):
            best, bestg, flag = med, g, '  <-- closest'
        print(f'  gain {g:4.2f} -> median nonzero {med:6.1f}  occupancy {occ:5.2f}%  '
              f'saturated {sat:5.2f}%{flag}')
    print(f'\nSET EVENT_GAIN_BY_H[{H32}] = {bestg}   (median {best:.1f} vs 64x64 {base:.1f})')
