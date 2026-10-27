"""Calibrate the single-channel GenX320 (`gx`) observation against the RECORDED sensor data.

What the hardware actually does (openmv/drivers/sensors/genx320.c in the OpenMV N6 port,
read 2026-08-18):

    1. EHC DIFF3D accumulates a SIGNED net (ON-OFF) per pixel at 320x320 over a 20 ms window
       (INTEGRATION_DEF_PREIOD = 20000 us, EHC_DIFF3D_N_BITS_SIZE = 7 -> signed 8-bit).
    2. post_process_histo:  px = USAT(net * CONTRAST_DEFAULT(16) + BRIGHTNESS_DEFAULT(128))
       -- per pixel, AT 320x320, so a single pixel saturates at |net| = 8.
    3. build_img: draw_image(..., hint=image.AREA) 320 -> 32 = a 10x10 MEAN of those uint8s.

    => output cell = 128 + (16/100) * SUM over the 10x10 block of the (clipped) net
                   = 128 + 0.16 * (block event count, signed)

So `contrast` is 16 ONLY against a native-resolution int8 net. Our proxy frame is ALREADY a pooled
aggregate (render at STEER_H, sum-pool to 32), so reusing 16 overdrives the frame by roughly the
bin area. The scale that belongs in the wrapper is fitted here, from the recording.

Method: the free parameters are the deposit gain (how many net units an edge lays down per window)
and the wrapper scale GENX_CONTRAST (uint8 counts per net unit). They are degenerate in the final
image except through mag_clip, so we sweep the gain, fit the contrast per gain to the recorded
ACTIVE-amplitude distribution, and report the whole quantile curve plus the background fraction.

We match the distribution of |px-128| CONDITIONED ON being active rather than the raw active
fraction, because the active fraction is scene content (rec1.bin is a hand moving close to the
lens, so it fills the frame; a forest frame is mostly empty sky) while the per-edge amplitude is
set by the sensor pipeline.

Usage: JAX_PLATFORMS=cpu python calibration/calibrate_genx.py [task] [episodes]
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import os, sys, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proxy'))
import quad_course_env as QC
from skycourse import SkyCourse

REC = os.environ.get('GENX_REC', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rec1.bin'))
TASK = os.environ.get('GXTASK', 'yawzoneh32r10w20gx')
EPISODES = int(os.environ.get('GXEPS', '8'))
TICKS = 120
WARMUP = 12           # skip the vmin->vmax speed ramp
QS = (50, 75, 90, 95, 99)


def load_real(path=REC):
    d = open(path, 'rb').read()
    assert d[:8] == b'GENXSW02', d[:8]
    G, C, VLEN, _ = struct.unpack('<HHHH', d[8:16])
    assert C == 1, 'not a single-channel recording'
    rec = 12 + G * G * C + VLEN * 4 + 8
    n = (len(d) - 16) // rec
    out = np.stack([np.frombuffer(d[16 + i * rec + 12: 16 + i * rec + 12 + G * G], np.uint8)
                    .reshape(G, G) for i in range(n)])
    return out.astype(np.int16) - 128          # signed deviation from the 128 background


def sim_net(task, gain, episodes=EPISODES, seed=0):
    """Raw (pre-contrast) net frames along a fixed-speed / sinusoidal-yaw sweep -- the same
    maneuver n6_sweep_record_1ch.py flies, so sim and real see comparable ego-motion."""
    import jax
    QC.EVENT_GAIN_BY_H_DT[(32, 0.02)] = gain
    e = SkyCourse(task=task, seed=seed)
    adim = e.act_space['action'].shape[0]
    frames, t = [], 0
    o = e.step({'reset': True, 'action': np.zeros(adim, np.float32)})
    key = jax.random.PRNGKey(seed)
    for step in range(episodes * TICKS):
        a = np.zeros(adim, np.float32)
        a[0] = 0.075 * np.sin(2 * np.pi * 0.5 * step * 0.1)   # +-15 deg/s at yawrate_max 3.5
        if adim > 1:
            a[1] = 0.6                                         # steady cruise
        o = e.step({'reset': False, 'action': a})
        t += 1
        if t >= WARMUP and not o['is_last']:
            # RAW pre-contrast net: _build_obs would have already quantised it to uint8
            key, k = jax.random.split(key)
            fr, _, _ = e._obs_st(e._state, k, e._obs_xy, e._obs_r, e._pts, e._valid, e._drp)
            frames.append(np.asarray(fr, np.float32))
        if o['is_last']:
            o = e.step({'reset': True, 'action': np.zeros(adim, np.float32)})
            t = 0
    return np.stack(frames)


def curve(dev, label):
    dev = np.abs(dev)
    act = dev[dev > 0]
    line = '  %-22s zero %5.1f%%   ' % (label, 100.0 * np.mean(dev == 0))
    line += '  '.join('p%d %6.2f' % (q, np.percentile(act, q)) for q in QS)
    return line


def main():
    real = load_real()
    print('REAL rec1.bin: %d frames, mean px %.2f, max |dev| %d, saturated %.3f%%'
          % (len(real), 128 + real.mean(), np.abs(real).max(),
             100.0 * np.mean((real <= -128) | (real >= 127))))
    print(curve(real, 'real'))
    ra = np.abs(real)[np.abs(real) > 0]
    rq = {q: np.percentile(ra, q) for q in QS}

    print('\nSIM task %s (%d episodes, ticks %d-%d)' % (TASK, EPISODES, WARMUP, TICKS))
    print('%-8s %-9s %s' % ('gain', 'contrast', 'fitted-scale match (uint8 units)'))
    for gain in (0.10, 0.15, 0.25, 0.40, 0.60, 1.00, 1.60):
        net = sim_net(TASK, gain)[..., 0]
        sa = np.abs(net)[np.abs(net) > 1e-6]
        if sa.size < 100:
            print('  gain %-6.2f  -- frame essentially empty, skipped' % gain)
            continue
        # contrast that puts the sim p90 of the active amplitude on the real p90
        c = rq[90] / np.percentile(sa, 90)
        dev = np.rint(np.clip(net * c, -128, 127))
        err = np.mean([abs(np.percentile(np.abs(dev)[np.abs(dev) > 0], q) - rq[q]) for q in QS])
        print('  gain %-6.2f  contrast %6.3f   mean|dq| %5.2f' % (gain, c, err))
        print(curve(dev, 'sim g=%.2f c=%.2f' % (gain, c)))


if __name__ == '__main__':
    main()
