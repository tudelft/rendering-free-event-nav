"""Matched-trajectory renderer comparison: replay Isaac-logged sqroom episodes
(pose recovered per frame) through the analytic proxy over the SAME obstacle
layout; compare proxy frames vs the actual Isaac+EVIS frames pairwise.
Usage: python matched_traj_compare.py <ep_dir> <out_prefix> [neps]
"""
import os, sys, glob
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proxy'))
import numpy as np, jax, jax.numpy as jnp
import quad_course_env as QC
import event_proxy as EP
from skycourse import SkyCourse

EPD, OUT = sys.argv[1], sys.argv[2]
NEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
X0, XG, HW, ZT = 0.0, 22.0, 11.0, 3.0
VMAX, YRM, ACT_HOLD = 1.5, 1.0, 7

env = SkyCourse(task='yawzoneh64r7w20gxv3sq', seed=0)
obs_fn = env._obs_st
NPTS = None

def wall_points():
    L = []
    def wline(p0, p1, z, n=90):
        t = np.linspace(0, 1, n)[:, None]
        xy = p0[None] * (1 - t) + p1[None] * t
        return np.concatenate([xy, np.full((n, 1), z)], -1)
    xa, xb = X0 - 1.0, XG + 1.0
    for z in (0.2, ZT):
        L += [wline(np.array([xa, -HW]), np.array([xb, -HW]), z),
              wline(np.array([xa, HW]), np.array([xb, HW]), z),
              wline(np.array([xa, -HW]), np.array([xa, HW]), z),
              wline(np.array([xb, -HW]), np.array([xb, HW]), z)]
    zs = np.linspace(0.2, ZT, 10)
    for cx, cy in ((xa, -HW), (xa, HW), (xb, -HW), (xb, HW)):
        L.append(np.stack([np.full(10, cx), np.full(10, cy), zs], -1))
    return np.concatenate(L, 0)

WALLS = wall_points()
NEUTRAL_DRP = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.8, 0.0])  # noise0, gain1, knee1, grain.8, dilate off

def stats5(frames):
    d = np.abs(frames.astype(float) - 128.0); act = d > 0.5; a = d[act]
    m = act[..., 0] if frames.ndim == 4 else act
    n4 = (m[:, 1:, :] & m[:, :-1, :]).sum() + (m[:, :, 1:] & m[:, :, :-1]).sum()
    return dict(bg=100 * (1 - act.mean()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), blob=n4 / max(m.sum(), 1),
                on=100 * (frames.astype(float) > 128.5).sum() / max(act.sum(), 1))

all_iso, all_prox, corrs = [], [], []
strip = []
for f in sorted(glob.glob(EPD + '/ep*.npz'))[:NEPS]:
    d = np.load(f, allow_pickle=True)
    frames, vecs, traj = d['frames'], d['vecs'], d['traj']
    if frames.ndim < 4 or len(frames) < 4: continue
    oxy, orr, oh = d['obs_xy'], d['obs_r'], d['obs_h']
    trunks = EP.cylinder_edge_points(jnp.array(oxy), jnp.array(orr), jnp.array(oh),
                                     nv=QC.FOREST_NV_DR, naz=QC.FOREST_NAZ_DR)
    pts = jnp.concatenate([trunks, jnp.array(WALLS)], 0)
    valid = jnp.ones(pts.shape[0], bool)
    for i in range(len(frames)):
        ti = min(i * ACT_HOLD, len(traj) - 1)
        p = traj[ti].copy(); vec = vecs[i]
        beta = np.arctan2(vec[1], vec[0])
        theta_w = np.arctan2(0.0 - p[1], XG - p[0])
        psi = theta_w - beta
        spd = max(float(vec[3]) * VMAX, 0.05)
        s = QC.CState(p=jnp.array(p),
                      v=jnp.array([np.cos(psi) * spd, np.sin(psi) * spd, 0.0]),
                      q=jnp.array([np.cos(psi / 2), 0, 0, np.sin(psi / 2)]),
                      w=jnp.array([0.0, 0.0, float(vec[2]) * YRM]),
                      Om=jnp.zeros(4), prev_dist=jnp.float32(0), target=jnp.int32(0))
        fr, _, _ = obs_fn(s, jax.random.PRNGKey(i), jnp.array(oxy), jnp.array(orr),
                          pts, valid, NEUTRAL_DRP)
        prox = np.asarray(jnp.clip(fr * QC.GENX_CONTRAST + QC.GENX_BRIGHTNESS, 0, 255)).astype(np.uint8)
        if prox.ndim == 2: prox = prox[..., None]
        iso = frames[i]
        all_prox.append(prox); all_iso.append(iso)
        a = np.abs(iso.astype(float) - 128).ravel(); b = np.abs(prox.astype(float) - 128).ravel()
        if a.std() > 0 and b.std() > 0:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
        if len(strip) < 8 and i in (2, 6):
            strip.append((iso[..., 0], prox[..., 0]))

iso = np.stack(all_iso); prox = np.stack(all_prox)
si, sp = stats5(iso), stats5(prox)
print(f'{len(all_iso)} matched frames from {NEPS} episodes')
print(f"{'stat':>6} {'Isaac+EVIS':>11} {'proxy':>8}")
for k in si: print(f'{k:>6} {si[k]:11.2f} {sp[k]:8.2f}')
print(f'event-mass correlation: mean {np.mean(corrs):.3f}  median {np.median(corrs):.3f}  n={len(corrs)}')
np.savez(OUT + '.npz', iso=iso, prox=prox, corrs=np.array(corrs))
from PIL import Image
rows = [np.concatenate([np.concatenate([s[0] for s in strip], 1)], 1),
        np.concatenate([np.concatenate([s[1] for s in strip], 1)], 1)]
Image.fromarray(np.concatenate(rows, 0)).resize((128 * len(strip), 256), Image.NEAREST).save(OUT + '_strip.png')
print('saved', OUT + '_strip.png')
