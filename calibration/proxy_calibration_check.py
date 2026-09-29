"""THE PROXY CALIBRATION GATE (2026-08-24). Run after EVERY proxy/DR change.

Renders N frames from the current training task and compares the standard
representation statistics against pinned references (real rec1.bin, and the
latest Isaac eval reference). Prints PASS/WARN per statistic and appends to
docs/CALIBRATION_LOG.md. Robustness DR is allowed to SPREAD the distribution;
this gate checks the CENTER hasn't drifted.

Usage: JAX_PLATFORMS=cpu python scripts/proxy_calibration_check.py [task] [note]
"""
import os, sys, struct, datetime
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'proxy'))
import numpy as np

TASK = sys.argv[1] if len(sys.argv) > 1 else 'yawzoneh32r10w20gxv1sc'
NOTE = sys.argv[2] if len(sys.argv) > 2 else ''
# pinned references: real bench + Isaac gx6b evals (update deliberately, never silently)
REF = {
    'real rec1.bin':  dict(bg=39.2, p50=2, p90=12, blob=1.57, on=48.8),
    'Isaac (gx6b)':   dict(bg=57.3, p50=4, p90=13, blob=1.62, on=44.4),
}
TOL = dict(bg=12.0, p50=2.0, p90=5.0, blob=0.20, on=8.0)   # WARN outside these

from skycourse import SkyCourse
env = SkyCourse(task=TASK, seed=123)
obs = env.step({'action': np.zeros(2, np.float32), 'reset': True})
rng = np.random.default_rng(0); frames = []
for t in range(400):
    obs = env.step({'action': np.clip(rng.normal(0, .5, 2).astype(np.float32), -1, 1),
                    'reset': bool(obs['is_last'])})
    frames.append(obs['image'])
fr = np.stack(frames)
d = np.abs(fr.astype(float) - 128.0); act = d > 0.5; a = d[act]
m = act[..., 0]
n4 = (m[:, 1:, :] & m[:, :-1, :]).sum() + (m[:, :, 1:] & m[:, :, :-1]).sum()
cur = dict(bg=100 * (1 - act.mean()), p50=np.percentile(a, 50),
           p90=np.percentile(a, 90), blob=n4 / max(m.sum(), 1),
           on=100 * (fr.astype(float) > 128.5).sum() / max(act.sum(), 1))

lines = [f"## {datetime.date.today()} task={TASK} {NOTE}",
         f"current: " + "  ".join(f"{k}={v:.2f}" for k, v in cur.items())]
status = 'PASS'
for k, v in cur.items():
    # compare against the CLOSER of the two references (either anchor is fine)
    dist = min(abs(v - REF[r][k]) for r in REF)
    flag = 'ok' if dist <= TOL[k] else 'WARN'
    if flag == 'WARN':
        status = 'WARN'
    lines.append(f"  {k:5s} {v:7.2f}  nearest-ref dist {dist:6.2f} ({flag}, tol {TOL[k]})")
lines.append(f"=> {status}")
out = "\n".join(lines)
print(out)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_log.md'), 'a') as f:
    f.write(out + "\n\n")
