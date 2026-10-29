"""Score proxy variants against the real onboard recordings.

Each metric is scored as a signed relative error against the real target, and the overall
figure of merit is the mean |relative error| over the scored metrics — lower is better.
Metrics whose target the proxy already matches are included, so a change that fixes one axis
by breaking another cannot look like progress.

  python3 bench/score.py                       # score every run in bench/runs
  python3 bench/score.py baseline step1        # just these
"""
import glob, os, sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import ppofly

ACT = 0.5
RNG = np.random.default_rng(0)
K8 = np.ones((3, 3), bool); K8[1, 1] = False
# scored axes: name -> (higher-is-not-better is irrelevant; we score |relative error|)
AXES = ['active_pct', 'on_pct', 'p50', 'p90', 'clustering', 'polarity', 'jaccard',
        'row_top', 'row_mid', 'row_bot']


def _nb(a):
    c = np.zeros_like(a, np.int16)
    c[:, 1:, :] += a[:, :-1, :]; c[:, :-1, :] += a[:, 1:, :]
    c[:, :, 1:] += a[:, :, :-1]; c[:, :, :-1] += a[:, :, 1:]
    return c


def metrics(dev):
    act = np.abs(dev) > ACT
    a = np.abs(dev)[act]
    n = act.sum((1, 2))
    blob = (_nb(act) * act).sum((1, 2)) / np.maximum(n, 1)
    sh = np.stack([RNG.permutation(f.ravel()).reshape(f.shape) for f in act])
    bn = (_nb(sh) * sh).sum((1, 2)) / np.maximum(sh.sum((1, 2)), 1)
    real_adj, null_adj = [], []
    for d in dev[::2]:
        on, off = d > ACT, d < -ACT
        if on.sum() < 20 or off.sum() < 20:
            continue
        real_adj.append((on & ndimage.binary_dilation(off, K8)).sum() / on.sum())
        idx = np.flatnonzero((on | off).ravel()); RNG.shuffle(idx)
        s = np.zeros(d.size, np.int8); s[idx[:on.sum()]] = 1; s[idx[on.sum():]] = -1
        s = s.reshape(d.shape)
        null_adj.append(((s > 0) & ndimage.binary_dilation(s < 0, K8)).sum() / max((s > 0).sum(), 1))
    f0, f1 = act[:-1], act[1:]
    jac = ((f0 & f1).sum((1, 2)) / np.maximum((f0 | f1).sum((1, 2)), 1)).mean()
    rows = act.mean((0, 2)) * 100
    return dict(active_pct=act.mean() * 100,
                on_pct=(dev > ACT).sum() / max(act.sum(), 1) * 100,
                p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)),
                clustering=float(np.nanmean(blob) / np.nanmean(bn)),
                polarity=float(np.mean(real_adj) / np.mean(null_adj)),
                jaccard=float(jac),
                row_top=float(rows[:21].mean()), row_mid=float(rows[21:43].mean()),
                row_bot=float(rows[43:].mean()))


def real_target():
    paths = sorted(glob.glob(os.path.join(ROOT, 'analysis/data/onboard_events/**/*.bin'),
                             recursive=True))
    return metrics(np.concatenate([ppofly.read(p)['dev'] for p in paths]))


def main():
    tags = sys.argv[1:] or [os.path.basename(p)[:-4]
                            for p in sorted(glob.glob(os.path.join(HERE, 'runs', '*.npz')))]
    T = real_target()
    runs = {}
    for t in tags:
        f = os.path.join(HERE, 'runs', t + '.npz')
        if not os.path.exists(f):
            print(f'  (no run {t})'); continue
        runs[t] = metrics(np.load(f)['frames'].astype(np.int16) - 128)
    w = max(len(t) for t in runs) + 2 if runs else 10
    print(f'{"metric":<12}{"REAL":>8}' + ''.join(f'{t:>{w}}' for t in runs))
    print('-' * (20 + w * len(runs)))
    for k in AXES:
        line = f'{k:<12}{T[k]:>8.2f}'
        for t in runs:
            line += f'{runs[t][k]:>{w}.2f}'
        print(line)
    print('-' * (20 + w * len(runs)))
    line = f'{"mean |err|":<12}{"--":>8}'
    for t in runs:
        e = np.mean([abs(runs[t][k] - T[k]) / max(abs(T[k]), 1e-6) for k in AXES]) * 100
        line += f'{e:>{w}.1f}'
    print(line + '   %  (lower is better)')


if __name__ == '__main__':
    main()
