"""Statistical analysis of the onboard GenX320 DIFF3D recordings.

Covers: sparsity, polarity, magnitude/saturation, spatial clustering against a
density-matched null, ON/OFF adjacency (the covering/revealing signature), spatial
layout, temporal persistence, and successful vs crashed flights.

Usage: python3 analysis/analyse_onboard_events.py [--out analysis/out]
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppofly

ACT_T = 0.5          # |dev| above this counts as an event (matches the paper's frame stats)
RNG = np.random.default_rng(0)


def per_frame_stats(dev):
    """dev [N,G,G] int16 -> dict of per-frame arrays."""
    act = np.abs(dev) > ACT_T
    n_act = act.sum((1, 2))
    pos = (dev > ACT_T).sum((1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        on_frac = np.where(n_act > 0, pos / np.maximum(n_act, 1), np.nan)
    # 4-neighbour count per active pixel, and the same for a density-matched shuffle
    def nb(a):
        c = np.zeros_like(a, dtype=np.int16)
        c[:, 1:, :] += a[:, :-1, :]; c[:, :-1, :] += a[:, 1:, :]
        c[:, :, 1:] += a[:, :, :-1]; c[:, :, :-1] += a[:, :, 1:]
        return c
    blob = np.where(n_act > 0, (nb(act) * act).sum((1, 2)) / np.maximum(n_act, 1), np.nan)
    sh = np.stack([RNG.permutation(f.ravel()).reshape(f.shape) for f in act])
    blob_null = np.where(n_act > 0, (nb(sh) * sh).sum((1, 2)) / np.maximum(n_act, 1), np.nan)
    return dict(active=n_act / act[0].size * 100, on=on_frac * 100, blob=blob,
                blob_null=blob_null, mass=np.abs(dev).sum((1, 2)))


def onoff_adjacency(dev):
    """Is polarity spatially coherent?

    Fraction of ON pixels with an OFF pixel in their 8-neighbourhood, against a null that
    keeps the active SITES exactly and shuffles only which of them are ON vs OFF. That
    isolates polarity structure from density: a null that also scatters the sites saturates
    at these densities (~88%) and has no power. Ratio < 1 means same-polarity pixels clump,
    i.e. ON and OFF sit on opposite sides of an edge rather than interleaving.
    """
    k = np.ones((3, 3), bool); k[1, 1] = False
    real, null = [], []
    for d in dev:
        on, off = d > ACT_T, d < -ACT_T
        if on.sum() < 20 or off.sum() < 20:
            continue
        real.append((on & ndimage.binary_dilation(off, k)).sum() / on.sum())
        idx = np.flatnonzero((on | off).ravel()); RNG.shuffle(idx)
        s = np.zeros(d.size, np.int8); s[idx[:on.sum()]] = 1; s[idx[on.sum():]] = -1
        s = s.reshape(d.shape)
        null.append(((s > 0) & ndimage.binary_dilation(s < 0, k)).sum() / max((s > 0).sum(), 1))
    return float(np.mean(real)), float(np.mean(null))


def polarity_runs(dev, sign):
    """Horizontal run lengths of one polarity - the width of same-sign patches."""
    out = []
    for d in dev:
        m = (d > ACT_T) if sign > 0 else (d < -ACT_T)
        pad = np.zeros((m.shape[0], 1), bool)
        f = np.concatenate([pad, m, pad], 1).ravel()
        dif = np.diff(f.astype(np.int8))
        out.append(np.flatnonzero(dif == -1) - np.flatnonzero(dif == 1))
    return np.concatenate(out) if out else np.array([])


def component_sizes(dev, cap=40000):
    sizes = []
    for d in dev:
        lab, n = ndimage.label(np.abs(d) > ACT_T)
        if n:
            sizes.append(np.bincount(lab.ravel())[1:])
        if sum(len(s) for s in sizes) > cap:
            break
    return np.concatenate(sizes) if sizes else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='analysis/data/onboard_events')
    ap.add_argument('--out', default='analysis/out')
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(A.data, '**', '*.bin'), recursive=True))
    assert paths, f'no recordings under {A.data}'

    rows, all_dev = [], []
    for p in paths:
        r = ppofly.read(p)
        s = per_frame_stats(r['dev'])
        name = os.path.relpath(p, A.data)
        rows.append(dict(name=name, crashed='crashed' in name, n=r['n'],
                         active=float(np.nanmean(s['active'])), on=float(np.nanmean(s['on'])),
                         blob=float(np.nanmean(s['blob'])), blob_null=float(np.nanmean(s['blob_null'])),
                         mass=float(np.mean(s['mass'])),
                         p50=float(np.percentile(np.abs(r['dev'])[np.abs(r['dev']) > ACT_T], 50)),
                         p90=float(np.percentile(np.abs(r['dev'])[np.abs(r['dev']) > ACT_T], 90)),
                         sat=float((np.abs(r['dev']) >= 127).mean() * 100),
                         speed=float(np.mean((r['act'][:, 1] + 1) / 2 * 3.5))))
        all_dev.append(r['dev'])
    dev = np.concatenate(all_dev)
    json.dump(rows, open(os.path.join(A.out, 'per_recording.json'), 'w'), indent=1)

    A_ = np.array([r['active'] for r in rows]); O = np.array([r['on'] for r in rows])
    B = np.array([r['blob'] for r in rows]); BN = np.array([r['blob_null'] for r in rows])
    cr = np.array([r['crashed'] for r in rows])
    adj, adj_null = onoff_adjacency(dev[::3])
    comp = component_sizes(dev[::3])
    gs = per_frame_stats(dev)

    print(f'=== {len(paths)} recordings, {dev.shape[0]} frames, {dev.shape[0]/13.3:.0f} s ===\n')
    print('SPARSITY        active %%: mean %.1f  sd %.1f  range %.1f-%.1f (per recording)'
          % (A_.mean(), A_.std(), A_.min(), A_.max()))
    print('                per-frame : p5 %.1f  p50 %.1f  p95 %.1f'
          % tuple(np.nanpercentile(gs['active'], [5, 50, 95])))
    print('POLARITY        ON %% of active: mean %.1f  sd %.1f  range %.1f-%.1f'
          % (O.mean(), O.std(), O.min(), O.max()))
    print('                per-frame ON%%: p5 %.1f  p50 %.1f  p95 %.1f'
          % tuple(np.nanpercentile(gs['on'], [5, 50, 95])))
    print('MAGNITUDE       |dev| p50 %.1f  p90 %.1f   saturated(>=127) %.3f %% of pixels'
          % (np.mean([r['p50'] for r in rows]), np.mean([r['p90'] for r in rows]),
             np.mean([r['sat'] for r in rows])))
    print('CLUSTERING      4-neighbours/active: real %.2f  shuffled %.2f  -> %.1fx clustered'
          % (B.mean(), BN.mean(), B.mean() / BN.mean()))
    print('                components: n %d  median %d px  p90 %d px  max %d px  single-pixel %.0f %%'
          % (len(comp), np.median(comp), np.percentile(comp, 90), comp.max(), (comp == 1).mean() * 100))
    ron, roff = polarity_runs(dev[::3], 1), polarity_runs(dev[::3], -1)
    print('POLARITY LAYOUT ON pixels touching an OFF pixel: real %.1f %%  polarity-shuffled null %.1f %%'
          % (adj * 100, adj_null * 100))
    print('                ratio %.2f  -> same-polarity pixels clump (ON and OFF segregate)' % (adj / adj_null))
    print('                horizontal same-polarity run: ON mean %.2f p90 %.0f | OFF mean %.2f p90 %.0f px'
          % (ron.mean(), np.percentile(ron, 90), roff.mean(), np.percentile(roff, 90)))
    print('OUTCOME         successful (n=%d): active %.1f %%  ON %.1f %%  speed %.2f m/s'
          % ((~cr).sum(), A_[~cr].mean(), O[~cr].mean(),
             np.mean([r['speed'] for r in rows if not r['crashed']])))
    print('                crashed    (n=%d): active %.1f %%  ON %.1f %%  speed %.2f m/s'
          % (cr.sum(), A_[cr].mean(), O[cr].mean(),
             np.mean([r['speed'] for r in rows if r['crashed']])))
    rowp = (np.abs(dev) > ACT_T).mean((0, 2)) * 100
    colp = (np.abs(dev) > ACT_T).mean((0, 1)) * 100
    print('LAYOUT          row activity top/mid/bottom third: %.1f / %.1f / %.1f %%'
          % (rowp[:21].mean(), rowp[21:43].mean(), rowp[43:].mean()))
    print('                column activity left/centre/right: %.1f / %.1f / %.1f %%'
          % (colp[:21].mean(), colp[21:43].mean(), colp[43:].mean()))
    f0 = (np.abs(dev[:-1]) > ACT_T); f1 = (np.abs(dev[1:]) > ACT_T)
    jac = (f0 & f1).sum((1, 2)) / np.maximum((f0 | f1).sum((1, 2)), 1)
    print('TEMPORAL        consecutive-frame active-set overlap (Jaccard): %.3f' % jac.mean())
    json.dump(dict(active_mean=A_.mean(), on_mean=O.mean(), blob=B.mean(), blob_null=BN.mean(),
                   adj=adj, adj_null=adj_null, frames=int(dev.shape[0])),
              open(os.path.join(A.out, 'summary.json'), 'w'), indent=1)
    np.save(os.path.join(A.out, 'row_profile.npy'), rowp)
    np.save(os.path.join(A.out, 'col_profile.npy'), colp)
    np.save(os.path.join(A.out, 'component_sizes.npy'), comp)


if __name__ == '__main__':
    main()
