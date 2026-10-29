"""Figures for the onboard-event analysis -> analysis/out/*.png"""
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppofly

OUT = 'analysis/out'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 130})
paths = sorted(glob.glob('analysis/data/onboard_events/**/*.bin', recursive=True))
recs = {os.path.relpath(p, 'analysis/data/onboard_events'): ppofly.read(p) for p in paths}
dev = np.concatenate([r['dev'] for r in recs.values()])
act = np.abs(dev) > 0.5

fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
# a: per-recording density, flagging the one used in the paper
names = list(recs); d_rec = np.array([(np.abs(recs[n]['dev']) > 0.5).mean()*100 for n in names])
o = np.argsort(d_rec); cols = ['#c63030' if 'crashed' in names[i] else '#1f6fd2' for i in o]
ax[0,0].bar(range(len(o)), d_rec[o], color=cols)
pap = [i for i, n in enumerate(names) if '171125' in n]
if pap:
    ax[0,0].bar([list(o).index(pap[0])], [d_rec[pap[0]]], color='#e0a010')
    ax[0,0].annotate('used in Table II', (list(o).index(pap[0]), d_rec[pap[0]]),
                     xytext=(-55, -22), textcoords='offset points', fontsize=7.5, color='#9a7000',
                     arrowprops=dict(arrowstyle='->', color='#9a7000', lw=.8))
ax[0,0].axhline(np.median(d_rec), ls='--', lw=1, color='k')
ax[0,0].text(0.5, np.median(d_rec)+1.5, f'median {np.median(d_rec):.0f}%', fontsize=7.5)
ax[0,0].axhline(15, ls=':', lw=1.2, color='#1c8250')
ax[0,0].text(0.5, 16.5, 'proxy (training) 15%', fontsize=7.5, color='#1c8250')
ax[0,0].set(title='(a) event density per flight', xlabel='flight (sorted)', ylabel='active pixels %')

# b: per-frame density distribution
ax[0,1].hist(act.mean((1,2))*100, bins=50, color='#1f6fd2')
ax[0,1].set(title='(b) per-frame density', xlabel='active pixels %', ylabel='frames')

# c: magnitude distribution
a = np.abs(dev)[act]
ax[0,2].hist(a, bins=np.arange(1, 60), color='#1f6fd2', log=True)
ax[0,2].set(title='(c) event magnitude |counts|', xlabel='|dev| per active pixel', ylabel='pixels (log)')

# d: connected-component sizes
comp = np.load(f'{OUT}/component_sizes.npy')
ax[1,0].hist(comp, bins=np.logspace(0, np.log10(comp.max()), 40), color='#1f6fd2')
ax[1,0].set_xscale('log'); ax[1,0].set_yscale('log')
ax[1,0].set(title='(d) connected-component size', xlabel='pixels per component', ylabel='count')

# e: spatial profile
rowp = np.load(f'{OUT}/row_profile.npy'); colp = np.load(f'{OUT}/col_profile.npy')
ax[1,1].plot(rowp, np.arange(64), color='#1f6fd2', label='row')
ax[1,1].invert_yaxis(); ax[1,1].set(title='(e) vertical activity profile',
                                     xlabel='active %', ylabel='image row (top=0)')
ax[1,1].legend(fontsize=7)

# f: temporal persistence
f0, f1 = act[:-1], act[1:]
jac = (f0 & f1).sum((1,2)) / np.maximum((f0 | f1).sum((1,2)), 1)
ax[1,2].hist(jac, bins=40, color='#1f6fd2')
ax[1,2].axvline(jac.mean(), ls='--', color='k')
ax[1,2].set(title=f'(f) frame-to-frame overlap (mean {jac.mean():.2f})',
            xlabel='Jaccard of active sets', ylabel='frame pairs')
fig.tight_layout(); fig.savefig(f'{OUT}/onboard_event_stats.png'); plt.close(fig)

# sample frames: sparsest, median, densest
fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.0))
per = act.mean((1,2))
for a_, i, t in zip(ax, [per.argmin(), np.argsort(per)[len(per)//2], per.argmax()],
                    ['sparsest', 'median', 'densest']):
    a_.imshow(np.clip(dev[i]/40, -1, 1), cmap='bwr_r', vmin=-1, vmax=1)
    a_.set_title(f'{t}  ({per[i]*100:.0f}% active)', fontsize=9); a_.axis('off'); a_.grid(False)
fig.tight_layout(); fig.savefig(f'{OUT}/onboard_sample_frames.png'); plt.close(fig)
print('wrote', OUT + '/onboard_event_stats.png and onboard_sample_frames.png')
