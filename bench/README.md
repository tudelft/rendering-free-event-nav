# bench/ — controlled A/B for proxy-model changes

Every change to the event model is measured on the **same flight**. Closed-loop rollouts can't
do that: change the observation and the policy acts differently, so you would be comparing a
model change and a different trajectory at once. `record_ref.py` therefore rolls the champion
policy once with fixed seeds, stores the action sequence in `ref_actions.npz`, and replays
those actions forever after — identical dynamics, identical poses, event model free to vary.

Reference set: three successful flights (seeds 1003, 11, 202), 307 frames, 23 s.

```bash
GX_BG_RATE=0.0 python3 bench/record_ref.py --tag baseline     # a variant
python3 bench/score.py                                        # scorecard vs the real flights
```

`score.py` grades ten axes against the 30 onboard recordings in `analysis/data/` and reports
the mean |relative error|. All axes are scored together, so a change that fixes one by
breaking another cannot look like progress.

## Progress

| metric | REAL | baseline | + rounding fix | + background activity |
|---|---|---|---|---|
| active % | 49.2 | 19.3 | 25.2 | **47.0** |
| ON % | 35.4 | 35.5 | 53.6 | **32.1** |
| \|dev\| p50 | 2.0 | 2.0 | 1.0 | 1.0 |
| \|dev\| p90 | 16.0 | 6.0 | 5.0 | 4.0 |
| clustering | 1.29 | 3.13 | 2.65 | **1.17** |
| polarity | 0.70 | 0.58 | 0.52 | 0.88 |
| jaccard | 0.45 | 0.34 | 0.35 | 0.36 |
| row top/mid/bot | 56/52/39 | 5/37/14 | 6/41/28 | **36/58/47** |
| **mean \|err\|** | — | **49.1 %** | 51.3 % | **26.2 %** |

Still open: the magnitude tail (p90 4 vs 16), temporal persistence (jaccard 0.36 vs 0.45), and
content above the horizon (row_top 36 vs 56).
