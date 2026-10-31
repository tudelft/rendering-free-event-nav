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

Mean |relative error| over 13 axes against the 30 real flights:

| step | what changed | mean err |
|---|---|---|
| baseline | as submitted | 43.8 % |
| rounding fix | `rint` instead of truncating `astype` in the uint8 map — the truncation silently deleted every weak ON event | (revealed true 54 % ON) |
| step 1 | motion-independent background activity, applied last | 23.9 % |
| step 2a | log-normal per-edge contrast (`GX_GAIN_SIGMA`) | 22.8 % |
| step 2b | background occupancy decoupled from magnitude, log-normal spread | 21.7 % |
| step 2c | background magnitude median raised to 2 counts | **16.8 %** |

Defaults now ship the step-2c calibration. Any axis can be re-swept with the env vars
(`GX_BG_RATE`, `GX_BG_ON`, `GX_BG_SCALE`, `GX_BG_SIGMA`, `GX_GAIN_SIGMA`).

| metric | REAL | baseline | now |
|---|---|---|---|
| active % | 49.2 | 19.3 | 47.9 |
| ON % | 35.4 | 35.5 | 30.5 |
| p50 / p90 | 2 / 16 | 2 / 6 | 2 / 14 |
| clustering | 1.29 | 3.13 | 1.19 |
| polarity | 0.70 | 0.58 | 0.97 |
| jaccard | 0.45 | 0.34 | 0.37 |
| row top/mid/bot | 56/52/39 | 5/37/14 | 35/58/50 |
| vrun | 3.71 | 2.48 | 2.28 |
| anisotropy | 1.02 | 0.83 | 0.96 |
| elongation | 2.07 | 2.62 | 2.14 |

Step 4 (rebalance): **16.1 %**. Step 2 and 3 scored well but looked *worse* by eye, and the
scorecard could not see why. Pooled p50/p90 cannot tell "strong contours over a weak floor"
from "uniformly medium everything", so a heavy-tailed BACKGROUND satisfied them by scattering
high-magnitude pixels at random — which reads as snow. `clust_s` (clustering restricted to
the strongest 15 % of events) makes it visible: real 3.08, baseline 3.47, step 2 **1.64**.

The underlying error was that the background was stronger than the edges: an edge point
deposits ~4.5 net units (0.7 uint8 counts) while each background event was set to 12.5 net
units (2 counts). `GX_GAIN_MUL` raises the edge deposit so the contour population sits clearly
above the floor; `GX_BG_SIGMA` drops so the floor stays weak.

| | REAL | baseline | step 3 | step 4 |
|---|---|---|---|---|
| vrun_s | 2.30 | 2.32 | 1.56 | 1.92 |
| aniso_s | 1.10 | 0.86 | 1.08 | 1.11 |
| clust_s | 3.08 | 3.47 | 1.90 | 2.33 |
| mean err | — | 37.6 % | 17.8 % | **16.1 %** |

Still open