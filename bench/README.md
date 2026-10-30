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

Still open, in priority order:

1. **Contour continuity** — `vrun` 2.28 against 3.71. Each cylinder is point-sampled at
   `FOREST_NV_DR x FOREST_NAZ_DR` positions, so a pole's silhouette arrives as a dotted line.
   Rasterising the two silhouette generators as *segments* would make it continuous and is
   cheaper than the points it replaces.
2. **Above the horizon** — `row_top` 35 against 56. The proxy's world has no overhead
   structure; the real arena has truss, lights and curtains.
3. **Temporal persistence** — `jaccard` 0.37 against 0.45. Frames are rendered independently;
   real pixels carry a residual toward their threshold. EVIS reaches 0.435, so it is learnable.
4. **Polarity coherence** — 0.97 against 0.70, pushed the wrong way by the diffuse background.

None of this is validated against transfer yet: the submitted proxy reached 83/70/82 % *with*
a 3x density gap, so statistical agreement is a hypothesis about transfer, not evidence of it.
