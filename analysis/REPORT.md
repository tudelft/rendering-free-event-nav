# Onboard event recordings — what the real sensor actually produces

30 `PPOFLY01` flight recordings from the N6, 3076 frames, 231 s of flight. 26 successful,
4 crashed (by filename). Reproduce with:

```bash
python3 analysis/analyse_onboard_events.py      # the numbers below
python3 analysis/make_figures.py                # analysis/out/*.png
```

## 0. The reader was wrong, and is now fixed

`PPOFLY01` has a **14-byte header**, not 16: `b'PPOFLY01'` + three `uint16` (grid 64, vector
length 4, reserved). With 14, all 30 files divide exactly into 4126-byte records and the action
field decodes into `[-1, 1]`; with 16 it decodes to `-18.03` and no file divides evenly.

The reader shipped in this project (`calibration/`, and the video tooling) assumed 16. That read
every frame two pixels late in raster order, pulled two bytes of the float vector in as pixels,
and dropped the last record. Aggregate statistics barely move under a two-pixel shift — re-reading
the Table II recording correctly gives active 65.9 % vs 65.8 % published, ON 39.0 % vs 39.0 %,
p50 3.0, p90 14.0, all unchanged — but mass (16240 vs 15815) and block variance (7646 vs 6947)
do shift, and the frames themselves were wrong. Use `analysis/ppofly.py`.

## 1. Sparsity — and the paper's real row is an outlier

| | active pixels % |
|---|---|
| proxy, training distribution | 15 |
| **these 30 flights, median** | **47** |
| these 30 flights, range | 35 – 66 |
| the single flight used in Table II (`sw_s35fly_171125`) | 66 |

The recording behind the paper's "real" row sits at the **97th percentile** of the 30 flights.
The honest population statement is that the real sensor fires about **3× denser** than the proxy,
not 4.4×. Per-frame the spread is wider still: p5 = 6.6 %, p50 = 52.6 %, p95 = 87.6 %.

## 2. Density is not driven by ego-motion

| | |
|---|---|
| corr(density, commanded speed) | **+0.04** |
| corr(density, \|yaw rate\|) | −0.09 |
| density at commanded speed ≈ 0 | **51 %** |

This is the most consequential finding. The proxy's response is motion-driven by construction —
`m = g·min(‖δ‖/κ, 1)`, so no ego-motion means no events. The real sensor produces ~50 % active
pixels *regardless of commanded speed*. Rotor vibration, attitude jitter and the sensor's own
noise floor dominate, and the free-running integration window collects them.

Two caveats: commanded speed is not measured speed, and the saturating knee compresses the
high-motion end. But a correlation of 0.04 cannot be explained by either — there is a large
motion-independent component the proxy does not model.

Density instead tracks **scene content**: it falls steadily through each flight, 55 % in the first
decile to 29 % in the last, while commanded speed stays near 1.9 m/s. The drone starts among
clutter and ends facing the open goal area.

## 3. Polarity

ON is the minority polarity throughout: **31.9 %** of active pixels (sd 4.0 across flights,
range 22.8–38.8). Per frame, p5 = 7.7 %, p95 = 62.5 % — individual frames swing hard, but no
flight is ON-dominated.

Polarity is **spatially coherent, not interleaved**. An ON pixel touches an OFF pixel 62.1 % of
the time, against 88.0 % for a null that keeps the active sites and shuffles only their signs —
a ratio of 0.70. Same-polarity horizontal runs average 2.4 px (p90 = 5 px). That is the
covering/revealing signature the proxy models: the two sides of a moving edge carry opposite
signs, in patches a few pixels wide.

*(An earlier version of this test compared against a null that also scattered the sites. At ~50 %
density that null saturates near 88 % and has no power; the site-preserving null is the correct
control.)*

## 4. Clustering and structure

- 4-neighbours per active pixel: **2.50** real vs **1.92** for a density-matched shuffle → only
  **1.3× clustered**. At these densities most neighbour contacts happen by chance.
- Connected components are strongly bimodal: **65 % are single pixels**, median 1 px, p90 5 px —
  yet the largest reaches 3999 of 4096 px. Dense frames percolate into one blob; sparse frames are
  mostly isolated pixels. There is no characteristic "blob size".
- Magnitude is heavy-tailed: median |dev| = 2.4 counts, p90 = 17.1, and only 0.074 % of pixels
  reach the ±127 clip. Saturation is not what limits the representation.

## 5. Spatial layout

Activity is top-heavy: 55.9 / 52.5 / 38.9 % for the top, middle and bottom thirds of the image.
The floor produces markedly fewer events than the obstacle field and ceiling structure. Columns
are flat (49.9 / 47.9 / 49.8 %), as expected for a forward-facing camera with no lateral bias.

## 6. Temporal persistence

Consecutive frames share **0.455** of their active sets (Jaccard). At 13.3 Hz roughly half the
firing pixels persist frame to frame — enough that a 3-frame stack carries real temporal
information rather than three independent noise draws.

## 7. Successful vs crashed flights

| | n | active % | ON % | mean commanded speed |
|---|---|---|---|---|
| successful | 26 | 49.8 | 32.4 | 1.80 m/s |
| crashed | 4 | 42.2 | 28.5 | 2.28 m/s |

Crashed flights were flown faster and saw *fewer* events. With n = 4 this is suggestive only, but
the direction is consistent: more speed, less usable structure per frame.

## What this suggests for the proxy

1. **Add a motion-independent floor.** The single biggest mismatch. Real density barely responds
   to ego-motion; the proxy's response is purely motion-driven. A vibration/noise term calibrated
   to the ~50 % floor would close most of the gap without touching the edge model.
2. **Re-state the density comparison against the population**, not the densest flight.
3. **Keep the polarity model** — the covering/revealing structure is confirmed at 0.70× against
   the correct null.
4. **Model the vertical gradient**: the proxy's analytic ground shader should produce *less* floor
   activity than the obstacle field, matching 39 % vs 56 %.
