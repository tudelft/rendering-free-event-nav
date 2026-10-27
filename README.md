# rendering-free-event-nav

**Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation**

An analytic event-camera simulator that synthesizes calibrated GenX320 DIFF3D observations
directly from scene geometry and ego-motion — with no rendering anywhere in the loop. It
produces a 64×64 event frame in **17 µs**, against 6.9 ms for a ray-traced pipeline, which is
fast enough to train goal-directed obstacle avoidance with PPO in **34 minutes on one RTX 4090**
(40 M environment steps at 19–21 k steps/s across 512 parallel environments).

The resulting policy transfers **zero-shot** to ray-traced Isaac Sim with a physically based
event camera, and to a real 56 g drone flying entirely onboard.

> Paper: link to follow · Video: link to follow

## Results

One checkpoint throughout, under a shared action interface (v ∈ [0, 3.5] m/s at 13.3 Hz),
evaluated in Isaac Sim + EVIS over n = 30 procedurally generated courses per condition.

| arm | training cost | shadow scene | texture scene | AGV (shadow) |
|---|---|---|---|---|
| **analytic proxy (ours)** | **34 min**, 1 GPU | **83 %** (25/30) | **70 %** (21/30) | 2.38 m/s |
| privileged ground-truth depth *(upper bound, not a peer)* | 34 min, 1 GPU | 100 % (30/30) | 97 % (29/30) | 2.19 m/s |
| PPO on rendered events (online EVIS) | 1 GPU-day (195 k steps) | 40 % | 53 % | 1.78 m/s |
| optic-flow balancing (reactive) | — | 13 % | 17 % | 1.10 m/s |

Real flights: **82 %** of flights reached the goal across three arenas of increasing difficulty
(5/5, 5/6, 4/6), with all sensing, inference and state estimation onboard, no map and no
external positioning.

The rendered-events row is an extrapolation of our own measured GPU-day throughput with this
renderer; optimised rendering stacks will do better than that figure.

## Layout

| path | contents |
|---|---|
| `proxy/` | the analytic event model and the flight/course environment it runs in |
| `training/` | PPO trainer, vectorized env wrapper, networks, in-proxy evaluation, Isaac bridge clients |
| `checkpoints/` | the policy reported in the paper |
| `calibration/` | the five-statistic calibration gate against real GenX320 recordings |
| `isaac_eval/` | closed-loop Isaac Sim 4.2 + EVIS evaluation, AvoidBench-style metrics |
| `deploy/` | the observation/action contract and the conversion chain to an int8 network onboard (`deploy/convert/`) |
| `openmv_scripts/` | the OpenMV N6 mission scripts (13.3 Hz inference loop) + the compiled `policy.bin` |
| `firmware/` | custom N6 firmware patches (the `framestack` module + GC-heap change that remove the in-flight GC freeze) |
| `cf_apps/` | the Crazyflie firmware app (UART link, action → setpoints, goal-bearing uplink) and the takeoff/hand-over/land flight script |

## Quickstart

No Isaac Sim, no dataset and no event recordings are needed to train or evaluate in the proxy.

```bash
git clone --recurse-submodules https://github.com/dfordequan/rendering-free-event-nav
cd rendering-free-event-nav
pip install -r requirements.txt            # install jax[cuda] matching your CUDA version

# evaluate the released policy (CPU is fine)
python training/eval_proxy.py checkpoints/ours_champion.pkl 30

# reproduce the training run: 40 M steps, ~34 min on an RTX 4090
bash training/configs/ours.sh
```

Nothing else is required for either command: the environment is vendored in `proxy/`, so no
agent framework, dataset, renderer or event recording needs to be installed. A short run
(`--total-steps 4e5`) is a good way to confirm the install — the first update includes JIT
compilation, after which throughput settles at 19--21 k steps/s on an RTX 4090.
Training writes `BEST.pkl`, `ckpt.pkl` and `scores.jsonl` into `--logdir`.

The submodule is only needed for the Isaac evaluation; proxy training and evaluation work
without it.

`training/configs/` also holds the privileged-depth baseline (`depth.sh`) and the Isaac
evaluation entry point (`isaac_eval.sh`).

## Reproducing the evaluations

- **In-proxy** — `training/eval_proxy.py` rebuilds the environment from the contract stored in
  the checkpoint, so the evaluation cannot drift from the arm that was trained.
- **Isaac Sim + EVIS** — needs Isaac Sim 4.2 and the EVIS plugin, included as a submodule
  (`git submodule update --init`). `isaac_eval/forest_policy_eval.py` runs the closed-loop
  server; the policy connects over a socket so the two simulators never share a process.
- **Calibration** — `calibration/` reproduces the five-statistic gate that fits the proxy's
  scalars to a real sensor. Calibration is sensor identification, not policy adaptation: it uses
  a single 10-second recording of the camera moved by hand, and no flight or task data.

## Onboard code

The OpenMV N6 mission scripts are in `openmv_scripts/` (with the compiled `policy.bin`), and the
custom N6 firmware patches in `firmware/`. `deploy/convert/` converts a trained checkpoint into the
onboard int8 `.bin` — see `deploy/convert/CONVERT.md`. The Crazyflie side is in `cf_apps/`: an
out-of-tree firmware app that implements the UART interface in `deploy/contracts.md`, and the
`cflib` script that takes off, hands control to the onboard policy and lands.

## Citing

See `CITATION.cff`.

## License

MIT for this repository (`LICENSE`). The vendored EVIS plugin under `isaac_eval/evis/` is a
submodule and carries its own MIT license from its authors.
