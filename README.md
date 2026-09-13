# rendering-free-event-nav

**Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation**

An analytic event-camera simulator that synthesizes calibrated GenX320 DIFF3D observations
directly from scene geometry and ego-motion — no rendering — fast enough to train
goal-directed obstacle-avoidance policies with PPO in **34 minutes on one RTX 4090**
(19–21k env steps/s across 512 parallel environments). Policies transfer **zero-shot**
to ray-traced Isaac Sim (EVIS event model) and to a real 56 g Crazyflie flying fully
onboard on an STM32N6 microcontroller.

> Paper: [link TBD] · Real-flight video: **[placeholder — coming with camera-ready]**

## Results (all policies one checkpoint, shared action contract v∈[0,3.5] m/s @ 13.3 Hz)

| arm | training cost | SR shadow (n=30) | SR texture (n=30) | AGV |
|---|---|---|---|---|
| **PPO, analytic proxy (ours)** | 34 min, 1 GPU | **83%** | **70%** | 2.38 m/s |
| PPO, privileged depth | 30 min, 1 GPU | 80% | 80% | 2.85 m/s |
| Flow balancing (reactive) | — | 13% | 17% | 1.10 m/s |
| PPO, rendered events (online EVIS) | 1 GPU-day (195k steps) | 40% | 53% | 1.78 m/s |

Real world: 100% / 83% / 67% success across three arenas (5/5, 5/6, 4/6 flights),
all sensing, inference, and state estimation onboard.

## Layout

- `proxy/` — the analytic event model (edge points, saturating contrast trigger, DIFF3D mapping) + training environments
- `training/` — PPO (`train_ppo.py`), vectorized env wrapper, networks, Isaac bridge clients
- `calibration/` — the five-statistic calibration gate against real GenX320 recordings
- `isaac_eval/` — closed-loop Isaac Sim 4.2 + EVIS evaluation server, AvoidBench-style metrics
- `deploy/` — action/observation contract; ONNX→int8→Neural-ART pipeline docs (see `contracts.md`)
- `firmware/`, `openmv_scripts/`, `cf_apps/` — flight firmware forks and mission scripts (submodules/inventory pending)

## Quickstart (proxy training, no Isaac needed)

```bash
pip install -r requirements.txt          # jax[cuda] per your CUDA version
bash training/configs/ours.sh            # 40M steps, ~34 min on RTX 4090
python calibration/proxy_calibration_check.py   # five-statistic gate vs reference stats
```

Isaac evaluation additionally needs NVIDIA Isaac Sim 4.2 (`ISAAC_SIM_PATH`) and the
EVIS plugin (`isaac_eval/evis` submodule). Checkpoints for every paper arm are published
as GitHub Release assets.

## License

MIT (this repository). Firmware forks keep their upstream licenses in their own submodules
(crazyflie-firmware: GPL-3.0; openmv: MIT). EVIS plugin: MIT (SPIKE Lab, JHU).
