# openmv_scripts/ — onboard mission scripts (OpenMV N6)

MicroPython that runs on the OpenMV N6 and implements `deploy/contracts.md`: a 13.3 Hz feedforward
inference loop over the GenX320 event stream, the goal-bearing handling, and the UART link to the
Crazyflie.

| file | what it does |
|---|---|
| `ppo_nav_main.py` | the flight loop: build the 3-frame 64×64 obs, run the int8 policy, `a=tanh(mean)` on the M55, send `[a0,a1]` to the CF, receive the goal vector. Deploy as `main.py`. |
| `ppo_nav_fly_rec.py` | same loop, plus records a `PPOFLY01` flight trace (obs + goal + action) for videos and re-calibration. |
| `policy.bin` | the champion policy compiled for the N6 (from `checkpoints/ours_champion.pkl` via `deploy/convert/`). Copy to `/sdcard/policy.bin`. |

## Deploy

1. Build custom N6 firmware with the `framestack` module (see `firmware/`), or run on stock firmware
   — the scripts auto-detect (`import framestack`) and fall back to a ulab path. The custom firmware
   removes the in-flight GC freeze and is what the reported flights used.
2. Copy `policy.bin` to the board's SD card as `/sdcard/policy.bin`.
3. Copy `ppo_nav_main.py` to the board as `main.py` (or run it from the OpenMV IDE).
4. Wire UART4 (115200) to the Crazyflie; the CF streams the goal vector (`0xCF55 …`) and consumes the
   action (`0xAA55 …`). The CF maps `a0→yaw rate (×1.0 rad/s)` and `a1→speed ((a1+1)/2×3.5 m/s)` and
   runs stabilization at 500 Hz with zero-order hold — see `deploy/contracts.md`.

## Making your own `policy.bin`

`policy.bin` is produced from a trained checkpoint by `deploy/convert/` — see
`deploy/convert/CONVERT.md`. The observation/action contract the scripts implement is
`deploy/contracts.md`.
