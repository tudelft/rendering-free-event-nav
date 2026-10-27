# cf_apps/ — Crazyflie side of the onboard stack

The Crazyflie half of the real-flight setup: an out-of-tree firmware app that turns the policy's
action from the OpenMV N6 into flight setpoints and streams the observation vector back, plus the
host script that takes off, hands control to the onboard policy, logs, and lands. It implements the
CF side of `deploy/contracts.md`; the N6 side is `openmv_scripts/`.

During the autonomous segment nothing runs off-board: the N6 runs the policy, the Crazyflie runs
the link, the goal bearing (from its own EKF on the Flow deck) and stabilization. The host script
only flies takeoff and landing and records telemetry over the radio.

| path | what it does |
|---|---|
| `tiny_event_oa/` | Crazyflie firmware app (`src/n6_uart.c`): UART2 link to the N6, `a0 → yaw rate`, `a1 → forward speed`, τ = 0.08 s yaw lag + 3 m/s² speed slew, link watchdog, goal-bearing uplink. |
| `n6_mission.py` | cflib host script: takeoff to cruise altitude, push the `n6.*` parameters, hand over (`n6.enable = 1`), monitor, and land in place on Ctrl+C, the 30 s cap or low battery. Writes `n6_mission_<stamp>.csv`. |
| `flightlog.py` | the CSV logger used by the mission script (EKF pose, Flow-deck health, N6 action and goal bearing). |
| `mocap_udp.py` | optional, **log-only** motion-capture ground truth from a [UnifiedMocapRouter](https://github.com/tudelft/UnifiedMocapRouter) `udp` agent. Never fed back to the drone. |

## Hardware

- Crazyflie 2.1 Brushless + Flow deck v2 (state estimation), OpenMV N6 + Prophesee GenX320 (policy).
- Wiring, 3.3 V logic, 115200 8N1, crossed, common GND:
  CF `PA2` (UART2 TX) → N6 UART4 RX, CF `PA3` (UART2 RX) ← N6 UART4 TX.

## Build and flash

The app builds against stock [bitcraze/crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware);
the reported flights used commit `c7d46ad2` (master, 2026-07-30). No firmware patches are needed.

```bash
# next to this repository (or anywhere, and pass CRAZYFLIE_BASE=/path/to/crazyflie-firmware)
git clone --recursive https://github.com/bitcraze/crazyflie-firmware
git -C crazyflie-firmware checkout c7d46ad2 && git -C crazyflie-firmware submodule update --init --recursive

cd rendering-free-event-nav/cf_apps/tiny_event_oa
make cf21bl_defconfig      # brushless platform + DSHOT
make -j                    # -> build/cf21bl.bin
make cload                 # flash over the Crazyradio
```

Use the **defconfig**, not `make PLATFORM=cf21bl`: the latter only renames the output and silently
builds brushed CF2 firmware, which will not fly a brushless drone. Check with
`grep -E "CF21BL|DSHOT" build/.config`. Built and tested with `arm-none-eabi-gcc` 13.2.

`app-config` enables the app and force-loads the `n6uart` deck driver, which opens UART2 at boot.

## Flying

```bash
pip install cflib
CFLIB_URI=radio://0/80/2M/E7E7E7E7E7 python n6_mission.py
```

1. Flash and power the N6 (`openmv_scripts/`). It runs the policy continuously; the Crazyflie
   ignores it until hand-over.
2. Place the drone on a textured, well-lit floor, nose pointing along the intended "forward". The
   EKF is reset at takeoff, so the goal `(GOAL_X, GOAL_Y)` is in that takeoff frame (default 5 m
   straight ahead).
3. Run the script. It takes off, hovers 1.5 s, sets `n6.enable = 1` and stops commanding; the app
   then flies on the policy's actions until the mission cap, low battery or Ctrl+C, after which the
   script disables the app and lands in place.

Do the first run with props off. If the link drops for 700 ms the app ramps to a hover.

## Parameters

The app exposes everything as `n6.*` parameters. The mission script sets them before takeoff; these
are the values used for the reported flights:

| parameter | value | meaning |
|---|---|---|
| `n6.cruiseAlt` | 0.8 m | altitude hold during the autonomous segment |
| `n6.yawScale` | 1.0 rad/s | `a0 = 1` → yaw rate; also the obs[2] normalization (contract ψ̇max) |
| `n6.vmin`, `n6.vmax` | 0, 3.5 m/s | `a1 = −1 … +1` → speed; `vmax` is also the obs[3] normalization (contract vmax) |
| `n6.maxSpeed` | 3.5 m/s | hard clamp on the commanded speed |
| `n6.posMode`, `n6.lookahead` | 1, 0.6 m | position "carrot" setpoint 0.6 m ahead of the nose |
| `posCtlPid.xVelMax`, `yVelMax` | 3.5 m/s | position-controller speed cap in carrot mode |
| `n6.yawTau`, `n6.speedSlew` | 0.08 s, 3 m/s² | action shaping, as in training (firmware defaults) |
| `n6.linkTimeoutMs` | 700 ms | no valid N6 frame for this long → hover (firmware default) |

In carrot mode (`posMode = 1`), the app places a position setpoint `lookahead` ahead of the nose
whenever the speed command is above zero. The position controller then sets the forward speed,
so `a1` acts mainly as a go/hover gate. `posMode = 0` sends body-velocity setpoints at the
commanded speed instead, which follows the contract literally.

The Flow deck saturates at roughly v/h ≈ 7 rad/s. Keep `n6.maxSpeed` within that for your flying
height.

## Flow-deck coexistence

UART2 traffic can corrupt the Flow deck's SPI reads. The app avoids this in two ways:

- UART2 is opened in the forced deck driver at boot, before the flow task starts, rather than
  lazily from the app.
- Transmission is polled rather than interrupt-driven; interrupt-driven RX is safe.

Both choices are commented in `src/n6_uart.c`.

## Logs

Each run writes `n6_mission_<stamp>.csv` at 20 Hz with these columns:

- `time_s` (host clock) and `cf_ts` (onboard clock)
- EKF pose `x y z yaw`
- Flow-deck health `deltaX deltaY zrange varX`
- `vbat`
- the N6 action `a0raw a1raw`
- the shaped commands `yawRateCmd` (deg/s) and `speedCmd` (m/s)
- the uplinked goal bearing `goalx goaly`

With `USE_MOCAP = True` it also logs `mx my mz m_ts`.
