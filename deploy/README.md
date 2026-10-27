# deploy/

`contracts.md` is the authoritative interface between the policy and the airframe: the
observation layout, the action semantics, the control rate, and the quantization steps that
take a trained checkpoint to an int8 network running on the microcontroller.

## What runs onboard

| stage | component |
|---|---|
| sensing | Prophesee GenX320, DIFF3D mode, free-running integration windows |
| observation | signed per-pixel counts, downsampled 320x320 to 64x64, 3-frame stack |
| policy | 1.2 MB int8 network, 13.3 Hz |
| actuation | yaw rate and forward speed, sent to the flight controller |

State estimation, obstacle avoidance and goal seeking all run on the board the drone carries;
there is no off-board computation and no external positioning.

## Converting a trained checkpoint to the onboard `.bin`

`convert/` turns a `checkpoints/*.pkl` actor into a Neural-ART `network_rel.bin` for the OpenMV N6,
and `convert/CONVERT.md` is the step-by-step guide (port → calibrate → int8 → stedgeai). The chain
was run end-to-end to produce `openmv_scripts/policy.bin` from `checkpoints/ours_champion.pkl`:

```bash
python deploy/convert/port_to_torch.py checkpoints/ours_champion.pkl        # Flax -> clean NCHW ONNX
python deploy/convert/make_calib.py   --from-proxy checkpoints/ours_champion.pkl --n 512
python deploy/convert/quantize_int8.py                                      # -> actor_int8.onnx
STEDGEAI=/path/to/stedgeai deploy/convert/run_stedgeai.sh                   # -> network_rel.bin
```

## Firmware and onboard scripts

The mission scripts live in `openmv_scripts/`, the custom N6 firmware notes/patches in
`firmware/`, and the Crazyflie app and flight script in `cf_apps/`. The event-camera driver and the Neural-ART runtime are part of the upstream OpenMV N6
port and are not duplicated here. `contracts.md` specifies everything needed to reimplement the
interface against a different airframe or sensor.
