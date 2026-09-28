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

## Firmware

The flight firmware (the event-camera driver, the quantized-network runtime and the mission
layer) is maintained in separate repositories tied to their vendor SDKs, and is not vendored
here. `contracts.md` specifies everything needed to reimplement the interface against a
different airframe or sensor.
