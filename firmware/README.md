# firmware/ — placeholder

The onboard flight firmware is not in this repository yet.

What will land here: the GenX320 DIFF3D driver and integration-window logic, and the
Neural-ART runtime that executes the quantized policy on the OpenMV N6 board.

Until then, `deploy/contracts.md` fully specifies the observation layout, the action
semantics and the control rate, which is what you need to reimplement the interface
against a different airframe or sensor.
