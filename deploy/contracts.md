# The action/observation contract

**Observation**: 64×64×1 DIFF3D event frame (three stacked 20 ms windows; inter-frame
stride = decision period) + vector[4] = [cosβ, sinβ, ψ̇/ψ̇max, |v|/vmax].
β = egocentric goal bearing from onboard state estimation only.

**Action** (tanh head, 2-dim, normalized): a0 → yaw rate ψ̇ = a0·ψ̇max through τ=0.08 s lag;
a1 → forward speed v = ½(a1+1)·vmax, slew-limited 3 m/s². +yaw = LEFT.
Contract values: ψ̇max = 1.0 rad/s, vmax = 3.5 m/s, decision rate 13.3 Hz
(alternating 7/8-tick holds on 100 Hz sim). NO output filtering anywhere.

**UART link (N6 ↔ Crazyflie)**: 115200 baud, fixed-length frames, 2-byte sync header
(0xAA55 down / 0xCF55 up), int16×10⁴ scaling, trailing XOR checksum.
The N6 runs only the policy; the Crazyflie's F405 runs stabilization at 500 Hz
with zero-order hold on the last action.

**Deploy chain**: JAX → Torch port (≤5e-6) → ONNX opset 18 (strip allowzero) →
int8 PTQ on closed-loop observations → stedgeai v4 → Neural-ART relocatable .bin
(1.2 MB full width). Conv/dense on NPU; frame assembly, normalization, tanh on Cortex-M55.
Toolchain: ARM GCC 14.3.1 + stedgeai v4 (exact versions matter).
