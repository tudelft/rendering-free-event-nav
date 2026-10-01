# Converting a trained checkpoint to an N6-deployable `.bin`

This turns a `checkpoints/*.pkl` actor (Flax) into a Neural-ART `network_rel.bin` that runs on the
OpenMV N6 (STM32N6 + Neural-ART NPU). Four steps; each is one script. The result is the file the
mission script loads as `/sdcard/policy.bin` (see `openmv_scripts/`).

```
Flax ckpt ──port_to_torch.py──▶ actor_fp32.onnx  (clean NCHW, Conv/Gemm only)
                                      │
calib obs ──make_calib.py──▶ calib.npz
                                      ▼
                        quantize_int8.py ──▶ actor_int8.onnx  (QDQ, per-channel)
                                      ▼
                        run_stedgeai.sh ──▶ network_rel.bin   (copy to /sdcard/policy.bin)
```

Everything below was run end-to-end to produce the committed `openmv_scripts/policy.bin` from
`checkpoints/ours_champion.pkl`.

## Prerequisites

- **Port / export / quantize**: `torch`, `onnx`, `onnxruntime`, `numpy`. Loading the `.pkl` also needs
  `jax` (the arrays are jax arrays); `flax` is **not** required (parity uses an independent NumPy
  reference, so the port is validated without importing the training stack).
- **Compile**: **ST Edge AI Core v4** (`stedgeai`) and the Neural-ART config `neuralart.json` that
  ships with the OpenMV N6 port (`openmv/lib/stai/scripts/neuralart.json`). Also ARM GCC 14.3.
  Exact tool versions matter (contracts.md).

## Step 1 — port Flax → Torch → clean ONNX  (`port_to_torch.py`)

```bash
python deploy/convert/port_to_torch.py checkpoints/ours_champion.pkl \
    --out-onnx deploy/convert/out/actor_fp32.onnx --out-pt deploy/convert/out/actor.pt
# -> parity: max|flax-torch| = 9.5e-07  OK
```

Why a re-port rather than exporting the Flax graph directly, and the two things it gets right:

1. **NCHW, no Transpose.** A Flax actor exported as-is carries an NHWC input + a Transpose that
   stedgeai mis-infers into a scrambled board tensor `[b:1,h:64,w:3,c:64]` → **E103**. The Torch twin
   is NCHW end to end; the compiled board input is the correct `[b:1,h:64,w:64,c:3]`.
2. **The conv→dense flatten reorder.** Flax flattens the conv trunk in **H,W,C** order; Torch (NCHW)
   flattens in **C,H,W**. The first dense's 4096 conv-input columns are reordered HWC→CHW so the two
   graphs compute the identical result with **no Transpose in the graph**. (Conv kernels are also
   transposed `[kh,kw,in,out]→[out,in,kh,kw]`, dense weights transposed `[in,out]→[out,in]`.)
3. **SAME padding fused into Conv.** Flax Conv uses asymmetric SAME padding (Torch Conv2d can't);
   the script pads explicitly for the parity model, then **fuses those pads into each Conv's `pads`
   attribute** in the exported ONNX (removing the dynamic `Pad`/`ConstantOfShape` subgraph), giving a
   Conv/Gemm/Relu-only graph.

The graph outputs the raw action **mean**; `a = tanh(mean)` is applied on the Cortex-M55 (two scalar
ops) in the mission script — it is intentionally **not** in the graph. The `(x-128)/32` input affine
**is** the first op so it folds into conv0's int8 scale.

## Step 2 — representative observations for PTQ  (`make_calib.py`)

Quantization ranges must reflect the deploy input distribution. Two sources:

```bash
# A) real GenX320 recordings (IMAVEO01 .bin), box-downsampled 320->64 and 3-stacked as the board does:
python deploy/convert/make_calib.py --from-events /path/to/recordings --n 512 \
    --out deploy/convert/out/calib.npz
# B) self-contained: roll the analytic proxy policy and dump the obs it sees (needs the training env):
python deploy/convert/make_calib.py --from-proxy checkpoints/ours_champion.pkl --n 512 \
    --out deploy/convert/out/calib.npz
```

The committed `policy.bin` was calibrated on real recordings (source A) — the true deploy image
distribution. Source B needs no recordings and is the reproducible default.

## Step 3 — int8 quantization  (`quantize_int8.py`)

```bash
python deploy/convert/quantize_int8.py \
    --onnx deploy/convert/out/actor_fp32.onnx \
    --calib deploy/convert/out/calib.npz \
    --out  deploy/convert/out/actor_int8.onnx
```

QDQ format, per-channel int8 weights, int8 activations. The script re-strips the Reshape `allowzero`
attribute afterwards (the quantizer can re-add it; stedgeai v4 rejects it).

## Step 4 — compile to the Neural-ART `.bin`  (`run_stedgeai.sh`)

```bash
STEDGEAI=/path/to/stedgeai \
NEURALART=$PWD/../openmv/lib/stai/scripts/neuralart.json \
  deploy/convert/run_stedgeai.sh deploy/convert/out/actor_int8.onnx deploy/convert/out
# -> OK -> deploy/convert/out/network_rel.bin  (~1.15 MB)
cp deploy/convert/out/network_rel.bin openmv_scripts/policy.bin   # ready to flash to /sdcard
```

Under the hood: `stedgeai generate --target stm32n6 --type onnx --relocatable
--input-data-type float32 --output-data-type float32 --st-neural-art default@<neuralart.json>`.

**Verify the compile**: open `out/network_generate_report.txt` and confirm the input tensor is
`[b:1,h:64,w:64,c:3]` (NOT `[b:1,h:64,w:3,c:64]` — that scramble is the E103 symptom) and that
`acts_sz` (~96 KB) fits AXISRAM and `params_sz` (~1.12 MB) fits external flash.

## Gotchas (each cost a debugging session)

- **Wrong `neuralart.json` → E103.** Use the one from the OpenMV N6 port, not the one bundled with
  the OpenMV IDE.
- **Exporting the Flax/NHWC graph directly → E103** via the `[b:1,h:64,w:3,c:64]` board-tensor
  scramble. The NCHW re-port with the flatten reorder is the fix (step 1).
- **`Reshape` `allowzero`** is emitted by both torch export and the onnxruntime quantizer; stedgeai v4
  rejects it. Both scripts strip it.
- **Don't put tanh in the graph.** Keep the graph linear-headed and squash on the M55 — matches the
  contract and keeps the NPU pass clean.
- **Toolchain versions are load-bearing** — ST Edge AI Core v4 + ARM GCC 14.3 (contracts.md).

## Architecture note (if your checkpoint differs)

`port_to_torch.py` assumes the champion trunk/head widths (16/32/64 conv, 256/128 dense, 64×64×3 in,
vec[4], action[2]) — the default in `training/nets.py`. If you trained with `PPO_WIDTHS` set, pass the
matching widths (edit the `cw, dw` in `main()`); the parity check will catch any mismatch before you
waste a compile.
