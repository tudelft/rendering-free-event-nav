#!/usr/bin/env bash
# Step 4: compile the int8 ONNX to a Neural-ART relocatable .bin for the OpenMV N6.
#
# Produces <OUT>/network_rel.bin -- copy that to the board's SD card under the name the mission
# script expects (openmv_scripts/ppo_nav_main.py -> /sdcard/policy.bin).
#
# Requires ST Edge AI Core v4 (`stedgeai`) and the Neural-ART config `neuralart.json`. Using the
# WRONG neuralart.json fails with E103 -- point NEURALART at the one shipped with the OpenMV N6 port.
#
# Usage:
#   STEDGEAI=/path/to/stedgeai NEURALART=/path/to/neuralart.json \
#     deploy/convert/run_stedgeai.sh deploy/convert/out/actor_int8.onnx deploy/convert/out
set -euo pipefail

ONNX="${1:-deploy/convert/out/actor_int8.onnx}"
OUT="${2:-deploy/convert/out}"
WS="${OUT}/ws"
STEDGEAI="${STEDGEAI:-stedgeai}"
NEURALART="${NEURALART:?set NEURALART to the neuralart.json of the OpenMV N6 port (openmv/lib/stai/scripts/neuralart.json)}"

mkdir -p "$OUT" "$WS"
"$STEDGEAI" generate \
  --model "$ONNX" --target stm32n6 --type onnx --relocatable \
  --input-data-type float32 --output-data-type float32 \
  --st-neural-art "default@${NEURALART}" \
  --workspace "$WS" --output "$OUT"

echo
if [ -f "$OUT/network_rel.bin" ]; then
  echo "OK -> $OUT/network_rel.bin  ($(stat -c%s "$OUT/network_rel.bin") bytes)"
  echo "Copy it to the board as /sdcard/policy.bin (or the name set in openmv_scripts/ppo_nav_main.py)."
else
  echo "stedgeai finished but network_rel.bin was not produced -- check $OUT/network_generate_report.txt"
  exit 1
fi
