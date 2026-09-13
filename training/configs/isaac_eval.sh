#!/bin/bash
# Closed-loop Isaac evaluation, one seed batch (10 episodes). Repeat for seeds 55/66/77 (n=30).
# SCENE: default (shadow) | hangar (texture).  Requires ISAAC_SIM_PATH and the evis submodule.
SEED=${1:-55}; SCENE=${2:-default}; CKPT=${3:-runs/ppo_ours/BEST.pkl}; PORT=5790
REPO=$(git rev-parse --show-toplevel)
PYTHONPATH=$REPO/isaac_eval:$REPO/isaac_eval/evis \
$ISAAC_SIM_PATH/python.sh $REPO/isaac_eval/forest_policy_eval.py --task sqroom --wall-style plain \
  --obs-res 64 --steer-mode rate --speed-mode cmd2rate --yaw-tau 0.08 --genx --dt-evt 0.02 \
  --genx-p90-target 8 --vmin 0.0 --vmax 3.5 --yawrate-max 1.0 --act-hold-alt 7,8 --room 22.0 \
  --props --episodes 10 --save-frames 0 --scene-style $SCENE --seed $SEED --port $PORT --out out_${SCENE}_s$SEED &
sleep 5
JAX_PLATFORMS=cpu python training/isaac_client_ppo.py $CKPT $PORT
