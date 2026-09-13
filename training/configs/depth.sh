#!/bin/bash
# Privileged-depth baseline: identical settings, analytic ground-truth depth observation.
# NOTE the seed: default seed collapses on this arm (documented seed pathology); seed 3 converges.
python training/train_ppo.py --task yawzoneh64r8w20gxv3sq --hold-alt \
  --vmin 0.0 --vmax 3.5 --yawrate-max 1.0 --ceiling --head-pen --props --keep-lo 0.4 \
  --depth-obs --room 22.0 --total-steps 40e6 --seed 3 --logdir runs/ppo_depth
