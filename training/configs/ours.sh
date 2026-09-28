#!/bin/bash
# Champion arm ("ours"): 0-3.5 m/s, 13.3 Hz contract, props, 22 m rooms. ~34 min on RTX 4090.
python training/train_ppo.py --task yawzoneh64r7w20gxv3sq --hold-alt \
  --vmin 0.0 --vmax 3.5 --yawrate-max 1.0 --ceiling --head-pen --props --keep-lo 0.4 \
  --room 22.0 --total-steps 40e6 --logdir runs/ppo_ours
