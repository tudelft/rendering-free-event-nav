"""Deterministic (mean-action) evaluation of a checkpoint inside the analytic proxy.

Rebuilds the environment from the contract stored in the checkpoint, so the evaluation
cannot silently drift from the arm that was trained.

Usage:
  python training/eval_proxy.py checkpoints/ours_champion.pkl [n_episodes]
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import numpy as np

from env_vec import make_env
from nets import Actor

# The champion contract (paper Sec. IV-A). These are the settings the reported
# checkpoint was trained under; eval_proxy reuses them so scores are comparable.
CONTRACT = dict(vmin=0.0, vmax=3.5, yawrate_max=1.0, ceiling=True, head_pen=True,
                props=True, keep_lo=0.4, room=22.0, hold_alt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('n', nargs='?', type=int, default=30)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    ck = pickle.load(open(args.ckpt, 'rb'))
    meta = ck['meta']
    env_reset, env_step, _ = make_env(ck['task'], meta['frame_stack'], **CONTRACT)
    actor = Actor(adim=meta['adim'])
    step_j, apply_j = jax.jit(env_step), jax.jit(actor.apply)

    reach = 0
    speeds = []
    for i in range(args.n):
        key = jax.random.PRNGKey(args.seed0 + i)
        state, stack, obs = env_reset(key)
        got, v = False, []
        for _ in range(meta['max_steps']):
            mean, _ = apply_j(ck['actor'], obs['image'][None], obs['vector'][None])
            key, k = jax.random.split(key)
            state, stack, obs, r, done, info = step_j(state, stack, jnp.tanh(mean[0]), k)
            v.append(float(np.linalg.norm(np.array(state['cs'].v)[:2])))
            got |= bool(info['reached'])
            if bool(done):
                break
        reach += got
        speeds.append(float(np.mean(v)) if v else 0.0)
        if not args.quiet:
            print(f'episode {i:3d}: {"reached" if got else "failed ":8} '
                  f'mean speed {speeds[-1]:.2f} m/s', flush=True)
    print(f'\n{ck["task"]}: {reach}/{args.n} = {100*reach/args.n:.0f}% goal-reach, '
          f'mean speed {np.mean(speeds):.2f} m/s')


if __name__ == '__main__':
    main()
