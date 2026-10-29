"""Fixed-environment reference recording for proxy-model work.

The point is a controlled A/B: every variant of the event model must be measured on the
SAME flight. Closed-loop rollouts can't do that — change the observation and the policy
acts differently and the trajectory diverges, so you'd be measuring the model change and a
different flight at once.

So the first call rolls the champion policy once with a fixed seed and stores the action
sequence; every later call REPLAYS those actions. Same seed, same scene, same actions ->
identical dynamics and identical camera poses, with the event model the only thing free to
change.

  python3 bench/record_ref.py --tag baseline          # creates bench/ref_actions.npz, records
  python3 bench/record_ref.py --tag step1_bgfloor     # replays, records under the new model

The set is three successful flights (seeds 1003, 11, 202; ~23 s, ~307 frames). One 10 s flight
was the original plan, but three cost nothing to render and cut the per-metric variance.
"""
import argparse, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, 'training'), os.path.join(ROOT, 'proxy')]
import jax, jax.numpy as jnp, numpy as np
from env_vec import make_env
from nets import Actor

CKPT = os.path.join(ROOT, 'checkpoints', 'ours_champion.pkl')
CONTRACT = dict(vmin=0.0, vmax=3.5, yawrate_max=1.0, ceiling=True, head_pen=True,
                props=True, keep_lo=0.4, room=22.0, hold_alt=True)
ACTIONS = os.path.join(HERE, 'ref_actions.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1003, 11, 202])
    ap.add_argument('--max-steps', type=int, default=140)
    ap.add_argument('--out', default=os.path.join(HERE, 'runs'))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    ck = pickle.load(open(CKPT, 'rb'))
    meta = ck['meta']
    reset, step, _ = make_env(ck['task'], meta['frame_stack'], **CONTRACT)
    step_j, apply_j = jax.jit(step), jax.jit(Actor(adim=meta['adim']).apply)

    store = dict(np.load(ACTIONS)) if os.path.exists(ACTIONS) else None
    out_acts, all_frames, all_poses, lens = {}, [], [], []
    for seed in A.seeds:
        replay = store[f's{seed}'] if store is not None else None
        key = jax.random.PRNGKey(seed)
        state, stack, obs = reset(key)
        frames, acts, poses = [], [], []
        for i in range(A.max_steps if replay is None else len(replay)):
            if replay is not None:
                a = jnp.asarray(replay[i])
            else:
                mean, _ = apply_j(ck['actor'], obs['image'][None], obs['vector'][None])
                a = jnp.tanh(mean[0])
            cs = state['cs']
            poses.append(np.concatenate([np.array(cs.p), np.array(cs.q),
                                         np.array(cs.v), np.array(cs.w)]))
            frames.append(np.array(obs['image'][..., -1]))
            acts.append(np.array(a))
            key, k = jax.random.split(key)
            state, stack, obs, r, done, info = step_j(state, stack, a, k)
            if bool(done) and replay is None:
                break
        out_acts[f's{seed}'] = np.array(acts)
        all_frames.append(np.array(frames)); all_poses.append(np.array(poses))
        lens.append(len(frames))
    if store is None:
        np.savez_compressed(ACTIONS, **out_acts)
        print(f'wrote {ACTIONS} — later runs replay these actions')
    else:
        same = all(np.allclose(store[k], out_acts[k]) for k in out_acts)
        print(f'replayed fixed actions, identical = {same}')
    frames = np.concatenate(all_frames); poses = np.concatenate(all_poses)
    out = os.path.join(A.out, f'{A.tag}.npz')
    np.savez_compressed(out, frames=frames, poses=poses, lens=np.array(lens),
                        seeds=np.array(A.seeds))
    print(f'{out}: {frames.shape[0]} frames ({frames.shape[0]/13.3:.1f} s) from {len(lens)} flights '
          f'{lens}, active {np.mean(np.abs(frames.astype(np.int16)-128) > 0.5)*100:.1f}%')


if __name__ == '__main__':
    main()
