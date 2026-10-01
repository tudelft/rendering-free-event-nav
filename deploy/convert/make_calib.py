"""Step 2: build a representative observation set for int8 post-training quantization (PTQ).

Quantization ranges should reflect the *deployment* input distribution. Two sources:

  --from-events DIR   : real GenX320 IMAVEO01 recordings (320x320 DIFF3D), box-downsampled 320->64
                        and 3-frame-stacked exactly as the board does -> the true deploy image
                        distribution. Best when you have hand/flight recordings of the real sensor.
  --from-proxy CKPT   : roll the analytic proxy policy and dump the observations it actually sees.
                        Fully self-contained (needs the training env: jax+flax), no recordings.

The vector branch [cosβ, sinβ, ψ̇/ψ̇max, |v|/vmax] is sampled across its full valid range so the
dense-0 ranges are covered regardless of source.

Output: an .npz with image[N,64,64,3] uint8 (NCHW is applied by the quantizer) and vector[N,4] f32.

Usage:
  python deploy/convert/make_calib.py --from-events /path/to/recordings --n 512 \
      --out deploy/convert/out/calib.npz
  python deploy/convert/make_calib.py --from-proxy checkpoints/ours_champion.pkl --n 512 \
      --out deploy/convert/out/calib.npz
"""
import argparse, glob, os, struct
import numpy as np

H = 64


def _box_down(frame320, out=64):
    """320x320 -> 64x64 by 5x5 box average (deploy downsample), keeping the uint8 DIFF3D centering."""
    b = frame320.shape[0] // out                       # 5
    return frame320[:out*b, :out*b].reshape(out, b, out, b).mean((1, 3)).round().astype(np.uint8)


def _rand_vec(rng, n):
    beta = rng.uniform(-np.pi, np.pi, n)
    yaw = rng.uniform(-1, 1, n)
    v = rng.uniform(0, 1, n)
    return np.stack([np.cos(beta), np.sin(beta), yaw, v], 1).astype(np.float32)


def from_events(paths, n, rng):
    META = '<IIBhhhhBhhhhhh'
    imgs = []
    for p in paths:
        d = open(p, 'rb').read()
        if d[:8] != b'IMAVEO01':
            continue
        W, Hh, ML, _ = struct.unpack('<HHHH', d[8:16]); REC = ML + W*Hh; nf = (len(d)-16)//REC
        small = np.empty((nf, H, H), np.uint8)
        for i in range(nf):
            q = 16 + i*REC + ML
            small[i] = _box_down(np.frombuffer(d[q:q+W*Hh], np.uint8).reshape(Hh, W))
        # 3-frame stacks spaced by 1 (deploy stacks the last 3 decision frames)
        for i in range(2, nf):
            imgs.append(np.stack([small[i-2], small[i-1], small[i]], -1))
    if not imgs:
        raise SystemExit('no IMAVEO01 frames found under --from-events')
    imgs = np.array(imgs)
    sel = rng.choice(len(imgs), min(n, len(imgs)), replace=False)
    return imgs[sel]


def from_proxy(ckpt_path, n, rng):
    import pickle, jax, jax.numpy as jnp, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'training'))
    from env_vec import make_env
    from nets import Actor
    from eval_proxy import CONTRACT
    ck = pickle.load(open(ckpt_path, 'rb')); meta = ck['meta']
    reset, step, _ = make_env(ck['task'], meta['frame_stack'], **CONTRACT)
    actor = Actor(adim=meta['adim'])
    step_j, apply_j = jax.jit(step), jax.jit(actor.apply)
    imgs = []
    key = jax.random.PRNGKey(0)
    while len(imgs) < n:
        key, k = jax.random.split(key)
        state, stack, obs = reset(k)
        for _ in range(meta['max_steps']):
            imgs.append(np.asarray(obs['image']))
            mean, _ = apply_j(ck['actor'], obs['image'][None], obs['vector'][None])
            key, k = jax.random.split(key)
            state, stack, obs, r, done, info = step_j(state, stack, jnp.tanh(mean[0]), k)
            if bool(done) or len(imgs) >= n:
                break
    return np.array(imgs[:n]).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from-events', default=None, help='dir of IMAVEO01 .bin recordings')
    ap.add_argument('--from-proxy', default=None, help='checkpoint to roll for representative obs')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--out', default='deploy/convert/out/calib.npz')
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    rng = np.random.default_rng(0)
    if A.from_events:
        paths = sorted(glob.glob(os.path.join(A.from_events, '*.bin')))
        img = from_events(paths, A.n, rng)
    elif A.from_proxy:
        img = from_proxy(A.from_proxy, A.n, rng)
    else:
        raise SystemExit('pass --from-events DIR or --from-proxy CKPT')
    vec = _rand_vec(rng, len(img))
    np.savez(A.out, image=img, vector=vec)
    print(f'wrote {A.out}: image {img.shape} {img.dtype}, vector {vec.shape}')


if __name__ == '__main__':
    main()
