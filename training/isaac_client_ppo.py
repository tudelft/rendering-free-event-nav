"""Isaac bridge client for PPO checkpoints — TCP protocol served by
isaac_eval/forest_policy_eval.py (b'O' + flags + reward + vector + image /
action floats back). Maintains the frame stack client-side.

Usage: JAX_PLATFORMS=cpu python training/isaac_client_ppo.py <ckpt.pkl> [port]
"""
import os
import pickle
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
import jax.numpy as jnp

def build_actor_from_ck(ck):
    # width-adaptive rebuild: nets.Actor's trunk widths changed after the h64 arm
    # (16/32/64 -> 32/128/128); read every layer width from the saved params.
    P = ck['actor']['params']
    cw = [P['ConvTrunk_0'][f'Conv_{i}']['kernel'].shape[-1] for i in range(3)]
    dv = P['Dense_0']['kernel'].shape[-1]
    d1 = P['Dense_1']['kernel'].shape[-1]
    d2 = P['Dense_2']['kernel'].shape[-1]
    adim = P['Dense_3']['kernel'].shape[-1]
    class ConvTrunk(nn.Module):
        @nn.compact
        def __call__(self, img):
            x = (img.astype(jnp.float32) - 128.0) / 32.0
            for f, k, st in zip(cw, (5, 3, 3), (2, 2, 2)):
                x = nn.relu(nn.Conv(f, (k, k), (st, st))(x))
            return x.reshape((x.shape[0], -1))
    class A(nn.Module):
        @nn.compact
        def __call__(self, img, vec):
            z = ConvTrunk()(img)
            v = nn.relu(nn.Dense(dv)(vec))
            h = nn.relu(nn.Dense(d1)(jnp.concatenate([z, v], -1)))
            h = nn.relu(nn.Dense(d2)(h))
            return nn.Dense(adim)(h), None   # (mean, log_std slot unused)
    return A()

CK = pickle.load(open(sys.argv[1], 'rb'))
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5555
meta = CK['meta']
H, C, K = meta['H'], meta['C'], meta['frame_stack']
VEC_DIM, ACT_DIM = meta['vec_dim'], meta['adim']
IMG_BYTES = H * H * C
actor = build_actor_from_ck(CK)
print(f"[ppo-client] task {CK['task']} image[{H}x{H}x{C}]x{K}stack "
      f"vector[{VEC_DIM}] action[{ACT_DIM}]", flush=True)

for attempt in range(600):
    try:
        sock = socket.create_connection(('127.0.0.1', PORT), timeout=5)
        break
    except OSError:
        time.sleep(2)
else:
    sys.exit('[ppo-client] server never came up')
sock.settimeout(None)


def recv_exact(n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('closed')
        buf += chunk
    return buf


ACT_LOG = os.environ.get('ACT_LOG', '')
act_rows = []
stack = None
n = 0
t0 = time.perf_counter()
while True:
    hdr = recv_exact(1)
    if hdr == b'Q':
        break
    assert hdr == b'O', hdr
    flags, reward = struct.unpack('<Bf', recv_exact(5))
    vec = np.frombuffer(recv_exact(4 * VEC_DIM), '<f4').copy()
    img = np.frombuffer(recv_exact(IMG_BYTES), np.uint8).reshape(H, H, C).copy()
    if flags & 1 or stack is None:                  # is_first: refill the stack
        stack = np.repeat(img, K, axis=-1)
    else:
        stack = np.concatenate([stack[..., C:], img], axis=-1)
    if flags & 2:                                   # is_last: no action consumed
        continue
    mean, _ = actor.apply(CK['actor'], stack[None], vec[None].astype(np.float32))
    a = np.tanh(np.asarray(mean)[0]).astype('<f4')
    if ACT_LOG:
        act_rows.append(a.copy())
    sock.sendall(a.tobytes())
    n += 1
    if n % 200 == 0:
        print(f'[ppo-client] {n} steps, {(time.perf_counter()-t0)/n*1e3:.0f} ms/step',
              flush=True)
if ACT_LOG:
    np.savez(ACT_LOG, actions=np.array(act_rows))
    print(f'[ppo-client] wrote {len(act_rows)} actions -> {ACT_LOG}', flush=True)
print(f'[ppo-client] done after {n} steps', flush=True)
