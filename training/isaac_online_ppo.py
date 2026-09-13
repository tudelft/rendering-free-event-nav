"""Equal-budget baseline arm: PPO trained ONLINE on rendered Isaac+EVIS events
through the TCP bridge (same protocol as isaac_client_ppo.py), from scratch,
for a fixed wall-clock budget. Saves a ckpt compatible with isaac_client_ppo.

Usage: JAX_PLATFORMS=cpu python ppo/isaac_online_ppo.py <out.pkl> [port]
Env: BUDGET_S (default 1800), ROLLOUT (256), LR (3e-4)
"""
import os, pickle, socket, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

OUT = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5771
BUDGET_S = float(os.environ.get('BUDGET_S', 1800))
ROLLOUT = int(os.environ.get('ROLLOUT', 256))
LR = float(os.environ.get('LR', 3e-4))
H, C, K, VEC_DIM, ACT_DIM = 64, 1, 3, 4, 2
GAMMA, LAM, CLIP, ENT, VF, EPOCHS, NMB = 0.99, 0.95, 0.2, 3e-4, 0.5, 4, 4

class ConvTrunk(nn.Module):
    @nn.compact
    def __call__(self, img):
        x = (img.astype(jnp.float32) - 128.0) / 32.0
        for f, k, st in zip((32, 128, 128), (5, 3, 3), (2, 2, 2)):
            x = nn.relu(nn.Conv(f, (k, k), (st, st))(x))
        return x.reshape((x.shape[0], -1))

class Actor(nn.Module):
    @nn.compact
    def __call__(self, img, vec):
        z = ConvTrunk()(img)
        v = nn.relu(nn.Dense(128)(vec))
        h = nn.relu(nn.Dense(128)(jnp.concatenate([z, v], -1)))
        h = nn.relu(nn.Dense(128)(h))
        mean = nn.Dense(ACT_DIM)(h)
        log_std = self.param('log_std', lambda k: jnp.full((ACT_DIM,), -0.5))
        return mean, log_std

class Critic(nn.Module):
    @nn.compact
    def __call__(self, img, vec):
        z = ConvTrunk()(img)
        v = nn.relu(nn.Dense(128)(vec))
        h = nn.relu(nn.Dense(128)(jnp.concatenate([z, v], -1)))
        h = nn.relu(nn.Dense(128)(h))
        return nn.Dense(1)(h)[..., 0]

key = jax.random.PRNGKey(0)
k1, k2, key = jax.random.split(key, 3)
dimg = jnp.zeros((1, H, H, C * K), jnp.uint8); dvec = jnp.zeros((1, VEC_DIM))
actor, critic = Actor(), Critic()
pa = actor.init(k1, dimg, dvec); pc = critic.init(k2, dimg, dvec)
if os.path.exists(OUT):                     # resume: warm-start params from prior checkpoint
    _ck = pickle.load(open(OUT, 'rb'))
    pa = {**pa, 'params': _ck['actor']['params']}
    pc = {**pc, 'params': _ck['critic']['params']}
    _PRIOR = dict(steps=_ck.get('steps', 0), updates=_ck.get('updates', 0),
                  ep_rets=list(_ck.get('ep_rets', [])), wall=_ck.get('wall', 0.0))
    print(f"[online-ppo] RESUMED from {OUT}: {_PRIOR['updates']} updates, "
          f"{_PRIOR['steps']} steps, {_PRIOR['wall']:.0f}s already banked", flush=True)
else:
    _PRIOR = dict(steps=0, updates=0, ep_rets=[], wall=0.0)
txa = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR))
txc = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR))
oa, oc = txa.init(pa), txc.init(pc)

def logp_fn(mean, log_std, u):
    # tanh-gaussian: u is pre-tanh sample
    lp = -0.5 * (((u - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    lp = lp.sum(-1) - jnp.log(1 - jnp.tanh(u) ** 2 + 1e-6).sum(-1)
    return lp

@jax.jit
def act(pa, pc, img, vec, k):
    mean, log_std = actor.apply(pa, img, vec)
    u = mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape)
    return jnp.tanh(u)[0], u[0], logp_fn(mean, log_std, u)[0], critic.apply(pc, img, vec)[0]

@jax.jit
def update(pa, pc, oa, oc, batch, k):
    def loss_a(p):
        mean, log_std = actor.apply(p, batch['img'], batch['vec'])
        lp = logp_fn(mean, log_std, batch['u'])
        ratio = jnp.exp(lp - batch['lp'])
        adv = (batch['adv'] - batch['adv'].mean()) / (batch['adv'].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
        ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        return pg - ENT * ent
    def loss_c(p):
        v = critic.apply(p, batch['img'], batch['vec'])
        return VF * optax.huber_loss(v, batch['ret'], delta=10.0).mean()
    for _ in range(1):
        ga = jax.grad(loss_a)(pa); ua, oa = txa.update(ga, oa); pa = optax.apply_updates(pa, ua)
        gc = jax.grad(loss_c)(pc); uc, oc = txc.update(gc, oc); pc = optax.apply_updates(pc, uc)
    return pa, pc, oa, oc

for attempt in range(600):
    try:
        sock = socket.create_connection(('127.0.0.1', PORT), timeout=5); break
    except OSError:
        time.sleep(2)
else:
    sys.exit('[online-ppo] server never came up')
sock.settimeout(None)

def recv_exact(n):
    buf = b''
    while len(buf) < n:
        c_ = sock.recv(n - len(buf))
        if not c_: raise ConnectionError('closed')
        buf += c_
    return buf

buf = {k: [] for k in ('img', 'vec', 'u', 'lp', 'rew', 'val', 'done')}
stack = None; t0 = time.time(); n = _PRIOR['steps']; nupd = _PRIOR['updates']; ep_rets = list(_PRIOR['ep_rets']); ep_ret = 0.0
print(f'[online-ppo] budget {BUDGET_S:.0f}s rollout {ROLLOUT}', flush=True)
try:
  while time.time() - t0 < BUDGET_S:
    hdr = recv_exact(1)
    if hdr == b'Q': break
    flags, reward = struct.unpack('<Bf', recv_exact(5))
    vec = np.frombuffer(recv_exact(4 * VEC_DIM), '<f4').copy()
    img = np.frombuffer(recv_exact(H * H * C), np.uint8).reshape(H, H, C).copy()
    is_first = bool(flags & 1); is_last = bool(flags & 2)
    if is_first or stack is None:
        stack = np.repeat(img, K, axis=-1)
    else:
        stack = np.concatenate([stack[..., C:], img], axis=-1)
        if buf['rew']: buf['rew'][-1] += 0  # reward arrives with NEXT obs:
    if buf['done'] and len(buf['done']) > len(buf['rew']):
        pass
    # bookkeeping: reward in this message pairs with the PREVIOUS action
    if not is_first and buf['u']:
        buf['rew'].append(reward); buf['done'].append(is_last); ep_ret += reward
    if is_last:
        ep_rets.append(ep_ret); ep_ret = 0.0
        stack = None
        continue
    key, ks = jax.random.split(key)
    a, u, lp, val = act(pa, pc, stack[None], vec[None].astype(np.float32), ks)
    buf['img'].append(stack.copy()); buf['vec'].append(vec.astype(np.float32))
    buf['u'].append(np.asarray(u)); buf['lp'].append(float(lp)); buf['val'].append(float(val))
    sock.sendall(np.tanh(np.asarray(u)).astype('<f4').tobytes())
    n += 1
    m = min(len(buf['rew']), len(buf['u']))
    if m >= ROLLOUT:
        adv = np.zeros(m, np.float32); last = 0.0
        vals = np.array(buf['val'][:m] + [buf['val'][m - 1]], np.float32)
        for i in reversed(range(m)):
            nonterm = 1.0 - float(buf['done'][i])
            delta = buf['rew'][i] + GAMMA * vals[i + 1] * nonterm - vals[i]
            last = delta + GAMMA * LAM * nonterm * last
            adv[i] = last
        ret = adv + vals[:m]
        batch = dict(img=jnp.array(buf['img'][:m]), vec=jnp.array(buf['vec'][:m]),
                     u=jnp.array(buf['u'][:m]), lp=jnp.array(buf['lp'][:m]),
                     adv=jnp.array(adv), ret=jnp.array(ret))
        idx = np.random.permutation(m)
        for _ in range(EPOCHS):
            for mb in np.array_split(idx, NMB):
                sub = {k_: v[mb] for k_, v in batch.items()}
                key, ku = jax.random.split(key)
                pa, pc, oa, oc = update(pa, pc, oa, oc, sub, ku)
        nupd += 1
        if nupd % 20 == 0:            # periodic checkpoint (lesson: SIGINT once cost 13h)
            _tmp = OUT + '.tmp'
            pickle.dump(dict(actor={'params': pa['params']}, critic={'params': pc['params']},
                             task='isaac_online_evis', steps=n, updates=nupd,
                             meta=dict(H=H, C=C, frame_stack=K, vec_dim=VEC_DIM, adim=ACT_DIM),
                             ep_rets=ep_rets, wall=time.time() - t0 + _PRIOR['wall']), open(_tmp, 'wb'))
            os.replace(_tmp, OUT)
        for k_ in buf: buf[k_] = buf[k_][m:] if len(buf[k_]) > m else []
        el = time.time() - t0
        print(f'[online-ppo] upd {nupd} steps {n} elapsed {el:.0f}s '
              f'ep_ret_mean10 {np.mean(ep_rets[-10:]) if ep_rets else 0:.2f} '
              f'eps {len(ep_rets)}', flush=True)
except (ConnectionError, KeyboardInterrupt) as e:
    print(f'[online-ppo] stream ended: {e}', flush=True)

ck = dict(actor={'params': pa['params']}, critic={'params': pc['params']},
          task='isaac_online_evis',
          meta=dict(H=H, C=C, frame_stack=K, vec_dim=VEC_DIM, adim=ACT_DIM),
          steps=n, updates=nupd, ep_rets=ep_rets, wall=time.time() - t0 + _PRIOR['wall'])
pickle.dump(ck, open(OUT, 'wb'))
print(f'[online-ppo] saved {OUT}: {n} env steps, {nupd} updates, '
      f'{len(ep_rets)} episodes in {time.time()-t0:.0f}s', flush=True)
