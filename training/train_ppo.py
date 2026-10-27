"""PPO trainer (PureJaxRL-style, fully jitted rollout+update) for the gx task
family. The env is SkyCourse's own jitted closures, vmapped (env_vec.py), so
the observation/dynamics contract is identical to the DreamerV3 runs.

Usage (the paper run is training/configs/ours.sh):
  python training/train_ppo.py --task yawzoneh32r7w20gxv3 --logdir runs/ppo_gx_v3 \
      --num-envs 512 --total-steps 100e6
Checkpoints: <logdir>/ckpt.pkl (latest, every --eval-every updates) and
<logdir>/BEST.pkl (highest stochastic training-rollout reach so far); scores.jsonl
logs per-update stats. Deterministic evaluation: training/eval_proxy.py.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from env_vec import make_env
from nets import Actor, Critic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='yawzoneh32r7w20gxv3')
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--num-envs', type=int, default=512)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--total-steps', type=float, default=100e6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=5e-3)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--mean-reg', type=float, default=0.0)
    ap.add_argument('--aug', action='store_true',
                    help='RAD/DrQ-style update-time augmentation: +-4px random '
                         'shift (128-padded) + amplitude jitter U(0.7,1.4) on '
                         'the 128-centred deviation')
    ap.add_argument('--rew-scale', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--vmin', type=float, default=None,
                    help='override course speed range (slow-deploy arm)')
    ap.add_argument('--vmax', type=float, default=None)
    ap.add_argument('--yawrate-max', type=float, default=None,
                    help='override yaw-rate cap (rad/s; slow-deploy arm)')
    ap.add_argument('--ceiling', action='store_true',
                    help='sqroom: enable ceiling-line + high-wall patch DR')
    ap.add_argument('--head-pen', action='store_true',
                    help='gated deadzone heading penalty (>90deg off-bearing in clear space)')
    ap.add_argument('--goal-r', type=float, default=None,
                    help='point-goal capture radius override (match to v/yawrate turn radius)')
    ap.add_argument('--props', action='store_true',
                    help='sqroom: lab props lottery (screens, trusses, plants)')
    ap.add_argument('--hold-alt', action='store_true',
                    help='alternate 7/8-tick holds on an r8 task -> exact 13.33 Hz decisions')
    ap.add_argument('--room', type=float, default=None,
                    help='square-room size override (density scaling): x_goal=room, half_w=room/2')
    ap.add_argument('--depth-obs', action='store_true',
                    help='analytic ground-truth depth image obs (baseline arm)')
    ap.add_argument('--render-dr', action='store_true',
                    help='sqroom: per-episode event-RENDERER statistics DR (sim2sim robustness)')
    ap.add_argument('--keep-lo', type=float, default=None,
                    help='sqroom: raise the tree-density keep-fraction lower bound')
    ap.add_argument('--init-from', default=None,
                    help='warm-start actor+critic from a BEST.pkl (width-adaptive)')
    ap.add_argument('--target-kl', type=float, default=0.05)
    ap.add_argument('--minibatches', type=int, default=8)
    ap.add_argument('--frame-stack', type=int, default=3)
    ap.add_argument('--eval-every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    env_reset, env_step, meta = make_env(args.task, vmin=args.vmin, vmax=args.vmax, yawrate_max=args.yawrate_max, ceiling=args.ceiling, head_pen=args.head_pen, goal_r=args.goal_r, props=args.props, render_dr=args.render_dr, depth_obs=args.depth_obs, keep_lo=args.keep_lo, room=args.room, hold_alt=args.hold_alt)
    B, T = args.num_envs, args.rollout
    init_ck = None
    if args.init_from:
        import pickle as _pkl
        init_ck = _pkl.load(open(args.init_from, 'rb'))
        # width-adaptive rebuild (nets.py trunk widened after the h64 arm): read every layer
        # width from the saved params so warm-starting old checkpoints keeps working.
        import flax.linen as _nn
        def _widths(P):
            cw = [P['ConvTrunk_0'][f'Conv_{i}']['kernel'].shape[-1] for i in range(3)]
            ds = []
            j = 0
            while f'Dense_{j}' in P:
                ds.append(P[f'Dense_{j}']['kernel'].shape[-1]); j += 1
            return cw, ds
        acw, ads = _widths(init_ck['actor']['params'])
        ccw, cds = _widths(init_ck['critic']['params'])
        def _mk_trunk(cw):
            class ConvTrunk(_nn.Module):
                @_nn.compact
                def __call__(self, img):
                    x = (img.astype(jnp.float32) - 128.0) / 32.0
                    for f, k, st in zip(cw, (5, 3, 3), (2, 2, 2)):
                        x = _nn.relu(_nn.Conv(f, (k, k), (st, st))(x))
                    return x.reshape((x.shape[0], -1))
            return ConvTrunk
        ATr, CTr = _mk_trunk(acw), _mk_trunk(ccw)
        class ActorW(_nn.Module):
            @_nn.compact
            def __call__(self, img, vec):
                z = ATr()(img)
                v = _nn.relu(_nn.Dense(ads[0])(vec))
                h = _nn.relu(_nn.Dense(ads[1])(jnp.concatenate([z, v], -1)))
                h = _nn.relu(_nn.Dense(ads[2])(h))
                mean = _nn.Dense(ads[3])(h)
                log_std = self.param('log_std', _nn.initializers.constant(-0.5), (ads[3],))
                log_std = jnp.clip(log_std, -2.5, 0.5)
                return mean, jnp.broadcast_to(log_std, mean.shape)
        class CriticW(_nn.Module):
            @_nn.compact
            def __call__(self, img, vec, priv):
                z = CTr()(img)
                v = _nn.relu(_nn.Dense(cds[0])(jnp.concatenate([vec, priv], -1)))
                h = _nn.relu(_nn.Dense(cds[1])(jnp.concatenate([z, v], -1)))
                h = _nn.relu(_nn.Dense(cds[2])(h))
                return _nn.Dense(1)(h)[..., 0]
        actor, critic = ActorW(), CriticW()
    else:
        actor, critic = Actor(adim=meta['adim']), Critic()

    key = jax.random.PRNGKey(args.seed)
    key, ka, kc, ke = jax.random.split(key, 4)
    dummy_img = jnp.zeros((1, meta['H'], meta['H'],
                           meta['C'] * meta['frame_stack']), jnp.uint8)
    dummy_vec = jnp.zeros((1, meta['vec_dim']), jnp.float32)
    dummy_priv = jnp.zeros((1, meta['priv_dim']), jnp.float32)
    pa = actor.init(ka, dummy_img, dummy_vec)
    pc = critic.init(kc, dummy_img, dummy_vec, dummy_priv)
    if init_ck is not None:
        pa, pc = init_ck['actor'], init_ck['critic']
        print(f'[warm-start] actor+critic from {args.init_from} '
              f'(reach {init_ck.get("reach")}, update {init_ck.get("update")})', flush=True)
    tx = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr)),
        max_consecutive_errors=1000)
    ts_a = TrainState.create(apply_fn=actor.apply, params=pa, tx=tx)
    ts_c = TrainState.create(apply_fn=critic.apply, params=pc, tx=tx)

    vreset = jax.vmap(env_reset)
    vstep = jax.vmap(env_step)

    def logp_fn(mean, log_std, u):
        """log-prob of a = tanh(u), u ~ N(mean, std): gaussian logp minus the
        tanh jacobian. `u` is the PRE-squash sample (stored in the batch)."""
        var = jnp.exp(2 * log_std)
        lp = (-0.5 * ((u - mean) ** 2 / var + 2 * log_std
                      + jnp.log(2 * jnp.pi))).sum(-1)
        return lp - jnp.log(1.0 - jnp.tanh(u) ** 2 + 1e-6).sum(-1)

    @jax.jit
    def rollout(ts_a, ts_c, state, stack, obs, key):
        def one(carry, _):
            state, stack, obs, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            mean, log_std = ts_a.apply_fn(ts_a.params, obs['image'], obs['vector'])
            u = mean + jnp.exp(log_std) * jax.random.normal(k1, mean.shape)
            a = jnp.tanh(u)                           # bounded by construction
            lp = logp_fn(mean, log_std, u)
            mm = jnp.abs(mean).mean()
            val = ts_c.apply_fn(ts_c.params, obs['image'], obs['vector'], obs['priv'])
            ks = jax.random.split(k2, obs['image'].shape[0])
            state2, stack2, obs2, r, done, info = vstep(state, stack, a, ks)
            tr = dict(obs=obs, act=u, logp=lp, val=val, mm=mm,
                      rew=r * args.rew_scale, done=done,
                      reached=info['reached'], terminal=info['terminal'])
            return (state2, stack2, obs2, key), tr
        (state, stack, obs, key), tr = jax.lax.scan(
            one, (state, stack, obs, key), None, T)
        last_val = ts_c.apply_fn(ts_c.params, obs['image'], obs['vector'], obs['priv'])
        return state, stack, obs, key, tr, last_val

    @jax.jit
    def gae(tr, last_val, gamma, lam):
        def back(carry, x):
            adv_next, v_next = carry
            r, v, done, term = x
            # timeout (done & ~terminal) bootstraps; terminal cuts
            nonterm = 1.0 - term.astype(jnp.float32)
            cont = 1.0 - done.astype(jnp.float32)
            delta = r + gamma * v_next * nonterm - v
            adv = delta + gamma * lam * adv_next * cont
            return (adv, v), adv
        (_, _), advs = jax.lax.scan(
            back, (jnp.zeros_like(last_val), last_val),
            (tr['rew'], tr['val'], tr['done'], tr['terminal']), reverse=True)
        return advs, advs + tr['val']

    AUG_PAD = 4

    def augment(key, img):
        """[N,H,W,C] uint8 -> shifted + amplitude-jittered, still uint8."""
        k1, k2, k3 = jax.random.split(key, 3)
        N, H_, W_, C_ = img.shape
        dev = img.astype(jnp.float32) - 128.0
        g = jax.random.uniform(k1, (N, 1, 1, 1), minval=0.7, maxval=1.4)
        dev = dev * g
        padded = jnp.pad(dev, ((0, 0), (AUG_PAD, AUG_PAD), (AUG_PAD, AUG_PAD),
                               (0, 0)))
        oy = jax.random.randint(k2, (N,), 0, 2 * AUG_PAD + 1)
        ox = jax.random.randint(k3, (N,), 0, 2 * AUG_PAD + 1)
        crop = lambda p, y, x: jax.lax.dynamic_slice(p, (y, x, 0), (H_, W_, C_))
        dev = jax.vmap(crop)(padded, oy, ox)
        return jnp.clip(dev + 128.0, 0, 255).astype(jnp.uint8)

    @jax.jit
    def update(ts_a, ts_c, batch, key):
        def loss_a(pa, mb):
            mean, log_std = actor.apply(pa, mb['img'], mb['vec'])
            lp = logp_fn(mean, log_std, mb['act'])
            ratio = jnp.exp(jnp.clip(lp - mb['logp'], -10.0, 10.0))
            adv = (mb['adv'] - mb['adv'].mean()) / (mb['adv'].std() + 1e-8)
            l1 = ratio * adv
            l2 = jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv
            ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
            # restoring force for the clipped-Gaussian rail runaway: penalize
            # pre-clip mean OUTSIDE the action box (free inside it)
            reg = (mean ** 2).mean()          # mild pre-tanh center pull (off by default)
            return (-(jnp.minimum(l1, l2).mean()) - args.ent * ent
                    + args.mean_reg * reg)

        def loss_c(pc, mb):
            v = critic.apply(pc, mb['img'], mb['vec'], mb['priv'])
            return args.vf * optax.huber_loss(v, mb['ret'], delta=10.0).mean()

        N = B * T
        flat = dict(img=batch['obs']['image'].reshape((N,) + batch['obs']['image'].shape[2:]),
                    vec=batch['obs']['vector'].reshape(N, -1),
                    priv=batch['obs']['priv'].reshape(N, -1),
                    act=batch['act'].reshape(N, -1), logp=batch['logp'].reshape(N),
                    adv=batch['adv'].reshape(N), ret=batch['ret'].reshape(N))

        def epoch(carry, k):
            ts_a, ts_c = carry
            k, k_aug = jax.random.split(k)
            if args.aug:
                flat_e = dict(flat, img=augment(k_aug, flat['img']))
            else:
                flat_e = flat
            perm = jax.random.permutation(k, N)
            def mb_step(carry, idx):
                ts_a, ts_c, halted = carry
                mb = jax.tree_util.tree_map(lambda x: x[idx], flat_e)
                mean, log_std = actor.apply(ts_a.params, mb['img'], mb['vec'])
                lr_ = logp_fn(mean, log_std, mb['act']) - mb['logp']
                kl = (jnp.exp(lr_) - 1.0 - lr_).mean()      # proper approx-KL
                halted = halted | (kl > args.target_kl)
                ga = jax.grad(loss_a)(ts_a.params, mb)
                ga = jax.tree_util.tree_map(
                    lambda g: jnp.where(halted, jnp.zeros_like(g), g), ga)
                gc = jax.grad(loss_c)(ts_c.params, mb)
                return (ts_a.apply_gradients(grads=ga),
                        ts_c.apply_gradients(grads=gc), halted), kl
            idxs = perm.reshape(args.minibatches, -1)
            (ts_a, ts_c, _), kls = jax.lax.scan(
                mb_step, (ts_a, ts_c, jnp.bool_(False)), idxs)
            return (ts_a, ts_c), kls.mean()
        (ts_a, ts_c), kls = jax.lax.scan(
            epoch, (ts_a, ts_c), jax.random.split(key, args.epochs))
        return ts_a, ts_c, kls.mean()

    # (deterministic eval is training/eval_proxy.py; training logs stochastic reach)

    key, kr = jax.random.split(key)
    state, stack, obs = vreset(jax.random.split(kr, B))
    steps_per_update = B * T
    n_updates = int(args.total_steps // steps_per_update)
    log = open(os.path.join(args.logdir, 'scores.jsonl'), 'a')
    best_reach = -1.0
    t0 = time.time()
    for u in range(n_updates):
        key, k1, k2 = jax.random.split(key, 3)
        state, stack, obs, key, tr, last_val = rollout(
            ts_a, ts_c, state, stack, obs, k1)
        adv, ret = gae(tr, last_val, args.gamma, args.lam)
        ts_a, ts_c, kl = update(ts_a, ts_c,
                            dict(obs=tr['obs'], act=tr['act'], logp=tr['logp'],
                                 adv=adv, ret=ret), k2)
        ends = float(tr['done'].sum())
        reach = float(tr['reached'].sum()) / max(ends, 1)
        rew = float(tr['rew'].sum(0).mean()) / args.rew_scale
        fps = steps_per_update / max(time.time() - t0, 1e-9)
        t0 = time.time()
        mmag = float(tr['mm'].mean())
        ls = float(np.asarray(
            jnp.clip(ts_a.params['params']['log_std'], -2.5, 0.5)).mean())
        rec = dict(update=u, env_steps=(u + 1) * steps_per_update, kl=float(kl),
                   log_std=ls, mean_mag=mmag,
                   reach=reach, episodes=ends, mean_rollout_reward=rew,
                   sps=round(fps))
        log.write(json.dumps(rec) + '\n'); log.flush()
        if u % 5 == 0:
            print(f"[{u}/{n_updates}] steps {(u+1)*steps_per_update/1e6:.1f}M "
                  f"reach {reach:.2f} rew {rew:.1f} sps {fps:.0f}", flush=True)
        if u % args.eval_every == 0 or u == n_updates - 1:
            with open(os.path.join(args.logdir, 'ckpt.pkl'), 'wb') as f:
                pickle.dump(dict(actor=jax.device_get(ts_a.params),
                                 critic=jax.device_get(ts_c.params),
                                 meta=meta, task=args.task), f)
            if reach > best_reach:
                best_reach = reach
                with open(os.path.join(args.logdir, 'BEST.pkl'), 'wb') as f:
                    pickle.dump(dict(actor=jax.device_get(ts_a.params),
                                     critic=jax.device_get(ts_c.params),
                                     meta=meta, task=args.task,
                                     reach=reach, update=u), f)
    print('done; best train reach', best_reach)


if __name__ == '__main__':
    main()
