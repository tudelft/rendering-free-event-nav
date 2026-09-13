"""Vmappable pure-JAX env for PPO, contract-identical to SkyCourse's steer/gx
tasks BY CONSTRUCTION: it reuses the SkyCourse instance's own jitted closures
(_sample_st / _reset_fn / _step_st / _obs_st) rather than reimplementing them.
Everything skycourse does per episode is reproduced: scene sampling (incl. gx
enrichment + DR), yaw-DR reset, act-repeat with reward summing and early
termination, timeout at env.max_steps (already in policy steps), and the gx
uint8 mapping clip(net*GENX_CONTRAST + 128).

Auto-reset happens inside step (standard PPO): on done, the returned obs is
the FIRST obs of a fresh episode and `done=True` marks the boundary.
"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proxy'))

import jax
import jax.numpy as jnp
import numpy as np
from embodied.envs.skycourse import SkyCourse
import quad_course_env as QC


def make_env(task, frame_stack=3, vmin=None, vmax=None, yawrate_max=None, ceiling=False, head_pen=False, goal_r=None, props=False, render_dr=False, keep_lo=None, depth_obs=False, room=None, hold_alt=False):
    env = SkyCourse(task=task, seed=0)
    # speed-range override (2026-08-25, slow-PPO arm): mutate BEFORE closure extraction /
    # first trace so the jitted step captures the new range. Also v0 (reset speed).
    if vmax is not None:
        import jax.numpy as _jnp
        old_vmax = float(env.course['vmax'])
        env.course['vmax'] = _jnp.float32(vmax)
        # keep the time-slack RATIO fixed: a slower course needs proportionally more ticks
        # or it becomes timeout-bound (the eval-timeout lesson, in training form)
        env.max_steps = int(env.max_steps * old_vmax / vmax)
    if vmin is not None:
        import jax.numpy as _jnp
        env.course['vmin'] = _jnp.float32(vmin)
        env.course['v0'] = _jnp.float32(vmin)
    if ceiling:
        env.course['ceiling'] = True
    if head_pen:
        env.course['head_pen'] = True
    if goal_r is not None:
        env.course['goal_r'] = float(goal_r)
    if props:
        env.course['props'] = True
    if render_dr:
        env.course['render_dr'] = True
    if room is not None:
        import jax.numpy as _jnp
        # square-room size override BEFORE closures: all sampler/reset/reward
        # ranges derive from course['x_goal']/course['half_w']
        env.course['x_goal'] = _jnp.float32(room)
        env.course['half_w'] = _jnp.float32(room / 2.0)
    if keep_lo is not None:
        env.course['keep_lo'] = float(keep_lo)
    if yawrate_max is not None:
        import jax.numpy as _jnp
        # proprio yawrate slot normalizes by course['yawrate_max'] -> contract stays consistent
        env.course['yawrate_max'] = _jnp.float32(yawrate_max)
    assert env.steer, 'PPO env supports the steer/forest/zone task family'
    sample, reset_fn = env._sample_st, env._reset_fn
    step_fn, obs_fn = env._step_st, env._obs_st
    yawdr, rep = env.yawdr, env.act_repeat
    max_steps = env.max_steps                     # already divided by act_repeat
    H, genx = env.H, env.genx
    C = 1 if (genx or depth_obs) else 2
    adim = int(env.act_space['action'].shape[0])


    # ---- analytic ground-truth depth observation (baseline arm) -----------
    # distance_to_image_plane convention (matches the Isaac annotator): depth =
    # ray parameter t for unnormalized camera dir (1, yc, zc). Inverse-depth
    # mapping to uint8 so nearby structure gets resolution.
    _dx0 = float(env.course['x0']); _dxg = float(env.course['x_goal'])
    _dhw = float(env.course['half_w']); _dzt = float(env.course.get('z_top', 3.0))
    _dnear, _dfar = 0.3, 25.0

    def _depth_img(cs, oxy, orr):
        f = QC.focal_from_fov(90.0, H) if hasattr(QC, 'focal_from_fov') else H / 2.0
        cx = (H - 1) / 2.0
        u = (jnp.arange(H) - cx) / f
        yc = -u[None, :]                       # +y = drone-left, image col right
        zc = -u[:, None] * jnp.ones((1, H))    # image row down, +z up
        psi = 2.0 * jnp.arctan2(cs.q[3], cs.q[0])
        cp, sp = jnp.cos(psi), jnp.sin(psi)
        dx = cp - sp * yc; dy = sp + cp * yc; dz = zc      # (H,H)
        px, py, pz = cs.p[0], cs.p[1], cs.p[2]
        t = jnp.full((H, H), _dfar)
        # vertical cylinders (trees + prop collision slots), scanned in chunks
        # of 8 so the per-env intermediate stays (8,H,H) not (n_obs,H,H) (OOM)
        a2 = dx * dx + dy * dy                                          # (H,H)
        CH = 8
        n_pad = (-oxy.shape[0]) % CH
        oxp = jnp.concatenate([oxy, jnp.zeros((n_pad, 2))], 0).reshape(-1, CH, 2)
        orp = jnp.concatenate([orr, jnp.zeros((n_pad,))], 0).reshape(-1, CH)

        def chunk_min(tmin, ch):
            cxy, cr = ch
            ox = cxy[:, 0][:, None, None] - px; oy = cxy[:, 1][:, None, None] - py
            b = -2.0 * (ox * dx[None] + oy * dy[None])
            cq = ox * ox + oy * oy - (cr[:, None, None]) ** 2
            disc = b * b - 4 * a2[None] * cq
            tq = (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2 * a2[None] + 1e-12)
            zq = pz + dz[None] * tq
            ok = (disc > 0) & (tq > 1e-3) & (zq > 0.0) & (zq < _dzt) & (cr[:, None, None] > 1e-4)
            return jnp.minimum(tmin, jnp.where(ok, tq, _dfar).min(0)), None

        t, _ = jax.lax.scan(chunk_min, t, (oxp, orp))
        # room walls x=x0, x=xg, y=+-hw and floor/ceiling
        for tw in ((_dx0 - px) / jnp.where(jnp.abs(dx) > 1e-6, dx, 1e-6),
                   (_dxg - px) / jnp.where(jnp.abs(dx) > 1e-6, dx, 1e-6),
                   (_dhw - py) / jnp.where(jnp.abs(dy) > 1e-6, dy, 1e-6),
                   (-_dhw - py) / jnp.where(jnp.abs(dy) > 1e-6, dy, 1e-6),
                   (0.0 - pz) / jnp.where(jnp.abs(dz) > 1e-6, dz, 1e-6),
                   (_dzt - pz) / jnp.where(jnp.abs(dz) > 1e-6, dz, 1e-6)):
            t = jnp.minimum(t, jnp.where(tw > 1e-3, tw, _dfar))
        inv = (1.0 / jnp.clip(t, _dnear, _dfar) - 1.0 / _dfar) / (1.0 / _dnear - 1.0 / _dfar)
        # 4x gain spreads the useful depth range across uint8 (trunk-normalized
        # mean 0.4, std 2.2). NB: flat-training failures with this obs were a
        # SEED pathology (default seed collapses, seed 3 reaches 90% at 2M),
        # not an encoding problem -- keep apply_if_finite in mind.
        return jnp.clip(1020.0 * inv[..., None], 0.0, 255.0).astype(jnp.uint8)

    def _img(frame):
        if genx:
            return jnp.clip(frame * QC.GENX_CONTRAST + QC.GENX_BRIGHTNESS,
                            0, 255).astype(jnp.uint8)
        return jnp.clip(frame * 32.0, 0, 255).astype(jnp.uint8)

    def _obs(state, key):
        frame, proprio, priv = obs_fn(state['cs'], key, state['oxy'],
                                      state['orr'], state['pts'],
                                      state['valid'], state['drp'])
        if depth_obs:
            return (_depth_img(state['cs'], state['oxy'], state['orr']),
                    proprio.astype(jnp.float32), priv.astype(jnp.float32))
        return _img(frame), proprio.astype(jnp.float32), priv.astype(jnp.float32)

    def env_reset(key):
        ko, kr, kobs = jax.random.split(key, 3)
        oxy, orr, pts, valid, wall, drp = sample(ko)
        cs = reset_fn(kr, oxy) if yawdr else reset_fn(kr)
        state = dict(cs=cs, oxy=oxy, orr=orr, pts=pts, valid=valid, wall=wall,
                     drp=drp, t=jnp.int32(0))
        img, proprio, priv = _obs(state, kobs)
        stack = jnp.repeat(img, frame_stack, axis=-1)      # fill with first frame
        return state, stack, {'image': stack, 'vector': proprio, 'priv': priv}

    def env_step(state, stack, action, key):
        ks = jax.random.split(key, rep + 2)

        # hold_alt: alternate 7/8 effective ticks per decision (skip the last
        # tick on even decisions) -> exact 75 ms mean period = 13.33 Hz on the
        # 100 Hz base without touching dt, windows, or physics. Requires an r8
        # task so rep == 8.
        def tick(carry, kj):
            k, j = kj
            cs, r_acc, done = carry
            cs2, r, d, info = step_fn(cs, action, k, state['oxy'],
                                      state['orr'], state['wall'])
            skip = jnp.bool_(False)
            if hold_alt:
                skip = (j == rep - 1) & (state['t'] % 2 == 0)
            frz = done | skip
            # freeze after first done (skycourse breaks out of the repeat loop)
            cs2 = jax.tree_util.tree_map(
                lambda a, b: jnp.where(frz, a, b), cs, cs2)
            r_acc = r_acc + jnp.where(frz, 0.0, r)
            return (cs2, r_acc, done | (d & ~skip)), info

        (cs, r_sum, done), infos = jax.lax.scan(
            tick, (state['cs'], jnp.float32(0.0), jnp.bool_(False)),
            (ks[:rep], jnp.arange(rep)))
        t = state['t'] + 1
        timeout = t >= max_steps
        is_last = done | timeout
        reached = infos['passed'].any()
        state2 = dict(state, cs=cs, t=t)
        img, proprio, priv = _obs(state2, ks[rep])
        stack2 = jnp.concatenate([stack[..., C:], img], axis=-1)

        # auto-reset on episode end
        rstate, rstack, robs = env_reset(ks[rep + 1])
        state3 = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_last, a, b), rstate, state2)
        stack3 = jnp.where(is_last, rstack, stack2)
        obs = {'image': jnp.where(is_last, robs['image'], stack2),
               'vector': jnp.where(is_last, robs['vector'], proprio),
               'priv': jnp.where(is_last, robs['priv'], priv)}
        return state3, stack3, obs, r_sum, is_last, {
            'reached': reached & is_last, 'terminal': done, 'timeout': timeout}

    meta = dict(H=H, C=C, frame_stack=frame_stack, adim=adim,
                max_steps=max_steps, genx=genx,
                vec_dim=int(env.obs_space['vector'].shape[0]),
                priv_dim=int(env.obs_space['priv'].shape[0]))
    return env_reset, env_step, meta
