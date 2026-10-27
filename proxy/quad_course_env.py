"""Multi-obstacle IMAV Mission-1 course env (JAX) for SkyDreamer-CF.

Reuses the validated CFB dynamics/CTBR from quad_ctbr_env; adds a flight-plan
over TYPED obstacles with per-type traversal:
  THROUGH  window / tube   — cross the opening plane within the lateral+vertical opening
  OVER     red bar         — cross the bar plane while ABOVE it (z > thr)
  UNDER    blue bar         — cross the bar plane while BELOW it (z < thr)
  LAND     marker           — reach the marker position (within radius thr), slow

Reward = SkyDreamer distance-progress toward the current target's goal
         - body-rate penalty + 30 * pass-bonus. Passing advances the target.

Observation = SIMULATED detector output (course_sim): per-class heatmap [8,H,W]
+ per-class metric vector [40] + proprio, with optional detector domain rand.
"""
from typing import NamedTuple
import numpy as np          # host-side only (static shapes/counts computed at course build time)
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quad_ctbr_env as E
import course_sim as CS
import event_proxy as EP

THROUGH, OVER, UNDER, LAND = 0, 1, 2, 3


class CState(NamedTuple):
    p: jax.Array
    v: jax.Array
    q: jax.Array
    w: jax.Array
    Om: jax.Array
    prev_dist: jax.Array   # scalar, to current target's goal
    target: jax.Array      # scalar int32


LANE_Y = 3.5   # centre lane of the 14x7 arena (matches robosim build_course.py)
START = (1.5, LANE_Y, 1.2)   # near the take-off pad


def make_course(n_targets=7, red_h=1.6, blue_h=0.8, gb=0.6, mission2=False):
    """FAITHFUL IMAV course (mirrors robosim/imav2026_course/build_course.py).
    Two separate sets:
      * DETECTIONS (det_cls, det_corners): everything the camera sees.
      * TRAVERSAL targets (tt/pc/ph/thr/normal): the flight plan. Each target carries a
        plane NORMAL — +x for the whole M1 fly-through, but the M2 exit window is crossed
        flying -x, so its normal is -x (crossing detection & goal_beyond both use it).
    Mission 1: window panel -> over red bar -> under 2 blue bars -> 2 tubes -> land.
    mission2=True: drop the landing, append the M2 DARK ROOM (rulebook §4.4.3, Table 3
    scores Fin+Fout): fly IN the red window (2 pts) -> deep inside (inspect) -> turn ~180
    -> fly OUT the blue window (1 pt). Real 2.5 m cube, front face at x=11.5."""
    Y = LANE_Y
    NX = (1.0, 0.0, 0.0)     # +x plane normal (whole M1 fly-through)
    # -- detections: (class_id, box_center, box_half) --
    dets = [
        (5, (1.5, Y, 0.06),  (0.25, 0.25, 0.02)),   # marker_takeoff pad
        (0, (4.0, 3.15, 1.65), (0.03, 0.25, 0.25)), # window_blue opening (0.50 m)
        (1, (4.0, 3.90, 1.65), (0.03, 0.20, 0.20)), # window_red opening  (0.40 m)
        (2, (6.0, Y, red_h),  (0.03, 0.75, 0.10)),  # bar_red (over) — bbox tall enough to detect
        (3, (8.0, Y, blue_h), (0.03, 0.75, 0.10)),  # bar_blue #1 (under)
        (3, (8.6, Y, blue_h), (0.03, 0.75, 0.10)),  # bar_blue #2 (under)
        (4, (10.3, Y, 0.9),   (0.05, 0.50, 0.60)),  # tube_gate #1
        (4, (10.9, Y, 0.9),   (0.05, 0.50, 0.60)),  # tube_gate #2
        (6, (12.8, Y, 0.06),  (0.25, 0.25, 0.10)),  # marker_landing pad
    ]
    # -- M1 traversal targets: (tt, pass_center, pass_half(lat,vert), thr, normal) --
    m1 = [
        (THROUGH, (4.0, 3.15, 1.65), (0.25, 0.25), 0.0,    NX),  # through blue window
        (OVER,    (6.0, Y, red_h),   (0.75, 1.0),  red_h,  NX),  # over red bar
        (UNDER,   (8.0, Y, blue_h),  (0.75, 1.0),  blue_h, NX),  # under blue bar #1
        (UNDER,   (8.6, Y, blue_h),  (0.75, 1.0),  blue_h, NX),  # under blue bar #2
        (THROUGH, (10.3, Y, 0.9),    (0.45, 0.60), 0.0,    NX),  # through tube #1
        (THROUGH, (10.9, Y, 0.9),    (0.45, 0.60), 0.0,    NX),  # through tube #2
    ]
    if mission2:
        XF = 11.5                            # dark-room front (windowed) face
        # THE DARK ROOM IS EMPTY. Drop the M1 landing pad: it sits at x=12.8, i.e. INSIDE the
        # room volume (x 11.5..14.0) — physically impossible, and it gave the policy a bright
        # phantom landmark for "I'm deep in the room" that will NOT exist on hardware
        # (a sim2real leak). Mission 2 is: fly the course -> in one window -> turn -> out the other.
        dets = [d for d in dets if d[0] != 6]           # remove marker_landing
        # dark-room windows (both on the +x-facing front wall, ~1 m apart in y)
        dets += [
            (1, (XF, 4.0, 1.35), (0.03, 0.20, 0.20)),   # red entry window (0.40 m)
            (0, (XF, 3.0, 1.35), (0.03, 0.25, 0.25)),   # blue exit window (0.50 m)
        ]
        targets = m1 + [
            (THROUGH, (XF,   4.0, 1.35), (0.18, 0.18), 0.0, NX),            # fly IN red (+x)
            (THROUGH, (13.0, 3.5, 1.35), (1.2, 1.2),   0.0, NX),           # inside waypoint (12.2 crammed enter/exit -> broke entering; 13.3 sprint-tumbled; 13.0 middle)
            (THROUGH, (XF,   3.0, 1.35), (0.22, 0.22), 0.0, (-1., 0., 0.)),# fly OUT blue (-x)
        ]
    else:
        landing = [(LAND, (12.8, Y, 0.6), (0.30, 0.30), 0.4, NX)]
        targets = (m1 + landing)[:n_targets]

    det_cls = jnp.array([d[0] for d in dets], dtype=jnp.int32)
    det_corners = jax.vmap(CS.box_corners)(
        jnp.array([d[1] for d in dets], float), jnp.array([d[2] for d in dets], float))
    tt = jnp.array([t[0] for t in targets], dtype=jnp.int32)
    pc = jnp.array([t[1] for t in targets], float)
    ph = jnp.array([t[2] for t in targets], float)
    thr = jnp.array([t[3] for t in targets], float)
    pn = jnp.array([t[4] for t in targets], float)      # PER-TARGET plane normal
    return dict(det_cls=det_cls, det_corners=det_corners,
                tt=tt, pc=pc, pn=pn, ph=ph, thr=thr, gb=jnp.float32(gb))


def _goal_of(course, t):
    pc, pn, gb, tt = course['pc'][t], course['pn'][t], course['gb'], course['tt'][t]
    return jnp.where(tt == LAND, pc, pc + gb * pn)


def reset(key, pm: E.Params, course):
    p = jax.random.uniform(key, (3,), minval=-0.25, maxval=0.25) + jnp.array(START)
    Om_hover = E._thrust_to_Om(pm.mass * pm.g / 4.0, pm) * jnp.ones(4)
    target = jnp.array(0, dtype=jnp.int32)
    dist = jnp.linalg.norm(p - _goal_of(course, target))
    return CState(p=p, v=jnp.zeros(3), q=jnp.array([1.0, 0.0, 0.0, 0.0]),
                  w=jnp.zeros(3), Om=Om_hover, prev_dist=dist, target=target)


def reset_at(key, pm: E.Params, course, p0, v0, target_idx):
    """Reset DIRECTLY at a given target — used by the `landonly` pre-training stage.
    The full course teaches only FAST fly-through (~2 m/s); the landing is the first task
    needing slow, stable flight, and the drone tumbles in the final 1.9 m (tube exit -> pad).
    Starting here lets it practise ONLY that segment in short episodes (fast iteration)."""
    p = jax.random.uniform(key, (3,), minval=-0.25, maxval=0.25) + jnp.array(p0)
    Om_hover = E._thrust_to_Om(pm.mass * pm.g / 4.0, pm) * jnp.ones(4)
    target = jnp.array(target_idx, dtype=jnp.int32)
    dist = jnp.linalg.norm(p - _goal_of(course, target))
    return CState(p=p, v=jnp.array(v0, float), q=jnp.array([1.0, 0.0, 0.0, 0.0]),
                  w=jnp.zeros(3), Om=Om_hover, prev_dist=dist, target=target)


def _reward(s_prev: CState, core: E.State, pm, course, k_prog, k_alt=2.0,
            v_cap=2.0, k_speed=1.0, k_land=4.0, k_w=0.0, k_dw=0.0, k_tilt=0.0, v_land=1.2):
    t = s_prev.target
    pc = course['pc'][t]; n = course['pn'][t]; ph = course['ph'][t]
    thr = course['thr'][t]; tt = course['tt'][t]
    goal = _goal_of(course, t)
    dist = jnp.linalg.norm(core.p - goal)
    r_prog = s_prev.prev_dist - dist                       # bounded distance progress
    r_rate = (jnp.exp(jnp.minimum(jnp.sum(jnp.abs(core.w)), 17.0)) - 1.0) \
        / (2.0 * (1.0 / pm.dt) * 1e5)
    sd_prev = jnp.dot(s_prev.p - pc, n)
    sd = jnp.dot(core.p - pc, n)
    off = jnp.abs(core.p - pc)
    lat_ok = off[1] < ph[0]
    crossed = (sd_prev < 0) & (sd >= 0)
    through = crossed & lat_ok & (off[2] < ph[1])
    over = crossed & lat_ok & (core.p[2] > thr)
    under = crossed & lat_ok & (core.p[2] < thr)
    land = (dist < thr) & (jnp.linalg.norm(core.v) < v_land)
    passed = jnp.where(tt == THROUGH, through,
                       jnp.where(tt == OVER, over,
                                 jnp.where(tt == UNDER, under, land)))

    # --- altitude shaping for bar obstacles (POTENTIAL-BASED, so no reward hacking) ---
    # As the drone nears a bar plane, reward being on the correct side of its threshold:
    #   OVER  -> reward z ABOVE thr;  UNDER -> reward z BELOW thr.
    # Phi weighted by horizontal proximity `near` (ramps up over ~1.8 m before the bar).
    # Reward = Phi(new) - Phi(old): telescoping => bounded total, loitering earns ~0.
    # This gives the over->under transition a descent gradient it otherwise lacks.
    def _alt_phi(p):
        dx = pc[0] - p[0]                                    # >0 while approaching the bar
        near = jnp.exp(-(jnp.maximum(dx, 0.0) / 1.8) ** 2)
        sign = jnp.where(tt == OVER, 1.0, -1.0)             # OVER: high good; UNDER: low good
        return k_alt * near * jnp.tanh(sign * (p[2] - thr) / 0.5)
    is_bar = (tt == OVER) | (tt == UNDER)
    r_alt = jnp.where(is_bar, _alt_phi(core.p) - _alt_phi(s_prev.p), 0.0)

    # --- speed cap (ENTIRE flight): soft-penalize ground speed above v_cap m/s ---
    # Linear penalty above the cap; zero when already slow. Keeps CTBR dynamics
    # physical (no velocity clamp) while pushing the policy to fly <= v_cap.
    speed = jnp.linalg.norm(core.v)
    # LANDING APPROACH FUNNEL: near the LAND target, shrink the cap with distance so the
    # drone bleeds speed gradually instead of braking hard at the pad (a late brake needs
    # ~15 m/s^2 => ~58 deg tilt => trips the 60 deg crash limit).
    # CRITICAL: the funnel must be CONTINUOUS where LAND activates. LAND switches on right
    # after tube#2 (x=10.9), i.e. d~1.9 m. A cap below cruise speed there is a STEP change ->
    # the policy panic-brakes, over-pitches and crashes 1.8 m short (observed with 0.4+0.5d,
    # which demanded 1.35 m/s at activation). So anchor cap(1.9)=2.0 (== v_cap, no jump) and
    # taper to cap(0.4)=0.6 (== the landing |v| gate): only ~1.2 m/s^2 => ~7 deg tilt.
    v_cap_eff = jnp.where(tt == LAND,
                          jnp.minimum(v_cap, 1.0 + 0.5 * jnp.linalg.norm(core.p - pc)),
                          v_cap)
    r_speed = -k_speed * jnp.maximum(speed - v_cap_eff, 0.0)

    # --- landing-approach shaping: DECELERATE as you near the LAND marker (potential) ---
    # The landing needs (dist<thr) AND (|v|<0.6) simultaneously; nothing rewarded slowing,
    # so the policy coasted over the pad at ~2 m/s and crashed. Phi = -k_land*near*|v|
    # (near ramps up within ~1.5 m of the marker): slowing while close -> Phi rises ->
    # positive reward; approaching fast -> negative. Telescoping => no loiter farming.
    def _land_phi(p, v):
        d = jnp.linalg.norm(p - pc)
        near = jnp.exp(-(d / 3.0) ** 2)          # wide zone: start braking ~3 m out
        return -k_land * near * jnp.linalg.norm(v)
    r_land = jnp.where(tt == LAND,
                       _land_phi(core.p, core.v) - _land_phi(s_prev.p, s_prev.v), 0.0)

    # --- smoothness (control-jitter) penalties: body-rate magnitude + angular JERK ---
    # analyze_state showed twitchy control (|w| to 520 deg/s, tilt to 58 deg, near-bang-
    # bang commands) though the path was smooth. CTBR action == body-rate setpoint, so
    # penalize achieved rates directly (no need to store prev action):
    #   -k_w * |w|^2         (overall aggressiveness)
    #   -k_dw * |w - w_prev|^2 (step-to-step jerk = the high-freq buzz). Jerk weighted
    # higher so sustained maneuvering rates (needed for ducks/landing) stay allowed.
    r_smooth = -k_w * jnp.sum(core.w ** 2) - k_dw * jnp.sum((core.w - s_prev.w) ** 2)

    # --- terminal STABILITY: keep upright while settling on the pad (TARGETED) ---
    # Per-step trace of failed landings: tilt diverges MONOTONICALLY 7deg->58deg over 0.18 s
    # while the drone sinks 0.39->0.26 m and speed never exceeds the funnel cap. It is not a
    # braking pitch (decel is only ~1.7 m/s^2 => ~10 deg) — it's an ATTITUDE TUMBLE: at large
    # tilt vertical thrust is T*cos(tilt) (0.53 at 58deg), so it loses lift, sinks, and trips
    # the 60 deg crash limit. Cause: the policy only ever learned FAST fly-through; its jittery
    # control (|w| 200-500 deg/s) destabilises it in the low-speed hover regime the landing needs.
    # A GLOBAL rate penalty already failed (it crippled the ducks/tubes), so gate this to the
    # LAND phase only: penalise tilt beyond ~20 deg near the pad, leaving earlier obstacles free.
    R22 = 1.0 - 2.0 * (core.q[1] ** 2 + core.q[2] ** 2)          # = R[2,2], cos(tilt)
    tilt = jnp.arccos(jnp.clip(R22, -1.0, 1.0))
    near_pad = jnp.exp(-(jnp.linalg.norm(core.p - pc) / 2.5) ** 2)
    r_tilt = jnp.where(tt == LAND,
                       -k_tilt * near_pad * jnp.maximum(tilt - 0.35, 0.0), 0.0)

    reward = (k_prog * r_prog - r_rate + 30.0 * passed.astype(jnp.float32)
              + r_alt + r_speed + r_land + r_smooth + r_tilt)

    M = course['pc'].shape[0]
    new_t = jnp.minimum(t + passed.astype(jnp.int32), M - 1)
    all_done = passed & (t == M - 1)
    next_goal = _goal_of(course, new_t)
    prev_dist_next = jnp.where(passed, jnp.linalg.norm(core.p - next_goal), dist)
    return reward, prev_dist_next, new_t, passed, all_done


def step(key, s: CState, a, pm: E.Params, M_inv, course, k_prog, k_alt=2.0,
         v_cap=2.0, k_speed=1.0, k_land=4.0, k_w=0.0, k_dw=0.0, k_tilt=0.0, v_land=1.2):
    core_in = E.State(s.p, s.v, s.q, s.w, s.Om, s.prev_dist)
    u = E._ctbr_to_u(a, core_in, pm, M_inv)
    core = E._rk4(core_in, u, pm)
    reward, prev_dist_next, new_t, passed, all_done = _reward(
        s, core, pm, course, k_prog, k_alt, v_cap, k_speed, k_land, k_w, k_dw, k_tilt, v_land)
    s_next = CState(core.p, core.v, core.q, core.w, core.Om, prev_dist_next, new_t)
    done = E._done(core, pm) | all_done
    return s_next, reward, done, {"passed": passed, "all_done": all_done, "target": new_t}


def observe_priv(s: CState, course):
    """Privileged state (Informed-decoder target): full state + vector to the
    CURRENT target's pass-center. 20-D."""
    pc = course['pc'][s.target]
    return jnp.concatenate([s.p, s.v, s.q, s.w, s.Om / 2900.0, pc - s.p])


if __name__ == "__main__":
    pm = E.Params()
    M_inv = E._mixer_inv(pm)
    course = make_course()
    key = jax.random.PRNGKey(0)
    B = 8
    s = jax.vmap(reset, in_axes=(0, None, None))(jax.random.split(key, B), pm, course)

    @jax.jit
    def rollout(s, key):
        def body(carry, _):
            s, key = carry
            key, ka = jax.random.split(key)
            a = jax.random.uniform(ka, (B, 4), -1.0, 1.0) if False else \
                jnp.tile(jnp.array([-0.32, 0, 0, 0]), (B, 1))
            s, r, d, info = jax.vmap(step, in_axes=(0, 0, 0, None, None, None, None))(
                jax.random.split(key, B), s, a, pm, M_inv, course, 5.0)
            return (s, key), (r, info["passed"], s.target)
        (s, _), out = jax.lax.scan(body, (s, key), None, length=100)
        return out

    r, passed, tgt = rollout(s, key)
    print("course env ok:", r.shape, "obstacles:", course['cls'].shape[0],
          "mean reward", round(float(r.mean()), 3), "max target reached", int(tgt.max()))
    print("COURSE_ENV_OK")


# ============================================================================
# CIRCLE task (HIL milestone): orbit a radius-R circle past 4 DISTINCT landmarks.
# 4 objects (classes 0-3) equally spaced (90 deg) on the circle = the vision landmarks.
# Reward keeps the drone ON the r=R circle at height z while progressing landmark-to-landmark
# in a LOOP (never terminates) => smooth CCW orbit. Reuses CFB dynamics + simulate_detections.
# ============================================================================
def make_circle(R=2.0, z=1.0, obj_h=0.12, r_pass=0.6):
    import math
    angs = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    centers = [(R * math.cos(a), R * math.sin(a), z) for a in angs]
    dets = [(i, centers[i], (obj_h, obj_h, obj_h)) for i in range(4)]   # 4 distinct classes 0-3
    det_cls = jnp.array([d[0] for d in dets], dtype=jnp.int32)
    det_corners = jax.vmap(CS.box_corners)(
        jnp.array([d[1] for d in dets], float), jnp.array([d[2] for d in dets], float))
    pc = jnp.array(centers, float)
    return dict(det_cls=det_cls, det_corners=det_corners, pc=pc,
                R=jnp.float32(R), z=jnp.float32(z), r_pass=jnp.float32(r_pass))


def reset_circle(key, pm: E.Params, course):
    import math
    R = float(course['R']); z = float(course['z'])
    k1, k2 = jax.random.split(key)
    a0 = jax.random.uniform(k1, (), minval=0.0, maxval=2 * math.pi)      # random start angle
    p = jnp.array([R * jnp.cos(a0), R * jnp.sin(a0), z]) \
        + jax.random.uniform(k2, (3,), minval=-0.15, maxval=0.15)
    Om_hover = E._thrust_to_Om(pm.mass * pm.g / 4.0, pm) * jnp.ones(4)
    t = (jnp.floor(a0 / (math.pi / 2)).astype(jnp.int32) + 1) % 4         # next landmark CCW
    # face the NEXT GATE (look where you're going): yaw toward course['pc'][t]
    gate = course['pc'][t]
    psi = jnp.arctan2(gate[1] - p[1], gate[0] - p[0])
    q = jnp.array([jnp.cos(psi / 2), 0., 0., jnp.sin(psi / 2)])
    dist = jnp.linalg.norm(p - gate)
    return CState(p=p, v=jnp.zeros(3), q=q, w=jnp.zeros(3),
                  Om=Om_hover, prev_dist=dist, target=t)


def _reward_circle(s_prev, core, pm, course, k_prog=1.0, k_rad=3.0, k_z=2.0, v_cap=1.5,
                   k_speed=1.0, k_face=0.3, k_tang=3.0, v_orbit=1.2):
    t = s_prev.target
    goal = course['pc'][t]
    dist = jnp.linalg.norm(core.p - goal)
    r_prog = s_prev.prev_dist - dist                                     # bounded progress to next landmark
    r_rate = (jnp.exp(jnp.minimum(jnp.sum(jnp.abs(core.w)), 17.0)) - 1.0) / (2.0 * (1.0 / pm.dt) * 1e5)
    r_xy = jnp.sqrt(core.p[0] ** 2 + core.p[1] ** 2)
    r_radial = -k_rad * jnp.abs(r_xy - course['R'])                      # stay on the r=R circle
    r_height = -k_z * jnp.abs(core.p[2] - course['z'])                   # stay at altitude z
    r_speed = -k_speed * jnp.maximum(jnp.linalg.norm(core.v) - v_cap, 0.0)
    # PRIMARY ORBIT DRIVER: reward CCW tangential speed (standing still => 0, so no hover trap).
    # tangent (CCW) at (x,y) = (-y, x)/r ; reward the along-tangent velocity up to v_orbit.
    tang = jnp.array([-core.p[1], core.p[0]]) / (r_xy + 1e-6)
    v_tang = core.v[0] * tang[0] + core.v[1] * tang[1]
    r_orbit = k_tang * jnp.clip(v_tang, 0.0, v_orbit)
    # face the CURRENT TARGET GATE (not the centre): the drone looks where it's going, flies
    # toward the landmark, passes it, turns to the next -> natural forward flight, no crab.
    # SMALL weight (progress+pass dominate, so no hover trap). q=[w,x,y,z]; body +x = camera fwd.
    q = core.q
    fx = 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)
    fy = 2.0 * (q[1] * q[2] + q[0] * q[3])
    to_g = goal[:2] - core.p[:2]
    to_g = to_g / (jnp.linalg.norm(to_g) + 1e-6)
    r_face = k_face * (fx * to_g[0] + fy * to_g[1])
    passed = dist < course['r_pass']
    reward = (r_orbit + k_prog * r_prog - r_rate + 30.0 * passed.astype(jnp.float32)
              + r_radial + r_height + r_speed + r_face)
    M = course['pc'].shape[0]
    new_t = (t + passed.astype(jnp.int32)) % M                           # LOOP: never terminates
    next_goal = course['pc'][new_t]
    prev_dist_next = jnp.where(passed, jnp.linalg.norm(core.p - next_goal), dist)
    return reward, prev_dist_next, new_t, passed


def step_circle(key, s: CState, a, pm: E.Params, M_inv, course, k_prog=5.0):
    core_in = E.State(s.p, s.v, s.q, s.w, s.Om, s.prev_dist)
    u = E._ctbr_to_u(a, core_in, pm, M_inv)
    core = E._rk4(core_in, u, pm)
    reward, prev_dist_next, new_t, passed = _reward_circle(s, core, pm, course, k_prog)
    s_next = CState(core.p, core.v, core.q, core.w, core.Om, prev_dist_next, new_t)
    done = E._done(core, pm)                                             # crash only; orbit forever otherwise
    return s_next, reward, done, {"passed": passed, "all_done": jnp.array(False), "target": new_t}


# ============================================================================
# OBSTACLE-AVOIDANCE task via LPTC wide-field optic-flow summary.
# Insect-style: no localisation, just steer away from looming / high-flow regions.
# simulate_flow_field: cast a ray grid over the FOV, analytic depth to primitive cylinder
# obstacles, per-ray TRANSLATIONAL optic flow (=|v_perp|/Z, derotated) + approach (looming),
# pooled into a handful of scalars (LPTC-like). No event rendering -> full RL speed.
# ============================================================================
def make_avoid(z=1.0, obs_r=0.30, x0=-3.5, x_goal=3.5, arena_y=1.8, z_top=2.5, n_obs=5):
    goal = jnp.array([[x_goal, 0.0, z]], float)                 # single "pc" so observe_priv works
    return dict(obs_r=jnp.float32(obs_r), n_obs=n_obs, z_top=jnp.float32(z_top),
                z=jnp.float32(z), x0=jnp.float32(x0), x_goal=jnp.float32(x_goal),
                arena_y=jnp.float32(arena_y), pc=goal,
                det_cls=jnp.zeros((0,), jnp.int32), det_corners=jnp.zeros((0, 8, 3), float))


def sample_avoid_obs(key, course, n_obs):
    """RANDOM obstacle layout per episode (generalisation) WITH HORIZONTAL SEPARATION: the
    corridor's x-range is split into n_obs equal bands, one cylinder per band (jittered within
    it), so obstacles are always separated along the flight direction -> the drone meets them
    one at a time with room to weave in y (no impassable wall of clustered obstacles).
    Returns (xy [N,2], r [N])."""
    k1, k2, k3 = jax.random.split(key, 3)
    xlo, xhi = course['x0'] + 1.0, course['x_goal'] - 0.8
    band = (xhi - xlo) / n_obs
    centres = xlo + band * (jnp.arange(n_obs) + 0.5)
    xs = centres + band * jax.random.uniform(k1, (n_obs,), minval=-0.30, maxval=0.30)  # jitter within band
    ys = jax.random.uniform(k2, (n_obs,), minval=-course['arena_y'] + 0.4, maxval=course['arena_y'] - 0.4)
    r = course['obs_r'] * jax.random.uniform(k3, (n_obs,), minval=0.7, maxval=1.3)
    return jnp.stack([xs, ys], -1), r


def reset_avoid(key, pm: E.Params, course):
    k1, = jax.random.split(key, 1)
    y0 = jax.random.uniform(k1, (), minval=-0.4, maxval=0.4)
    p = jnp.array([course['x0'], y0, course['z']])
    Om_hover = E._thrust_to_Om(pm.mass * pm.g / 4.0, pm) * jnp.ones(4)   # face +x (identity quat)
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    dist = jnp.linalg.norm(p - goal)
    return CState(p=p, v=jnp.zeros(3), q=jnp.array([1., 0., 0., 0.]), w=jnp.zeros(3),
                  Om=Om_hover, prev_dist=dist, target=jnp.array(0, jnp.int32))


FLOW_N_SECT = 8   # horizontal azimuth sectors (fine gap resolution) -> obs = N_SECT + 2

def simulate_flow_field(cam_pos, R, v_world, obs_xy, obs_r, z_top,
                        fov_h=2.10, fov_v=1.40, n_az=16, n_el=5):
    """LPTC wide-field flow summary, RICHER version: FLOW_N_SECT horizontal azimuth sectors of
    translational flow (L->R across the FOV, so the policy can localise the GAP) + central looming
    'div' + global max looming. Returns [FLOW_N_SECT + 2]. Vertical (U/D) dropped: the pole
    obstacles span all heights, so only horizontal steering matters."""
    az = jnp.linspace(-fov_h / 2, fov_h / 2, n_az)
    el = jnp.linspace(-fov_v / 2, fov_v / 2, n_el)
    AZ, EL = jnp.meshgrid(az, el)
    az_f, el_f = AZ.reshape(-1), EL.reshape(-1)                          # [K]
    ce = jnp.cos(el_f)
    d_cam = jnp.stack([ce * jnp.cos(az_f), ce * jnp.sin(az_f), jnp.sin(el_f)], -1)  # [K,3] cam frame
    d_world = d_cam @ R.T                                                # body->world
    o = cam_pos

    def depth_ray(dw):
        oxy, dxy = o[:2], dw[:2]

        def t_obs(c, r):
            a = jnp.dot(dxy, dxy) + 1e-9
            b = 2 * jnp.dot(dxy, oxy - c)
            cc = jnp.dot(oxy - c, oxy - c) - r * r
            disc = b * b - 4 * a * cc
            t = jnp.where(disc > 0, (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) / (2 * a), 1e6)
            zt = o[2] + t * dw[2]
            hit = (disc > 0) & (t > 0.05) & (zt >= 0.0) & (zt <= z_top)
            return jnp.where(hit, t, 1e6)
        return jnp.min(jax.vmap(t_obs)(obs_xy, obs_r))
    Z = jax.vmap(depth_ray)(d_world)                                     # [K]
    v_cam = R.T @ v_world
    valong = d_cam @ v_cam                                               # [K]
    v_perp = v_cam[None, :] - valong[:, None] * d_cam
    Zc = jnp.maximum(Z, 0.25)
    flow = jnp.linalg.norm(v_perp, axis=-1) / Zc                         # translational flow (derotated)
    loom = jnp.maximum(valong, 0.0) / Zc                                 # approach / depth ~ 1/TTC
    edges = jnp.linspace(-fov_h / 2, fov_h / 2, FLOW_N_SECT + 1)         # sector edges (L->R)
    def sect(s):
        mask = (az_f >= edges[s]) & (az_f < edges[s + 1])
        return jnp.sum(flow * mask) / (jnp.sum(mask) + 1e-6)
    sectors = jnp.stack([sect(s) for s in range(FLOW_N_SECT)])          # [N_SECT] horizontal flow
    center = jnp.abs(az_f) < 0.45
    div = jnp.sum(loom * center) / (jnp.sum(center) + 1e-6)             # central looming
    loom_max = jnp.max(loom)                                             # nearest-obstacle looming
    return jnp.concatenate([sectors, jnp.array([div, loom_max])])       # [N_SECT + 2]


def _reward_avoid(s_prev, core, pm, course, obs_xy, obs_r, k_prog=5.0, k_crash=10.0):
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    dist = jnp.linalg.norm(core.p - goal)
    r_prog = s_prev.prev_dist - dist                                     # progress toward far end
    r_rate = (jnp.exp(jnp.minimum(jnp.sum(jnp.abs(core.w)), 17.0)) - 1.0) / (2.0 * (1.0 / pm.dt) * 1e5)
    # collision: nearest cylinder surface distance in xy
    dxy = jnp.linalg.norm(core.p[:2][None, :] - obs_xy, axis=-1) - obs_r
    clr = jnp.min(dxy)
    collided = clr < 0.12
    oob = (jnp.abs(core.p[1]) > course['arena_y']) | (core.p[2] < 0.2) | (core.p[2] > course['z_top'])
    r_height = -1.0 * jnp.abs(core.p[2] - course['z'])
    reached = core.p[0] > course['x_goal']
    reward = (k_prog * r_prog - r_rate + r_height
              - k_crash * (collided | oob).astype(jnp.float32)
              + 30.0 * reached.astype(jnp.float32))
    prev_dist_next = dist
    done = collided | oob | reached
    return reward, prev_dist_next, done, reached


def step_avoid(key, s: CState, a, pm: E.Params, M_inv, course, obs_xy, obs_r, k_prog=5.0):
    core_in = E.State(s.p, s.v, s.q, s.w, s.Om, s.prev_dist)
    u = E._ctbr_to_u(a, core_in, pm, M_inv)
    core = E._rk4(core_in, u, pm)
    reward, prev_dist_next, done_task, reached = _reward_avoid(s, core, pm, course, obs_xy, obs_r, k_prog)
    s_next = CState(core.p, core.v, core.q, core.w, core.Om, prev_dist_next, s.target)
    done = E._done(core, pm) | done_task
    return s_next, reward, done, {"passed": reached, "all_done": reached, "target": s.target}


# ============ 2D fixed-height HEADING-STEERING obstacle avoidance (EVENT-PROXY obs) ============
# Simplest reactive task (user spec 2026-08-03): constant HEIGHT + constant forward SPEED; the
# policy outputs a desired heading on the UNIT CIRCLE (2-D), read EGOCENTRICALLY (a heading change
# relative to current facing). A proportional, rate-limited yaw law turns toward it each 100 Hz
# tick ("refresh & steer, don't need to reach"). Observation = analytic EVENT FRAME (event_proxy)
# of sparse pole obstacles + an egocentric GOAL BEARING (walls/goal aren't visible in events).
# No CFB dynamics here (kinematic) -> fast; the control scheme (heading + alt-hold) is deployable.
STEER_DT = 0.01            # 100 Hz control
STEER_V0 = 1.2             # fixed forward speed (m/s)
STEER_YAWRATE_MAX = 3.0    # rad/s (~172 deg/s)
STEER_KYAW = 2.5           # P gain: egocentric rel-heading (rad) -> yaw rate
STEER_H = 64               # DEFAULT event frame size (H x W x 2); tasks may override via 'h<N>'
STEER_FOV = 90.0
STEER_F = EP.focal_from_fov(STEER_FOV, STEER_H)

# ---- EVENT-FRAME RESOLUTION (deploy lever, 2026-08-11) ----------------------------------------
# The CNN encoder is the N6 latency bottleneck: the 64x64 policy measures 2.4 Hz on the board.
# Halving the linear resolution cuts first-layer conv work ~4x. Task suffix 'h32' selects it, e.g.
# `speedzoneh32`, `magzone24h32`. 64 stays the default so every existing checkpoint is unaffected.
#
# Two effects fight each other when H changes, which is why the gain is CALIBRATED, not assumed:
#   * focal ~ H, so image-plane speeds (and thus per-point event magnitude) scale DOWN with H;
#   * pixel area ~ 1/H^2, so MORE edge points accumulate into each pixel, scaling values UP.
# Meanwhile the nuisance DR (salt_val, horizon_mag) is in ABSOLUTE magnitude units, so if the
# signal level moves, the effective SNR moves with it and the task silently changes difficulty.
# EVENT_GAIN_BY_H is therefore chosen empirically to hold the frame's median nonzero value roughly
# constant across resolutions -- the same principle the Isaac bridge uses when it auto-calibrates
# its count->uint8 gain to the proxy's median nonzero of 51.
# Calibrated 2026-08-11 by calibration/calibrate_event_gain.py on `magzone`, matching the SATURATION
# FRACTION (H=64: 10.22% of pixels pegged at 255; H=32 @ gain 2.25: 10.06%). Median-nonzero is a
# poor target here because BOTH resolutions are already heavily saturated -- see the note below.
# Occupancy is gain-invariant and intrinsically higher at 32 (25.7% vs 14.7%): the same edge
# points fall into 4x fewer pixels. That is a genuine information change no gain can undo.
#
# NB (pre-existing, applies at 64 too): `_build_obs` scales the frame by 32 into uint8, so the
# delivered observation is near-binarised -- median nonzero 236/255 with ~10% fully saturated.
# Magnitude information is largely clipped away and the policy is effectively reading edge
# GEOMETRY. Worth revisiting for the int8 deploy path.
EVENT_GAIN = 3.0                       # simulate_event_frame's own default gain, used at H=64
# H=32 RECALIBRATED 2026-08-13 for the BINNED path (render 64 -> sum-pool 2x2). Pooling sums the
# per-pixel noise as well as the signal, so the old native-32 value (2.25) is far too hot here;
# 1.0 matches the 64x64 saturation fraction (2.98% vs 2.80%). Isaac --calib-target = 76.
EVENT_GAIN_BY_H = {64: 3.0, 32: 1.0}   # see calibration/calibrate_event_gain.py

# ---- GenX320 SINGLE-CHANNEL OBS (`gx` tasks, calibrated 2026-08-18 against genx320_grayscale/
# rec1.bin, 186 recorded frames, and openmv/drivers/sensors/genx320.c) --------------------------
# What the board actually delivers to the network (n6_sweep_record_1ch.py / post_process_histo):
#   1. EHC DIFF3D accumulates a SIGNED net (ON-OFF) per pixel at 320x320 over 20 ms.
#   2. px320 = USAT(net * 16 + 128)                       <- per pixel, AT 320
#   3. obs   = draw_image(hint=AREA) 320 -> 32            <- 10x10 MEAN of those uint8s
# so, away from the rare per-pixel saturation,
#   obs_cell = 128 + (16/100) * (signed event count in the 10x10 block).
# The firmware's CONTRAST_DEFAULT=16 therefore multiplies a NATIVE-resolution int8; our frame is
# already a pooled aggregate, so the scale that belongs in the wrapper is 16/100 = 0.16 counts->uint8
# and the proxy's deposit is expressed in EVENT COUNTS (GENX_EVT below). Using 16 here would
# overdrive the frame by the bin area (~100x) and peg every edge.
GENX_CONTRAST = 0.16                   # = CONTRAST_DEFAULT / (320/32)^2, uint8 units per net count


def genx_contrast(H):
    """Pooled-equivalent firmware contrast at obs resolution H: the hardware
    applies CONTRAST_DEFAULT=16 per NATIVE 320-pixel then AREA-means a
    (320/H)^2 block, so the aggregate scale is 16/(320/H)^2. H=32 -> 0.16
    (== GENX_CONTRAST), H=64 -> 0.64."""
    return 16.0 / (320.0 / float(H)) ** 2


def genx_evt(H):
    """Counts deposited per visible edge point at obs resolution H. Fitted at
    H=32 (GENX_EVT); scales with CELL AREA (mass per image area conserved) so
    the pooled amplitude stays on the rec1.bin fit at any H — hardware dev is
    scale-invariant (~1 count per swept NATIVE pixel), a point's deposit must
    shrink as cells get finer. H=32 -> 18, H=64 -> 4.5. Re-fit properly per H
    when a native-res recording exists."""
    return GENX_EVT / (float(H) / 32.0) ** 2
GENX_BRIGHTNESS = 128
# Counts deposited per visible edge point once it clears the firing knee, and that knee in render
# pixels. `sat_knee` exists because a DIFF3D pixel fires on the CONTRAST step, not on how far the
# edge travelled -- see event_proxy.simulate_event_frame. GENX_EVT is the one fitted number: it is
# set so the sim's active-|px-128| quantiles land on the recording's (p50 2, p75 6, p90 12, p95 17).
GENX_EVT = 18.0
GENX_KNEE = 0.35                       # render px; below this the edge does not reliably fire
# The nuisance DR magnitudes (salt_val, horizon_mag) are in the OLD linear-magnitude units; at
# 0.16 uint8/count they would quantise to zero, so scale them into count units for gx.
GENX_NOISE_COUNTS = 8.0


# ---- GX SCENE ENRICHMENT (round gx-2, 2026-08-19) --------------------------------------------
# Isaac A/B (NEXT_TRAINING, eval agent 2026-08-19): the gx policy scored 30% closed-loop against
# the 2-channel w30 baseline's 62%, with 12/30 out-of-bounds. Frame forensics found the cause and
# it is NOT calibration (p90 matched, median |net| per native px = 1.0, the rec1.bin convention):
#
#     background fraction   proxy 88%   |   Isaac 56%   |   real rec1.bin 39%
#
# i.e. the proxy scene is far emptier than either. The 2-channel path was insensitive to this
# because its x32 scaling near-binarises every lit pixel; the graded 1-channel frame exposes it.
# The missing content is ground, walls and clutter -- the parts of a forest the course never
# modelled because a sparse ON/OFF frame could not show them anyway.
#
# These points are OBSERVATION-ONLY: they enter `pts`/`valid` (what the event proxy renders) and
# never `obs_xy`/`obs_r` (what collision and the privileged decoder use), so dynamics, reward and
# the crash test are untouched. Applied ONLY on gx tasks, so every 2-channel task stays
# bit-identical.
# The ground is shaded ANALYTICALLY per pixel (event_proxy `ground=`), not sampled as points:
# the z-buffer keeps one point per pixel, so point coverage saturates at the pixels the ground
# subtends and extra points only compete for them (measured: 1500 -> 8000 ground points moved
# occupancy by <2 pt). Clutter and walls stay points -- they are objects, not a surface.
GX_GROUND_GRAIN = 0.25             # world texture cell (m); sets the ground's spatial frequency
GX_POL_GRAIN = 0.8                 # polarity-hash cell (m): one contrast sign per ~contour, not per point
                                   # (per-point signs annihilate 55% of event mass in 2x2 pooling; real ~11%)
GX_GROUND_FILL = (0.05, 0.50)      # sparse-biased wide range 2026-08-24 (user-agreed; flight-frame recording will arbitrate)
GX_GROUND_ZMAX = 40.0              # ignore ground beyond this range (horizon singularity)
GX_CLUTTER_N = 400                 # low undergrowth/debris just above the ground
GX_CLUTTER_Z = (0.02, 0.45)
GX_WALL_NX, GX_WALL_NZ = 240, 12   # denser corridor walls (the oob spike points here)
GX_KEEP = (0.45, 1.0)              # per-episode density DR on everything added here


# ---- SCENE-TYPE DR ('sc' token, 2026-08-22): per-episode scene family lottery -------------
# Literature pattern (Loquercio et al. Science Robotics 2021: forest/gap/disaster/city;
# MuJoCo-Drones-Gym: FOREST/URBAN/INDOOR/RANDOM): diversity ACROSS qualitatively different
# procedural scene TYPES, not more knobs on one family. Four families, sampled per episode:
#   0 corridor  — the classic: both walls, full height           (hw = 2.0)
#   1 open      — no walls, distant background ring 12-35 m      (hw ~ U(4, 8))
#   2 half-open — ONE low "hedge" wall (0.5-1.5 m), random side  (hw = 2.5)
#   3 messy     — walls with random height + 2.5 m segment GAPS,
#                 extra clutter                                   (hw ~ U(2, 3))
# All content is OBSERVATION-only except the arena half-width, which becomes DYNAMIC:
# it rides as a 7th element on the wall descriptor and _reward_steer reads it when present.
SC_BG_N = 300                       # background ring points (treeline/buildings analog)
SC_WALL_SEG = 2.5                   # messy-wall gap segment length (m)
SC_WALL_TIER_P = (0.45, 0.30)       # per-segment texture lottery: P(full), P(sparse); rest = white
SC_WALL_OUT_N = 120                 # outline points per line (top edge / floor junction), per side
GX_DILATE_W = 0.08                  # contour smear: real edges are ~2 px (optics+latency); each
                                    # 64x64 deposit leaks w into right/below/diag before pooling


# Scene-spec v2 (2026-08-23, user-directed): course length 5-50 m, width 1.5-10 m,
# flight height 0.3-2.0 m, wall height 1.5-10 m, per-episode edge-strength (deposit
# gain) 0.5-2x, tree-count thinning 0.4-1.0. Extras ride the wall-descriptor slots:
# wall[6]=half_w, wall[7]=x_goal, wall[8]=flight z, and drp gains a gain-multiplier.
SC_LEN = (5.0, 50.0)
SC_HW = (1.5, 10.0)
SC_ZFLY = (0.3, 2.0)
SC_WALL_H = (1.5, 10.0)
SC_GAIN = (0.5, 2.0)
SC_TREE_KEEP = (0.4, 1.0)


def cylinder_occlusion_mask(cam, pts, obs_xy, obs_r, obs_h):
    """Solid-silhouette occlusion (audit fix 2026-08-23): a point is HIDDEN if the
    camera->point ray passes through a NEARER trunk cylinder below that trunk's
    height. The per-pixel z-buffer only occludes where near-object POINTS land;
    this closes the gaps so trunks are opaque like Isaac/reality. Points on a
    trunk's own surface are exempted by the 1.3r self-distance test.
    cam [3], pts [N,3], obs_xy [M,2], obs_r [M], obs_h [M] -> keep [N] bool."""
    pts = pts[:, :3]
    d = pts[:, None, :2] - obs_xy[None, :, :]              # [N,M,2] point->centre
    seg = pts[:, :2] - cam[None, :2]                       # [N,2] cam->point (plan view)
    seglen2 = jnp.maximum((seg ** 2).sum(-1, keepdims=True), 1e-9)
    c = cam[None, None, :2] - obs_xy[None, :, :]           # cam->centre... build t along seg
    t = -((c) * seg[:, None, :]).sum(-1) / seglen2         # closest approach param [N,M]
    t = jnp.clip(t, 0.0, 1.0)
    closest = cam[None, None, :2] + t[..., None] * seg[:, None, :]
    dist_line = jnp.linalg.norm(closest - obs_xy[None, :, :], axis=-1)   # [N,M]
    z_at = cam[2] + t * (pts[:, 2:3] - cam[2])             # ray height at approach [N,M]
    frac_in = t * jnp.sqrt(seglen2)                        # metres along ray
    pt_dist = jnp.sqrt(seglen2)[:, 0]
    nearer = frac_in < (pt_dist[:, None] - 0.15)           # cylinder strictly in front
    own = jnp.linalg.norm(d, axis=-1) < obs_r[None, :] * 1.3
    blocked = (dist_line < obs_r[None, :] * 0.95) & (z_at < obs_h[None, :]) & nearer & (~own)
    return ~blocked.any(-1)


def tree_alive_mask(key, n_obs, keep):
    """Per-episode whole-tree dropout: alive [n_obs] bool with ~keep fraction."""
    return jax.random.uniform(key, (n_obs,)) < keep


def gx_scene_points_sc(key, course):
    """Scene-type DR v2 -> (pts, valid, extras dict). Static shapes; scene switching
    via validity masks. Texture density = per-element keep fractions."""
    ks = jax.random.split(key, 16)
    # gx-6: corridor scenes upweighted (deployment-regime specialist training)
    scene = jax.random.choice(ks[0], 4, p=jnp.array([0.4, 0.2, 0.2, 0.2]))
    xg_full = jax.random.uniform(ks[12], (), minval=SC_LEN[0], maxval=SC_LEN[1])
    xg_corr = jax.random.uniform(jax.random.fold_in(ks[12], 1), (),
                                 minval=10.0, maxval=30.0)
    xg_ep = jnp.where(scene == 0, xg_corr, xg_full)
    z_ep = jax.random.uniform(ks[13], (), minval=SC_ZFLY[0], maxval=SC_ZFLY[1])
    gain_mul = jax.random.uniform(ks[14], (), minval=SC_GAIN[0], maxval=SC_GAIN[1])
    hw_open = jax.random.uniform(ks[1], (), minval=3.0, maxval=SC_HW[1])
    hw_narrow = jax.random.uniform(ks[15], (), minval=SC_HW[0], maxval=3.0)
    hw_ep = jnp.select([scene == 0, scene == 1, scene == 2, scene == 3],
                       [hw_narrow, hw_open, hw_narrow + 0.5,
                        jax.random.uniform(ks[1], (), minval=2.0, maxval=4.0)])
    x0, xg = course['x0'] - 2.0, xg_ep + 2.0

    # ground clutter (all scenes; count DR via keep fraction 0.1-1.0)
    n_cl = GX_CLUTTER_N
    clutter = jnp.stack([
        jax.random.uniform(ks[2], (n_cl,), minval=x0, maxval=xg),
        jax.random.uniform(ks[3], (n_cl,), minval=-hw_ep, maxval=hw_ep),
        jax.random.uniform(ks[4], (n_cl,), minval=GX_CLUTTER_Z[0], maxval=GX_CLUTTER_Z[1])], -1)
    f_cl = jax.random.uniform(ks[5], (), minval=0.1, maxval=1.0)
    v_cl = EP.density_mask(ks[6], n_cl, f_cl * jnp.where(scene == 3, 1.0, 0.7))

    # walls: height + texture-density + segment-gap DR; per-side enable by scene
    wall_h = jnp.select([scene == 0, scene == 1, scene == 2, scene == 3],
                        [jax.random.uniform(ks[7], (), minval=SC_WALL_H[0], maxval=SC_WALL_H[1]),
                         1.0,
                         jax.random.uniform(ks[7], (), minval=0.5, maxval=1.5),
                         jax.random.uniform(ks[7], (), minval=SC_WALL_H[0], maxval=SC_WALL_H[1])])
    wl = EP.wall_edge_points(-hw_ep, course['x0'], xg_ep, wall_h,
                             nx=GX_WALL_NX, nz=GX_WALL_NZ)
    wr = EP.wall_edge_points(+hw_ep, course['x0'], xg_ep, wall_h,
                             nx=GX_WALL_NX, nz=GX_WALL_NZ)
    side = jax.random.bernoulli(ks[8], 0.5)              # half-open: which side survives
    on_l = jnp.select([scene == 0, scene == 1, scene == 2, scene == 3],
                      [1.0, 0.0, jnp.where(side, 1.0, 0.0), 1.0])
    on_r = jnp.select([scene == 0, scene == 1, scene == 2, scene == 3],
                      [1.0, 0.0, jnp.where(side, 0.0, 1.0), 1.0])
    f_w = jax.random.uniform(ks[9], (), minval=0.15, maxval=1.0)   # texture density
    nseg = 16
    seg_keep = jax.random.bernoulli(ks[10], jnp.where(scene == 3, 0.6, 1.0), (nseg,))
    # texture-tier lottery (2026-08-24): real walls range from fully textured to plain white.
    # Per SIDE x SEGMENT: tier 0 = textured (grid as before), tier 1 = sparse patches (poster/
    # door-frame scale clusters on an otherwise white wall), tier 2 = white (face silent; only
    # the silhouette outline below emits). Sides draw independently -> mixed corridors happen.
    kl, kr, ktl, ktr = jax.random.split(ks[11], 4)
    def tier_draw(k):
        u = jax.random.uniform(k, (nseg,))
        return jnp.where(u < SC_WALL_TIER_P[0], 0,
                         jnp.where(u < SC_WALL_TIER_P[0] + SC_WALL_TIER_P[1], 1, 2))
    tier_l, tier_r = tier_draw(ktl), tier_draw(ktr)
    def patch_mask(w):
        # clustered keep: hash 0.9x0.7 m wall cells -> ~22% of cells survive as contiguous patches
        cell = jnp.stack([jnp.floor(w[:, 0] / 0.9), jnp.floor(w[:, 2] / 0.7)], -1)
        hsh = jnp.mod(jnp.sin(cell @ jnp.array([12.9898, 78.233])) * 43758.5453, 1.0)
        return hsh < 0.22
    def wall_valid(w, on, tier, kmask):
        seg = jnp.clip(((w[:, 0] - course['x0']) / SC_WALL_SEG).astype(jnp.int32), 0, nseg - 1)
        t = tier[seg]
        tex = jnp.where(t == 0, True, jnp.where(t == 1, patch_mask(w), False))
        return (EP.density_mask(kmask, w.shape[0], f_w) & seg_keep[seg]
                & tex & (on > 0.5))
    v_wl, v_wr = wall_valid(wl, on_l, tier_l, kl), wall_valid(wr, on_r, tier_r, kr)
    # silhouette outline (all tiers): top edge + floor junction still emit on a white wall --
    # this is what a real plain-walled corridor gives the sensor. Present for textured walls
    # too (their top edge is also a contour), so it is unconditional on tier.
    def wall_outline(y):
        xs = jnp.linspace(course['x0'], xg_ep, SC_WALL_OUT_N)
        top = jnp.stack([xs, jnp.full(xs.shape, y), jnp.full(xs.shape, wall_h)], -1)
        bot = jnp.stack([xs, jnp.full(xs.shape, y), jnp.full(xs.shape, 0.2)], -1)
        return jnp.concatenate([top, bot], 0)
    wl_o, wr_o = wall_outline(-hw_ep), wall_outline(+hw_ep)
    def outline_valid(w, on):
        seg = jnp.clip(((w[:, 0] - course['x0']) / SC_WALL_SEG).astype(jnp.int32), 0, nseg - 1)
        return seg_keep[seg] & (on > 0.5)
    v_wlo, v_wro = outline_valid(wl_o, on_l), outline_valid(wr_o, on_r)

    # background ring (open + messy): distant clutter at 12-35 m, z 0-4
    kb = jax.random.split(jax.random.fold_in(key, 77), 4)
    az = jax.random.uniform(kb[0], (SC_BG_N,), minval=-jnp.pi, maxval=jnp.pi)
    rad = jax.random.uniform(kb[1], (SC_BG_N,), minval=12.0, maxval=35.0)
    cx = (course['x0'] + xg_ep) / 2.0
    bg = jnp.stack([cx + rad * jnp.cos(az), rad * jnp.sin(az),
                    jax.random.uniform(kb[2], (SC_BG_N,), minval=0.0, maxval=4.0)], -1)
    v_bg = EP.density_mask(kb[3], SC_BG_N,
                           jnp.where((scene == 1) | (scene == 3), 0.7, 0.0))

    # outward normals (motion-consistent polarity): markings get world-anchored random in-plane
    # directions; the outline's top edge points up (wall against sky), floor junction down.
    n_cl = EP.hashed_plane_normals(clutter, 'ground')
    n_wl_t = EP.hashed_plane_normals(wl, 'wallx')
    n_wr_t = EP.hashed_plane_normals(wr, 'wallx')
    up = jnp.array([0.0, 0.0, 1.0]); dn = jnp.array([0.0, 0.0, -1.0])
    n_out = jnp.concatenate([jnp.tile(up, (SC_WALL_OUT_N, 1)), jnp.tile(dn, (SC_WALL_OUT_N, 1))], 0)
    n_bg = EP.hashed_plane_normals(bg, 'ground')
    pts = jnp.concatenate([clutter, wl, wr, wl_o, wr_o, bg], 0)
    nrm = jnp.concatenate([n_cl, n_wl_t, n_wr_t, n_out, n_out, n_bg], 0)
    pts = jnp.concatenate([pts, nrm], -1)
    valid = jnp.concatenate([v_cl, v_wl, v_wr, v_wlo, v_wro, v_bg], 0)
    tk_full = jax.random.uniform(jax.random.fold_in(key, 5), (),
                                 minval=SC_TREE_KEEP[0], maxval=SC_TREE_KEEP[1])
    tk_corr = jax.random.uniform(jax.random.fold_in(key, 6), (),
                                 minval=0.8, maxval=1.0)
    return pts, valid, dict(hw=hw_ep, x_goal=xg_ep, z=z_ep, gain_mul=gain_mul,
                            tree_keep=jnp.where(scene == 0, tk_corr, tk_full))


def gx_scene_points(key, course):
    """Extra OBSERVATION-only edge points for the gx regime -> (pts [N,3], valid [N]).

    Ground texture is sampled over the corridor footprint at z=0: it is static world content, so
    the existing ego-motion projection makes it stream correctly and the z-buffer occludes it
    behind trunks. Density is randomised per episode (GX_KEEP) so the policy cannot depend on one
    particular texture -- real ground cover varies far more than this."""
    k_cx, k_cy, k_cz, k_kc, k_kw = jax.random.split(key, 5)
    x0 = course['x0'] - 2.0
    xg = course['x_goal'] + 2.0
    hw = course['half_w']

    clutter = jnp.stack([
        jax.random.uniform(k_cx, (GX_CLUTTER_N,), minval=x0, maxval=xg),
        jax.random.uniform(k_cy, (GX_CLUTTER_N,), minval=-hw, maxval=hw),
        jax.random.uniform(k_cz, (GX_CLUTTER_N,), minval=GX_CLUTTER_Z[0],
                           maxval=GX_CLUTTER_Z[1])], -1)
    walls = jnp.concatenate([
        EP.wall_edge_points(-hw, course['x0'], course['x_goal'], course['z_top'],
                            nx=GX_WALL_NX, nz=GX_WALL_NZ),
        EP.wall_edge_points(+hw, course['x0'], course['x_goal'], course['z_top'],
                            nx=GX_WALL_NX, nz=GX_WALL_NZ)], 0)

    def keep(k, n):
        k_f, k_m = jax.random.split(k)          # separate keys: fraction and mask
        f = jax.random.uniform(k_f, (), minval=GX_KEEP[0], maxval=GX_KEEP[1])
        return EP.density_mask(k_m, n, f)

    pts = jnp.concatenate([clutter, walls], 0)
    valid = jnp.concatenate([keep(k_kc, GX_CLUTTER_N), keep(k_kw, walls.shape[0])], 0)
    return pts, valid


def gx_ground_dr(key, H=32):
    """Per-episode params for event_proxy's analytic ground shader -> dict.
    `evt` is per-CELL counts, so it carries the same area scaling as the point
    deposits (genx_evt): finer cells hold proportionally fewer counts."""
    k_bare, k_fill = jax.random.split(key)
    # texture lottery (2026-08-24): p=0.2 the floor is BARE (polished concrete -> zero ground
    # events); otherwise textured up to 0.5. Bare floor + white walls = the real empty-room case.
    fill = jnp.where(jax.random.uniform(k_bare, ()) < 0.2, 0.0,
                     jax.random.uniform(k_fill, (), minval=GX_GROUND_FILL[0],
                                        maxval=GX_GROUND_FILL[1]))
    return dict(evt=jnp.float32(genx_evt(H)), knee=jnp.float32(GENX_KNEE),
                grain=jnp.float32(GX_GROUND_GRAIN), fill=fill,
                zmax=jnp.float32(GX_GROUND_ZMAX))


def sample_zoneforest_obs_sc(key, course, n_obs):
    """Scene-spec v2 sampler (2026-08-23): zone forest + per-episode length/width/
    flight-z/tree-count/edge-strength DR + the sc scenery lottery. Standard
    6-tuple, with wall extended to 9 slots ([6]=half_w, [7]=x_goal, [8]=z_fly)
    and drp extended to [3]=gain_mul, [4:4+n_obs]=tree heights (threaded for the
    per-frame solid-occlusion mask in the obs path)."""
    kb, ksc, ka = jax.random.split(key, 3)
    xy, r, pts, valid, wall, drp, h = sample_zoneforest_obs(kb, course, n_obs, return_h=True)
    spts, svalid, ex = gx_scene_points_sc(ksc, course)

    x0t = course['x0'] + 1.5
    sx = (ex['x_goal'] - 1.0 - x0t) / (course['x_goal'] - 1.0 - x0t)
    sy = (ex['hw'] - 0.3) / (course['half_w'] - 0.3)
    dx = (x0t + (xy[:, 0] - x0t) * sx) - xy[:, 0]
    dy = xy[:, 1] * (sy - 1.0)
    xy = jnp.stack([xy[:, 0] + dx, xy[:, 1] + dy], -1)
    blocks = (FOREST_NAZ_DR * FOREST_NV_DR, FOREST_NBR * 6, SHADOW_N)
    dxp = jnp.concatenate([jnp.repeat(dx, b) for b in blocks])
    dyp = jnp.concatenate([jnp.repeat(dy, b) for b in blocks])
    nTree = dxp.shape[0]
    pts = pts.at[:nTree, 0].add(dxp)
    pts = pts.at[:nTree, 1].add(dyp)

    valid = valid.at[nTree:].set(False)      # base corridor walls off (sc builds its own)

    alive = tree_alive_mask(ka, n_obs, ex['tree_keep'])
    pt_alive = jnp.concatenate([jnp.repeat(alive, b) for b in blocks])
    valid = valid.at[:nTree].set(valid[:nTree] & pt_alive)

    # SHAPE LOTTERY (2026-08-23): cylinder / square / rectangle / thin tube,
    # up to ~15 deg lean in a random direction. Regenerates the trunk-point
    # block from the translated centres; branches/shadows keep their blocks.
    ks1, ks2, ks3 = jax.random.split(jax.random.fold_in(key, 23), 3)
    shape = jax.random.randint(ks1, (n_obs,), 0, 4)
    lean_a = jax.random.uniform(ks2, (n_obs,), minval=0.0, maxval=2 * jnp.pi)
    lean_m = jnp.tan(jax.random.uniform(ks3, (n_obs,), minval=0.0, maxval=0.26))
    lean_vec = jnp.stack([lean_m * jnp.cos(lean_a), lean_m * jnp.sin(lean_a)], -1)
    new_trunks, trunk_nrm = EP.pillar_edge_points(xy, r, h, shape, lean_vec,
                                                  nv=FOREST_NV_DR, naz=FOREST_NAZ_DR,
                                                  with_normals=True)
    pts = pts.at[:blocks[0] * n_obs].set(new_trunks)
    # normals: trunks exact radial (the looming cue); branches/shadows/base-walls hashed markings
    nrm = jnp.concatenate([trunk_nrm,
                           EP.hashed_plane_normals(pts[blocks[0] * n_obs:], 'ground')], 0)
    pts = jnp.concatenate([pts, nrm], -1)

    # collision: shape-aware radius (box half-diagonal; tube thin), centre
    # shifted by the lean at the flight height
    r_shape = jnp.select([shape == 0, shape == 1, shape == 2, shape == 3],
                         [r, r * 1.4142, r * 1.4142 * 1.05, r / 3.0])
    xy = xy + lean_vec * ex['z']
    r_eff = jnp.where(alive & (h > ex['z'] + 0.05), r_shape, -1.0)   # fly-over allowed

    wall9 = jnp.concatenate([wall, jnp.stack([ex['hw'], ex['x_goal'], ex['z']])])
    drp_sc = jnp.concatenate([drp, ex['gain_mul'][None], h])
    return (xy, r_eff, jnp.concatenate([pts, spts], 0),
            jnp.concatenate([valid, svalid], 0), wall9, drp_sc)


# ---- BINNED PROXY (round D-REVISED, 2026-08-13): match how the frame is GENERATED ------------
# dv3_forest13s was healthy in-proxy (82-83% over 61.5k eps) and scored 0/30 on Isaac in EVERY
# configuration, with a systematic one-sided veer. Transport, bearing convention, image mirror,
# shadows and gain were all eliminated with hard evidence. The supported mechanism is BIN
# BLOBBINESS: Isaac sum-bins ~100 px into each 32x32 cell, producing fat contiguous blobs, while
# a native-32 proxy render deposits thin sparse streaks. Matching the intensity statistics was
# not enough -- the MICROSTRUCTURE differs.
#
# Fix: render the proxy at the native STEER_H and SUM-POOL down to the target resolution, so
# training frames are produced by the same binning the eval and the hardware perform. Plus light
# blob DR (a 3x3 box blur on a fraction of ticks) for robustness.
#
# NOTE: the nuisance DR (salt, horizon) is injected at RENDER resolution and therefore gets summed
# by the pooling, exactly as Isaac's noise does. The gain must be re-calibrated for this path --
# see calibration/calibrate_event_gain.py.
EVENT_BLOB_P = 0.3                     # probability of applying the 3x3 box blur


def _box_blur3(fr):
    """3x3 box blur, jit-safe (edge-padded shifts, no conv kernel needed)."""
    H, W = fr.shape[0], fr.shape[1]
    p = jnp.pad(fr, ((1, 1), (1, 1), (0, 0)), mode='edge')
    acc = jnp.zeros_like(fr)
    for i in range(3):
        for j in range(3):
            acc = acc + p[i:i + H, j:j + W]
    return acc / 9.0


def render_event_frame(cam_pos, R, v_world, omega_body, pts, valid, H_out, dt, key,
                       gain, horizon_mag, salt_p, salt_val, render_h=None, blob_p=EVENT_BLOB_P,
                       channels=2, pol_hash=False, sat_knee=None, ground=None, pol_grain=None,
                       dilate_w=None):
    """Event frame at H_out, produced by rendering at `render_h` (default STEER_H) and SUM-POOLING
    -- the same generation path as the Isaac/hardware binning. Falls back to a direct render when
    H_out == render_h, so the 64x64 tasks are bit-identical to before.

    channels=1 (`gx` tasks): simulate_event_frame returns the signed NET (ON-OFF) [H,W,1]; sum-pooling
    is linear so pool(net) == net(pool) -- the binned single-channel frame is exact."""
    render_h = int(render_h or STEER_H)
    H_out = int(H_out)
    # Pass `key` STRAIGHT THROUGH on the native path. Splitting it here would shift the noise
    # stream for the 64x64 tasks too and silently break reproducibility of every existing
    # checkpoint's evals -- caught by a frame-sum regression (12567200 -> 12566002).
    if H_out == render_h:
        return EP.simulate_event_frame(
            cam_pos, R, v_world, omega_body, pts, valid, steer_focal(render_h), render_h,
            dt=dt, key=key, dr=True, gain=gain,
            horizon_mag=horizon_mag, salt_p=salt_p, salt_val=salt_val, channels=channels,
            pol_hash=pol_hash, sat_knee=sat_knee, ground=ground, pol_grain=pol_grain)
    k_ev, k_blob = jax.random.split(key, 2)
    frame = EP.simulate_event_frame(
        cam_pos, R, v_world, omega_body, pts, valid, steer_focal(render_h), render_h,
        dt=dt, key=k_ev, dr=True, gain=gain,
        horizon_mag=horizon_mag, salt_p=salt_p, salt_val=salt_val, channels=channels,
        pol_hash=pol_hash, sat_knee=sat_knee, ground=ground, pol_grain=pol_grain)
    if True:
        if dilate_w is not None:
            # contour smear (V11): dilation is linear, so smearing the signed net == smearing each
            # polarity plane separately -- no extra cancellation is introduced by doing it on net.
            w_ = dilate_w
            frame = ((1.0 - 3.0 * w_) * frame
                     + w_ * (jnp.roll(frame, 1, 0) + jnp.roll(frame, 1, 1)
                             + jnp.roll(jnp.roll(frame, 1, 0), 1, 1)))
        b = render_h // H_out
        frame = frame.reshape(H_out, b, H_out, b, channels).sum((1, 3))   # SUM-bin, as Isaac does
        if blob_p > 0.0:
            do = jax.random.uniform(k_blob, ()) < blob_p
            frame = jnp.where(do, _box_blur3(frame), frame)
    return frame


def steer_focal(H):
    """Focal length in pixels for an H x H event frame at STEER_FOV."""
    return EP.focal_from_fov(STEER_FOV, int(H))


# The gain depends on the INTEGRATION WINDOW as well as the resolution: streak length (and hence
# both magnitude and the number of lit pixels) scales with dt_evt. Round 8 introduced a 0.10 s
# window for the ~10 Hz board rate, which needs a much lower gain than the 0.03 s window.
# Calibrated by calibration/calibrate_event_gain.py to match the SATURATION FRACTION:
#   (32, 0.03) gain 1.00 -> median  30, sat 0.79%   (33 Hz profile)
#   (32, 0.10) gain 0.15 -> median  32, sat 0.83%   (10 Hz profile; Isaac --calib-target 32)
EVENT_GAIN_BY_H_DT = {(32, 0.10): 0.15}


def steer_gain(H, dt_evt=None):
    """Event magnitude gain for resolution H and integration window dt_evt.

    Falls back to the resolution-only table when the (H, dt) pair is not listed, so existing
    callers and the 0.03 s profiles are unaffected."""
    if dt_evt is not None:
        key = (int(H), round(float(dt_evt), 3))
        if key in EVENT_GAIN_BY_H_DT:
            return float(EVENT_GAIN_BY_H_DT[key])
    return float(EVENT_GAIN_BY_H.get(int(H), EVENT_GAIN))
STEER_DT_EVT = 0.045       # event INTEGRATION window (decoupled from control dt; ~real sensor)
STEER_NV, STEER_NAZ = 12, 6   # pole edge sampling (dense enough for clean looming)


def _yaw_of(q):
    w, x, y, z = q
    return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_quat(psi):
    return jnp.array([jnp.cos(psi / 2), 0.0, 0.0, jnp.sin(psi / 2)])


def make_steer(n_obs=4, x0=0.0, x_goal=6.0, half_w=1.3, obs_r=0.16, z=1.2, z_top=3.0,
               v0=STEER_V0, yawrate_max=STEER_YAWRATE_MAX, kyaw=STEER_KYAW, dt_evt=STEER_DT_EVT,
               walls=False, wall_nx=None, wall_nz=None):
    """Narrow corridor of length (x_goal-x0), half-width half_w; n_obs sparse poles.

    `walls=True` (task `steer5`) renders the corridor boundary as event edges. steer4 had NO wall
    points, and the Isaac closed-loop eval scored it **0/30** on real events: textured side walls
    read as a giant obstacle -> avoid-turn death spiral into out-of-bounds. Making the boundary
    VISIBLE (with density DR, see sample_steer_obs_dr) is the fix."""
    goal = jnp.array([[x_goal, 0.0, z]], float)
    if walls:
        nx = STEER_WALL_NX if wall_nx is None else wall_nx
        nz = STEER_WALL_NZ if wall_nz is None else wall_nz
        wall_pts = jnp.concatenate([EP.wall_edge_points(-half_w, x0, x_goal, z_top, nx=nx, nz=nz),
                                    EP.wall_edge_points(+half_w, x0, x_goal, z_top, nx=nx, nz=nz)], 0)
    else:
        wall_pts = jnp.zeros((0, 3), float)
    return dict(n_obs=int(n_obs), x0=jnp.float32(x0), x_goal=jnp.float32(x_goal),
                half_w=jnp.float32(half_w), obs_r=jnp.float32(obs_r),
                z=jnp.float32(z), z_top=jnp.float32(z_top),
                v0=jnp.float32(v0), yawrate_max=jnp.float32(yawrate_max),
                kyaw=jnp.float32(kyaw), dt_evt=jnp.float32(dt_evt),
                wall_pts=wall_pts, pc=goal,
                det_cls=jnp.zeros((0,), jnp.int32), det_corners=jnp.zeros((0, 8, 3), float))


def sample_steer_obs(key, course, n_obs):
    """Sparse poles: one per x-band (met one at a time), jittered in y within the narrow corridor.
    Returns (xy [N,2], r [N])."""
    k1, k2, k3 = jax.random.split(key, 3)
    xlo, xhi = course['x0'] + 1.2, course['x_goal'] - 0.8
    band = (xhi - xlo) / n_obs
    centres = xlo + band * (jnp.arange(n_obs) + 0.5)
    xs = centres + band * jax.random.uniform(k1, (n_obs,), minval=-0.22, maxval=0.22)
    ymax = course['half_w'] - 0.35
    ys = jax.random.uniform(k2, (n_obs,), minval=-ymax, maxval=ymax)
    r = course['obs_r'] * jax.random.uniform(k3, (n_obs,), minval=0.8, maxval=1.3)
    return jnp.stack([xs, ys], -1), r


def steer_edge_points(obs_xy, obs_r, z_top):
    """Full-height poles -> [N*NAZ*NV, 3] world edge points for the event proxy."""
    h = z_top * jnp.ones(obs_xy.shape[0])
    return EP.cylinder_edge_points(obs_xy, obs_r, h, nv=STEER_NV, naz=STEER_NAZ)


def reset_steer(key, course):
    k1, k2 = jax.random.split(key, 2)
    y0 = jax.random.uniform(k1, (), minval=-course['half_w'] * 0.3, maxval=course['half_w'] * 0.3)
    psi0 = jax.random.uniform(k2, (), minval=-0.15, maxval=0.15)     # roughly facing +x (down course)
    p = jnp.array([course['x0'], y0, course['z']])
    v = course['v0'] * jnp.array([jnp.cos(psi0), jnp.sin(psi0), 0.0])
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    return CState(p=p, v=v, q=_yaw_quat(psi0), w=jnp.zeros(3), Om=jnp.zeros(4),
                  prev_dist=jnp.linalg.norm(p - goal), target=jnp.array(0, jnp.int32))


def steer_proprio(s, course):
    """Egocentric GOAL BEARING (unit vector to goal in body frame) + normalised yaw rate. [3]"""
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    gv = goal[:2] - s.p[:2]
    ang = jnp.arctan2(gv[1], gv[0]) - _yaw_of(s.q)
    return jnp.array([jnp.cos(ang), jnp.sin(ang), s.w[2] / course['yawrate_max']])


def observe_priv_steer(s, course, obs_xy, obs_r):
    """Privileged state for the Informed decoder: pos, vel, goal-vec, heading, nearest-obstacle, clearance."""
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    gv = goal - s.p
    psi = _yaw_of(s.q)
    dxy = jnp.linalg.norm(s.p[:2][None, :] - obs_xy, axis=-1) - obs_r
    j = jnp.argmin(dxy)
    nearest = obs_xy[j] - s.p[:2]
    pri = jnp.concatenate([s.p, s.v, gv, jnp.array([jnp.cos(psi), jnp.sin(psi)]),
                           nearest, jnp.array([jnp.min(dxy), s.w[2]])])   # 3+3+3+2+2+2 = 15
    return pri


# wall descriptor = [wall_x, win_y0, win_y1, win_hw, win_z0, win_z1]; inert wall sits far at x=1e9.
INERT_WALL = jnp.array([1e9, 0.0, 0.0, 0.1, 0.0, 0.0], float)
# Per-tick event-noise vector (horizon_mag, salt_p, salt_val) for NON-DR tasks: these are exactly
# `EP.simulate_event_frame`'s own defaults, so steer4/forest12 frames stay bit-identical.
INERT_DR = jnp.array([0.0, 0.0012, 1.0], float)


def _wall_hit(s_prev, p_n, wall):
    """True if the drone crossed the cross-wall plane this step OUTSIDE both window openings."""
    crossed = (s_prev.p[0] <= wall[0]) & (p_n[0] > wall[0])
    in_win = (((jnp.abs(p_n[1] - wall[1]) < wall[3]) | (jnp.abs(p_n[1] - wall[2]) < wall[3]))
              & (p_n[2] > wall[4]) & (p_n[2] < wall[5]))
    return crossed & (~in_win)


SC_ALIGN_K = 0.5        # potential-form: gradient toward goal-ahead (gx-6b)
SC_ALIGN_CLEAR = 3.0
SC_CLEAR_K = 2.0        # potential-based clearance shaping (telescopes; unfarmable)
SC_CLEAR_D0 = 0.8


def _clear_phi(p, obs_xy, obs_r):
    d = jnp.min(jnp.linalg.norm(p[:2][None, :] - obs_xy, axis=-1) - obs_r)
    return -SC_CLEAR_K * jnp.maximum(SC_CLEAR_D0 - d, 0.0) ** 2, d


def _wrap_pi(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _reward_steer(s_prev, p_n, course, obs_xy, obs_r, wall=INERT_WALL, k_prog=5.0,
                  k_crash=10.0, psi_n=None):
    xg = wall[7] if wall.shape[0] >= 8 else course['x_goal']            # 'sc': per-episode length
    zg = wall[8] if wall.shape[0] >= 9 else course['z']
    goal = jnp.stack([xg, jnp.float32(0.0), zg])
    dist = jnp.linalg.norm(p_n - goal)
    r_prog = s_prev.prev_dist - dist                                    # bounded distance progress
    phi_n, d_near = _clear_phi(p_n, obs_xy, obs_r)
    phi_p, _ = _clear_phi(s_prev.p, obs_xy, obs_r)
    collided = d_near < 0.10
    wall_hit = _wall_hit(s_prev, p_n, wall)                             # hit the cross-wall (missed windows)
    hw = wall[6] if wall.shape[0] >= 7 else course['half_w']            # 'sc': per-episode arena
    oob = jnp.abs(p_n[1]) > hw                                          # left the arena
    if course.get('point_goal', False):
        # square-room (2026-08-26): the goal is the POINT (xg, 0, z), not the line x=xg.
        # `dist` above is already distance to that point, so progress telescopes unchanged.
        reached = dist < course.get('goal_r', SQROOM_GOAL_R)
        oob = oob | (p_n[0] > xg + 1.0) | (p_n[0] < course['x0'] - 1.0)
    else:
        reached = p_n[0] > xg
    reward = (k_prog * r_prog + (phi_n - phi_p)
              - k_crash * (collided | oob | wall_hit).astype(jnp.float32)
              + 30.0 * reached.astype(jnp.float32))
    if psi_n is not None:
        # POTENTIAL-form gated alignment (gx-6b): Phi = -k*(1-cos err)*gate;
        # the difference telescopes: transient cost for turning away from the
        # bearing in clear space, reward for turning back, no steady-state
        # pressure. The clearance gate keeps it out of avoidance's way.
        def _phi_b(p, psi):
            brg = jnp.arctan2(goal[1] - p[1], goal[0] - p[0])
            dd = jnp.min(jnp.linalg.norm(p[:2][None, :] - obs_xy, axis=-1) - obs_r)
            gate = (dd > SC_ALIGN_CLEAR).astype(jnp.float32)
            return -SC_ALIGN_K * (1.0 - jnp.cos(brg - psi)) * gate
        reward = reward + _phi_b(p_n, psi_n) - _phi_b(s_prev.p, _yaw_of(s_prev.q))
    if course.get('head_pen', False) and psi_n is not None:
        # GATED DEADZONE heading penalty (user request 2026-08-26): a small DIRECT penalty only
        # when pointing > 90 deg away from the goal bearing AND no obstacle within 3 m. The
        # deadzone + clearance gate keep it from taxing avoidance (the historical penalty trap);
        # inside the deadzone the potential alignment term above remains the only pressure.
        brg_err = jnp.abs(_wrap_pi(jnp.arctan2(goal[1] - p_n[1], goal[0] - p_n[0]) - psi_n))
        dd_hp = jnp.min(jnp.linalg.norm(p_n[:2][None, :] - obs_xy, axis=-1) - obs_r)
        pen = HEAD_PEN_K * jnp.maximum(brg_err - HEAD_PEN_DEAD, 0.0) \
            * (dd_hp > SC_ALIGN_CLEAR).astype(jnp.float32)
        reward = reward - pen
    done = collided | oob | wall_hit | reached
    return reward, dist, done, reached


def step_steer(key, s: CState, a, course, obs_xy, obs_r, wall=INERT_WALL):
    """a: 2-D unit-circle heading (egocentric). Yaw toward it (rate-limited), fly forward, fixed z."""
    psi = _yaw_of(s.q)
    rel = jnp.arctan2(a[1], a[0])                                       # desired heading CHANGE
    yaw_rate = jnp.clip(course['kyaw'] * rel, -course['yawrate_max'], course['yawrate_max'])
    psi_n = psi + yaw_rate * STEER_DT
    fwd = jnp.array([jnp.cos(psi_n), jnp.sin(psi_n), 0.0])
    p_n = (s.p + course['v0'] * fwd * STEER_DT).at[2].set(s.p[2])  # lock height
    v_n = course['v0'] * fwd
    reward, dist, done, reached = _reward_steer(s, p_n, course, obs_xy, obs_r, wall)
    s_next = CState(p_n, v_n, _yaw_quat(psi_n), jnp.array([0.0, 0.0, yaw_rate]), s.Om, dist, s.target)
    return s_next, reward, done, {"passed": reached, "all_done": reached, "target": s.target}


# ---- FOREST = denser/varied steer task: poles + trees + BRANCHES + visible WALLS, long+fast ----
# Same control/obs/reward as steer (reuses reset_steer/step_steer/_reward_steer), but a richer scene:
# n_obs varied-radius trees each with 2 branches, a 15 m corridor at 3 m/s, and the corridor WALLS
# rendered as event edges (hallway flow) so the boundary is VISIBLE, not just an invisible penalty.
FOREST_V0 = 3.0
FOREST_YAWRATE_MAX = 3.5
FOREST_NV, FOREST_NAZ, FOREST_NBR = 8, 4, 2      # trunk verticals x azimuths, branches per tree

# ---- REAL-EVENT NUISANCE DR (2026-08-06; tasks `steer5`/`forest13`, see NEXT_TRAINING.md) ----
# The Isaac closed-loop eval on physically-rendered events (robosim EVIS warp) showed the
# proxy-trained policies fail on EVENT SOURCES THE PROXY NEVER RENDERS: trunk ground shadows
# (removing them alone took steer4 from 0% -> 23%), horizon-line flicker, dense textured-wall
# event mass, and RTX background count noise. These constants parameterise adding all four as
# cheap analytic DR. Old tasks (`steer4`/`forest12`) do NOT use any of this and stay bit-identical.
STEER_WALL_NX, STEER_WALL_NZ = 120, 6            # corridor-wall grid; density DR masks it down
FOREST_WALL_NX, FOREST_WALL_NZ = 120, 6          # raised from 40x3 so keep_frac=1.0 is REALLY dense
SHADOW_N = 6                                     # ground-shadow streak points per trunk
DR_WALL_KEEP = (0.15, 1.0)                       # wall-point keep fraction (sparse .. texture-dense)
DR_SUN_ELEV = (0.45, 1.1)                        # sun elevation rad (~25-63 deg)
DR_P_SHADOW = 0.7                                # 30% of episodes are overcast (no shadows)
DR_HORIZON_MAG = (0.0, 2.5)                      # horizon-band flicker magnitude
DR_SALT_P = (0.001, 0.006)                       # background count-noise rate
DR_P_SALT_HI = 0.3                               # 30% of episodes get the 2.0 noise amplitude
# Trunk-density DR: build at the DENSE end and mask down. The keep_frac LOWER bound reproduces the
# OLD point count exactly, so old-style sparse frames stay in-distribution (handoff requirement).
STEER_NV_DR, STEER_NAZ_DR = 16, 8
FOREST_NV_DR, FOREST_NAZ_DR = 16, 8
STEER_TRUNK_KEEP_MIN = (STEER_NV * STEER_NAZ) / float(STEER_NV_DR * STEER_NAZ_DR)      # 72/128
FOREST_TRUNK_KEEP_MIN = (FOREST_NV * FOREST_NAZ) / float(FOREST_NV_DR * FOREST_NAZ_DR)  # 32/128

# ---- ROUND-2 DR REBALANCE (2026-08-07, NEXT_TRAINING.md "Round 2 directives") ----
# Round 1 (tasks steer5/forest13) fixed what it aimed at — out-of-bounds failures went to 0/30 and
# steer5 went 0% -> 47% on Isaac — but introduced a NEW dominant failure: blind frontal collisions
# with THICK trunks (forest13 13/30, steer5 16/30; collided-tree radius 0.23 vs task mean 0.18).
# Read: heavy clutter DR taught "large diffuse event mass is safe", so wide near trunks now
# pattern-match to clutter. The fix is DENSITY REBALANCE toward obstacle salience, NOT a collision
# penalty (penalties suppress capability — project history, §0.9b B).
#   - trunks: raise the keep FLOOR so trunks render as solid bands like the real stream
#   - walls:  cap keep_frac so wall mass can no longer dominate a frame
#   - shadows: halve presence
#   - horizon / salt: unchanged (they were not implicated)
# Kept SEPARATE from the round-1 constants above so steer5/forest13 stay exactly reproducible.
DR_WALL_KEEP_R2 = (0.15, 0.6)
DR_P_SHADOW_R2 = 0.5
TRUNK_KEEP_R2 = (0.6, 1.0)


def sample_event_dr(key):
    """Per-episode PER-TICK event-noise params -> [3] = (horizon_mag, salt_p, salt_val).
    Threaded into `EP.simulate_event_frame` every step (the scene-level DR — shadows and
    point density — is baked into pts/valid at episode reset instead)."""
    k_hm, k_sp, k_sv = jax.random.split(key, 3)
    horizon_mag = jax.random.uniform(k_hm, (), minval=DR_HORIZON_MAG[0], maxval=DR_HORIZON_MAG[1])
    salt_p = jax.random.uniform(k_sp, (), minval=DR_SALT_P[0], maxval=DR_SALT_P[1])
    salt_val = jnp.where(jax.random.uniform(k_sv, ()) < DR_P_SALT_HI, 2.0, 1.0)
    return jnp.stack([horizon_mag, salt_p, salt_val])


# ---- CAMERA-TILT DR (2026-08-18; task token 'tilt', NEXT_TRAINING round 8.1) ----
# A real quad pitches into acceleration (a = g*tan(theta): ~17 deg at 3 m/s^2) and rolls
# into turns, and the body-mounted camera tilts with it; the kinematic sim never does, so
# policies have never seen a pitched/rolled event frame. Per-episode static offsets are
# the cheapest analytic stand-in (same spirit as the reset-yaw DR that fixed recovery).
TILT_PITCH_MAX = 0.35      # |pitch| <= ~20 deg, nose-down positive
TILT_ROLL_MAX = 0.26       # |roll|  <= ~15 deg


def tilt_R(pitch, roll):
    """Body-frame tilt rotation: pitch about body y (nose-down positive), roll about
    body x. Compose as R_cam = R_yaw @ tilt_R(...) (post-multiply = body frame)."""
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    Rp = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rr = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rp @ Rr


def sample_tilt_dr(key):
    """Per-episode static camera tilt -> [2] = (pitch, roll)."""
    kp, kr = jax.random.split(key)
    pitch = jax.random.uniform(kp, (), minval=-TILT_PITCH_MAX, maxval=TILT_PITCH_MAX)
    roll = jax.random.uniform(kr, (), minval=-TILT_ROLL_MAX, maxval=TILT_ROLL_MAX)
    return jnp.stack([pitch, roll])


def _shadow_and_density_dr(key, xy, r, h, wall_pts, trunk_keep_min,
                           wall_keep=DR_WALL_KEEP, p_shadow=DR_P_SHADOW, trunk_keep_max=1.0):
    """Shared scene-level nuisance DR: ground-shadow streaks (gated by an overcast draw) and a
    density mask for the corridor-wall points. Returns (shadows, shadow_valid, wall_valid,
    trunk_keep) — `trunk_keep` is the keep fraction for the caller's dense trunk grid.

    Defaults = the ROUND-1 profile (steer5/forest13, reproducible). Pass the *_R2 constants for
    the round-2 rebalance (see the DR_*_R2 block above)."""
    k_sun, k_el, k_sh, k_wk, k_wd, k_tk = jax.random.split(key, 6)
    sun_az = jax.random.uniform(k_sun, (), minval=0.0, maxval=2 * jnp.pi)
    sun_elev = jax.random.uniform(k_el, (), minval=DR_SUN_ELEV[0], maxval=DR_SUN_ELEV[1])
    shadows = EP.shadow_edge_points(xy, r, h, sun_az, sun_elev, n=SHADOW_N)
    has_shadow = jax.random.uniform(k_sh, ()) < p_shadow              # overcast episodes: none
    shadow_valid = jnp.full(shadows.shape[0], True) & has_shadow
    keep = jax.random.uniform(k_wk, (), minval=wall_keep[0], maxval=wall_keep[1])
    wall_valid = EP.density_mask(k_wd, wall_pts.shape[0], keep)
    trunk_keep = jax.random.uniform(k_tk, (), minval=trunk_keep_min, maxval=trunk_keep_max)
    return shadows, shadow_valid, wall_valid, trunk_keep


WALL_NY, WALL_NZ = 44, 18                         # cross-wall panel grid (dense enough to occlude)


def wall_panel_points(wall_x, wy0, wy1, win_hw, win_z0, win_z1, half_w, z_top):
    """A solid cross-corridor PANEL at x=wall_x with TWO window openings -> edge points + a `solid`
    mask (True where the panel is solid, i.e. NOT a window). Rendered as events => looks like a wall
    with two holes."""
    ys = jnp.linspace(-half_w, half_w, WALL_NY)
    zs = jnp.linspace(0.1, z_top, WALL_NZ)
    Y = jnp.repeat(ys, WALL_NZ)
    Z = jnp.tile(zs, WALL_NY)
    X = jnp.full(Y.shape, wall_x)
    in_win = (((jnp.abs(Y - wy0) < win_hw) | (jnp.abs(Y - wy1) < win_hw))
              & (Z > win_z0) & (Z < win_z1))
    return jnp.stack([X, Y, Z], -1), (~in_win)   # pts [WALL_NY*WALL_NZ,3], solid mask


def make_forest(n_obs=12, x0=0.0, x_goal=15.0, half_w=2.0, z=1.2, z_top=3.0, obs_r=0.18,
                wall_nx=40, wall_nz=3, p_wall=0.4):
    goal = jnp.array([[x_goal, 0.0, z]], float)
    wl = EP.wall_edge_points(-half_w, x0, x_goal, z_top, nx=wall_nx, nz=wall_nz)
    wr = EP.wall_edge_points(+half_w, x0, x_goal, z_top, nx=wall_nx, nz=wall_nz)
    wall_pts = jnp.concatenate([wl, wr], 0)
    return dict(n_obs=int(n_obs), x0=jnp.float32(x0), x_goal=jnp.float32(x_goal),
                half_w=jnp.float32(half_w), obs_r=jnp.float32(obs_r),
                z=jnp.float32(z), z_top=jnp.float32(z_top),
                v0=jnp.float32(FOREST_V0), yawrate_max=jnp.float32(FOREST_YAWRATE_MAX),
                kyaw=jnp.float32(STEER_KYAW), dt_evt=jnp.float32(0.03),
                wall_pts=wall_pts, p_wall=jnp.float32(p_wall), pc=goal,
                det_cls=jnp.zeros((0,), jnp.int32), det_corners=jnp.zeros((0, 8, 3), float))


def _forest_scene(key, course, n_obs, nv=FOREST_NV, naz=FOREST_NAZ):
    """Shared forest scene sampler (see sample_forest_obs / sample_forest_obs_dr).
    The RNG split structure is FIXED so `sample_forest_obs(key, ...)` stays bit-identical to the
    pre-DR version (old checkpoints/evals reproduce exactly). `nv`/`naz` set the trunk sampling
    density — the DR variant builds at the DENSE end and masks down (see FOREST_TRUNK_KEEP_MIN).
    Returns (xy, r, h, trunks, branches, wp, wall_valid, wall)."""
    k1, k2, k3, k4, kb1, kb2, kb3, kw, kwx, kwy = jax.random.split(key, 10)
    xlo, xhi = course['x0'] + 1.5, course['x_goal'] - 1.0
    band = (xhi - xlo) / n_obs
    centres = xlo + band * (jnp.arange(n_obs) + 0.5)
    xs = centres + band * jax.random.uniform(k1, (n_obs,), minval=-0.35, maxval=0.35)
    ymax = course['half_w'] - 0.3
    ys = jax.random.uniform(k2, (n_obs,), minval=-ymax, maxval=ymax)
    r = jax.random.uniform(k3, (n_obs,), minval=0.06, maxval=0.30)          # poles .. thick trees
    h_lo4, h_hi4 = course.get('h_range', (1.2, None))
    h = jax.random.uniform(k4, (n_obs,), minval=h_lo4,
                           maxval=(h_hi4 if h_hi4 is not None else course['z_top']))
    xy = jnp.stack([xs, ys], -1)
    trunks = EP.cylinder_edge_points(xy, r, h, nv=nv, naz=naz)               # [N*nv*naz, 3]
    # branches: FOREST_NBR horizontal segments per tree at random height/azimuth/length
    N = n_obs
    ang = jax.random.uniform(kb1, (N, FOREST_NBR), minval=0.0, maxval=2 * jnp.pi)
    blen = jax.random.uniform(kb2, (N, FOREST_NBR), minval=0.0, maxval=0.6)   # some ~0 = no branch
    bz = jax.random.uniform(kb3, (N, FOREST_NBR), minval=0.45, maxval=0.95) * h[:, None]
    cx = jnp.broadcast_to(xy[:, 0:1], (N, FOREST_NBR))
    cy = jnp.broadcast_to(xy[:, 1:2], (N, FOREST_NBR))
    P0 = jnp.stack([cx, cy, bz], -1)                                         # [N,NBR,3]
    P1 = P0 + jnp.stack([blen * jnp.cos(ang), blen * jnp.sin(ang), jnp.zeros((N, FOREST_NBR))], -1)
    branches = EP.segment_edge_points(P0.reshape(-1, 3), P1.reshape(-1, 3), n=6)  # [N*NBR*6, 3]

    # ---- occasional CROSS-WALL with two windows ----
    has_wall = jax.random.uniform(kw, ()) < course['p_wall']
    wall_x = jax.random.uniform(kwx, (), minval=course['x0'] + 4.0, maxval=course['x_goal'] - 3.0)
    ymax = course['half_w'] - 0.55
    kya, kyb = jax.random.split(kwy, 2)
    wy0 = jax.random.uniform(kya, (), minval=-ymax, maxval=-0.15)          # one window on each side
    wy1 = jax.random.uniform(kyb, (), minval=0.15, maxval=ymax)
    win_hw, win_z0, win_z1 = jnp.float32(0.45), jnp.float32(0.4), jnp.float32(2.2)  # 0.9m opening, tall
    wall_x_eff = jnp.where(has_wall, wall_x, jnp.float32(1e9))
    wall = jnp.array([wall_x_eff, wy0, wy1, win_hw, win_z0, win_z1])
    wp, solid = wall_panel_points(wall_x, wy0, wy1, win_hw, win_z0, win_z1,
                                  course['half_w'], course['z_top'])
    wall_valid = solid & has_wall                                          # panel visible only if it exists
    return xy, r, h, trunks, branches, wp, wall_valid, wall


def sample_forest_obs(key, course, n_obs):
    """Denser, VARIED obstacles + an OCCASIONAL cross-corridor WALL with two windows.
    Returns (xy [N,2], r [N], pts [M,3], valid [M] bool, wall [6]). pts = trunks + branches +
    corridor walls + wall-panel; valid masks the panel out when there's no wall this episode."""
    xy, r, h, trunks, branches, wp, wall_valid, wall = _forest_scene(key, course, n_obs)
    pts = jnp.concatenate([trunks, branches, course['wall_pts'], wp], 0)
    n_fixed = trunks.shape[0] + branches.shape[0] + course['wall_pts'].shape[0]
    valid = jnp.concatenate([jnp.ones(n_fixed, bool), wall_valid], 0)
    return xy, r, pts, valid, wall


# --------- nuisance-DR variants (tasks `steer5` / `forest13`) — see the DR block above ---------
def sample_steer_obs_dr(key, course, n_obs):
    """`steer5` scene: sparse poles + ground shadows + VISIBLE corridor walls (density-DR'd) +
    trunk-density DR. Returns (xy, r, pts, valid, wall, drp) — drp is the [3] per-tick event-noise
    vector for `EP.simulate_event_frame`."""
    k_obs, k_dr, k_td, k_ev = jax.random.split(key, 4)
    xy, r = sample_steer_obs(k_obs, course, n_obs)
    h = course['z_top'] * jnp.ones(n_obs)
    trunks = EP.cylinder_edge_points(xy, r, h, nv=STEER_NV_DR, naz=STEER_NAZ_DR)
    shadows, shadow_valid, wall_valid, trunk_keep = _shadow_and_density_dr(
        k_dr, xy, r, h, course['wall_pts'], STEER_TRUNK_KEEP_MIN)
    trunk_valid = EP.density_mask(k_td, trunks.shape[0], trunk_keep)
    pts = jnp.concatenate([trunks, shadows, course['wall_pts']], 0)
    valid = jnp.concatenate([trunk_valid, shadow_valid, wall_valid], 0)
    return xy, r, pts, valid, INERT_WALL, sample_event_dr(k_ev)


def sample_forest_obs_dr(key, course, n_obs):
    """`forest13` scene: the forest12 scene (trees + branches + corridor walls + occasional
    two-window cross-wall) PLUS ground shadows, wall-density DR and trunk-density DR.
    Returns (xy, r, pts, valid, wall, drp)."""
    k_base, k_dr, k_td, k_ev = jax.random.split(key, 4)
    xy, r, h, trunks, branches, wp, panel_valid, wall = _forest_scene(
        k_base, course, n_obs, nv=FOREST_NV_DR, naz=FOREST_NAZ_DR)
    shadows, shadow_valid, wall_valid, trunk_keep = _shadow_and_density_dr(
        k_dr, xy, r, h, course['wall_pts'], FOREST_TRUNK_KEEP_MIN)
    trunk_valid = EP.density_mask(k_td, trunks.shape[0], trunk_keep)
    pts = jnp.concatenate([trunks, branches, shadows, course['wall_pts'], wp], 0)
    valid = jnp.concatenate([trunk_valid, jnp.ones(branches.shape[0], bool),
                             shadow_valid, wall_valid, panel_valid], 0)
    return xy, r, pts, valid, wall, sample_event_dr(k_ev)


# ============================ SPEEDFOREST — VARIABLE SPEED (round 3) ============================
# "Fast when free, slow when dense" (user-directed 2026-08-07). At 2x density BOTH fixed-speed
# models collapse to ~10-13%: at a fixed 3 m/s the turn radius v/yawrate_max = 0.86 m against ~1 m
# lateral gaps, i.e. ~0.17 s per tree encounter. Speed as an ACTION should fix dense forests AND
# allow >3 m/s in the open. Physics helps: event rate and looming scale with v/d, so slowing down
# literally extends the sensory reaction distance — the sensor tells the policy when it is going
# too fast for the clutter.
# vmin RAISED 1.0 -> 1.5 (2026-08-08): events need motion, and ~1 m/s starves the event stream.
# NOTE the Isaac eval server defaults to --vmin 1.0 — pass --vmin 1.5 to match this.
SPEED_VMIN, SPEED_VMAX = 1.5, 6.0
SPEED_AMAX = 3.0                       # m/s^2 accel clamp (speed is an INTEGRATOR, not a filter)
SPEED_ZONES = 3
SPEED_ZONE_FACTORS = (0.3, 1.0, 2.0)   # per-zone density multipliers, SHUFFLED per episode
SPEED_X_GOAL = 22.0                    # 22 m / 900 steps -> must average >2.4 m/s (timeout is the
SPEED_MAX_STEPS = 900                  # only time pressure; no new reward shaping — it gets hacked)


def make_speedforest(n_obs=12, x0=0.0, x_goal=SPEED_X_GOAL, half_w=2.0, z=1.2, z_top=3.0,
                     vmin=SPEED_VMIN, vmax=SPEED_VMAX, a_max=SPEED_AMAX, p_wall=0.1):
    """Variable-speed forest. Action is [3]: a[0:2] egocentric heading EXACTLY as before (that
    path is deliberately untouched — cf. the action-low-pass failure), a[2] -> commanded speed."""
    goal = jnp.array([[x_goal, 0.0, z]], float)
    wl = EP.wall_edge_points(-half_w, x0, x_goal, z_top, nx=FOREST_WALL_NX, nz=FOREST_WALL_NZ)
    wr = EP.wall_edge_points(+half_w, x0, x_goal, z_top, nx=FOREST_WALL_NX, nz=FOREST_WALL_NZ)
    return dict(n_obs=int(n_obs), x0=jnp.float32(x0), x_goal=jnp.float32(x_goal),
                half_w=jnp.float32(half_w), obs_r=jnp.float32(0.18),
                z=jnp.float32(z), z_top=jnp.float32(z_top),
                v0=jnp.float32(vmin),                      # reset_steer starts us at vmin
                vmin=jnp.float32(vmin), vmax=jnp.float32(vmax), a_max=jnp.float32(a_max),
                yawrate_max=jnp.float32(FOREST_YAWRATE_MAX),
                kyaw=jnp.float32(STEER_KYAW), dt_evt=jnp.float32(0.03),
                wall_pts=jnp.concatenate([wl, wr], 0), p_wall=jnp.float32(p_wall), pc=goal,
                det_cls=jnp.zeros((0,), jnp.int32), det_corners=jnp.zeros((0, 8, 3), float))


# ===================== ROUND 4 (2026-08-08): ZONE REWORK — `magzone` / `speedzone` =====================
# Round 3's zoned course FAILED the Isaac gate (uniform-24 dense: 10% vs the >=40% target) and showed
# NO modulation (5.67/5.43/5.37 m/s across x0.3/x1/x2 zones). Diagnosed cause: round 3 varied density
# by scaling each zone's WIDTH as 1/factor to keep jit shapes static, so the x2 zone was only ~2 m =
# 0.37 s of exposure at 5.5 m/s and killed just 1/30 episodes. Braking's reward edge was far too small
# to find. Slow flight was REACHABLE (random v0) but never NECESSARY.
#
# Round 4 decouples the two: EQUAL zone widths (~6.5 m each = seconds of exposure), density set by the
# TREE COUNT per zone. Static-shape trick: the per-zone counts are a FIXED multiset computed at course
# construction; the per-episode shuffle only permutes WHICH zone gets WHICH count, so every array shape
# stays constant while the spatial density genuinely changes.
SPEED_ZONE_FACTORS_R4 = (0.3, 1.0, 3.5)      # dense end pushed to x3-x4 per directive 2


def _zone_counts(n_obs, factors=SPEED_ZONE_FACTORS_R4):
    """Split n_obs trees across the zones in proportion to `factors` (largest-remainder), with at
    least 1 tree per zone. Pure python at construction time => a STATIC tuple for jit.

    n_obs must be >= len(factors): every zone needs at least one tree (a zero-count zone would
    divide by zero when banding trees across its width). Below that the min-1 clamp cannot be
    reconciled with the total, which previously spun forever in the remainder loop."""
    Z = len(factors)
    if n_obs < Z:
        raise ValueError(f'n_obs={n_obs} must be >= number of zones ({Z})')
    w = np.asarray(factors, float)
    raw = w / w.sum() * n_obs
    cnt = np.maximum(np.floor(raw).astype(int), 1)
    # Settle the remainder: hand extras to the largest fractional parts, and claw back from the
    # largest counts when the min-1 clamp overshot. Each iteration moves cnt.sum() one step
    # toward n_obs, and n_obs >= Z guarantees a donor with count > 1 always exists while
    # cnt.sum() > n_obs -- so this terminates.
    while cnt.sum() != n_obs:
        if cnt.sum() < n_obs:
            cnt[np.argmax(raw - cnt)] += 1
        else:
            donors = [j for j in np.argsort(-cnt) if cnt[j] > 1]
            assert donors, f'no donor while reducing {cnt} toward {n_obs}'
            cnt[donors[0]] -= 1
    return tuple(int(c) for c in cnt)


# ---- SQUARE-ROOM variant (`sq` token, 2026-08-26): 22x22 arena, WHITE walls (outline only),
# trees over the whole floor with per-episode density DR, POINT goal at (x_goal, 0).
# Motivated by the real flight hall: big open room, featureless white walls, obstacles anywhere.
SQROOM_HW = 11.0                     # half-width = half-length -> square arena
SQROOM_N = 99                        # tree slots (~training density when all alive)
SQROOM_KEEP = (0.15, 1.0)            # per-episode alive fraction (decision c: density DR)
SQROOM_GOAL_R = 1.0                  # point-goal reach radius (m)
SQROOM_CLEAR = 1.3
SQROOM_NPROP = 8                     # prop collision slots: 2 screens + 2 trusses + 4 plants
HEAD_PEN_K = 0.05                    # per-tick penalty slope beyond the deadzone (rad)
HEAD_PEN_DEAD = jnp.pi / 2 if False else 1.5708   # 90 deg deadzone                   # spawn clearance: min centre-dist (1.0 + max trunk radius)


def sample_squareroom_obs(key, course, n_obs):
    """Square-room scene -> the zoneforest 6-tuple contract (xy, r, pts, valid, wall, drp).
    Walls emit ONLY their silhouette (top edge, floor junction, corner verticals) -- a real
    white wall's event signature; faces are silent. Trees fill the whole floor; a per-episode
    keep fraction sweeps density from sparse-real to dense-training."""
    (k_x, k_y, k_r, k_h, k_keep, k_kf, kc1, kc2, kc3, kc4, kc5,
     k_d1, k_d2, k_d3) = jax.random.split(key, 14)
    x0 = course['x0']; xg = course['x_goal']; hw = course['half_w']; zt = course['z_top']
    xs = jax.random.uniform(k_x, (n_obs,), minval=x0 + 1.5, maxval=xg - 0.5)
    ys = jax.random.uniform(k_y, (n_obs,), minval=-(hw - 1.0), maxval=hw - 1.0)
    xy = jnp.stack([xs, ys], -1)
    r = jax.random.uniform(k_r, (n_obs,), minval=0.06, maxval=0.30)
    h = jax.random.uniform(k_h, (n_obs,), minval=1.2, maxval=zt)
    keep = jax.random.uniform(k_kf, (), minval=course.get('keep_lo', SQROOM_KEEP[0]),
                              maxval=SQROOM_KEEP[1])
    alive = EP.density_mask(k_keep, n_obs, keep)
    alive = alive & (jnp.linalg.norm(xy - jnp.array([xg, 0.0])[None], axis=-1) > 2.2)
    trunks = EP.cylinder_edge_points(xy, r, h, nv=FOREST_NV_DR, naz=FOREST_NAZ_DR)
    t_alive = jnp.repeat(alive, FOREST_NV_DR * FOREST_NAZ_DR)

    def wline(p0, p1, z, n=90):
        t = jnp.linspace(0.0, 1.0, n)[:, None]
        xy2 = p0[None] * (1 - t) + p1[None] * t
        return jnp.concatenate([xy2, jnp.full((n, 1), z)], -1)
    xa, xb = x0 - 1.0, xg + 1.0
    L = []
    for z in (0.2, zt):
        L.append(wline(jnp.array([xa, -hw]), jnp.array([xb, -hw]), z))
        L.append(wline(jnp.array([xa,  hw]), jnp.array([xb,  hw]), z))
        L.append(wline(jnp.array([xa, -hw]), jnp.array([xa,  hw]), z))
        L.append(wline(jnp.array([xb, -hw]), jnp.array([xb,  hw]), z))
    zs = jnp.linspace(0.2, zt, 10)
    for cx, cy in ((xa, -hw), (xa, hw), (xb, -hw), (xb, hw)):
        L.append(jnp.stack([jnp.full(10, cx), jnp.full(10, cy), zs], -1))
    wallpts = jnp.concatenate(L, 0)

    clutter = jnp.stack([
        jax.random.uniform(kc1, (GX_CLUTTER_N,), minval=x0, maxval=xg),
        jax.random.uniform(kc2, (GX_CLUTTER_N,), minval=-hw, maxval=hw),
        jax.random.uniform(kc3, (GX_CLUTTER_N,), minval=GX_CLUTTER_Z[0],
                           maxval=GX_CLUTTER_Z[1])], -1)
    v_cl = EP.density_mask(kc4, GX_CLUTTER_N,
                           jax.random.uniform(kc5, (), minval=0.1, maxval=1.0))
    pts = jnp.concatenate([trunks, wallpts, clutter], 0)
    valid = jnp.concatenate([t_alive, jnp.ones(wallpts.shape[0], bool), v_cl], 0)
    prop_xy = jnp.zeros((SQROOM_NPROP, 2)); prop_r = jnp.full((SQROOM_NPROP,), -1.0)
    if course.get('props', False):
        # LAB PROPS (2026-08-31, from realworld/screen_candidate_frames.png):
        # screens (uniform surface -> BORDER only + small interior blob), hollow
        # beam/truss (thin segment lattice, background visible through), plants
        # (stem + diffuse canopy cloud; z-spread crosses 0.8 m polarity cells so
        # the foliage shows mixed signs like the real recording).
        kp = jax.random.split(jax.random.fold_in(key, 211), 10)
        def seg(p0, p1, n):
            t = jnp.linspace(0.0, 1.0, n)[:, None]
            return p0[None] * (1 - t) + p1[None] * t
        blocks = []; bvalid = []; pxy = []; pr = []
        # -- 2 screen slots (1.8x1.2 m, elevated 0.6 m, random wall-adjacent pose)
        scr_on = jax.random.bernoulli(kp[0], 0.6, (2,))
        scr_c = jnp.stack([jax.random.uniform(kp[1], (2,), minval=x0+2, maxval=xg-2),
                           jax.random.uniform(kp[2], (2,), minval=-(hw-1.5), maxval=hw-1.5)], -1)
        scr_a = jax.random.uniform(kp[3], (2,), minval=0, maxval=jnp.pi)
        for i in range(2):
            c = scr_c[i]; d = jnp.stack([jnp.cos(scr_a[i]), jnp.sin(scr_a[i])])
            w2 = 0.9
            corners = [jnp.concatenate([c - d*w2, jnp.array([0.6])]), jnp.concatenate([c + d*w2, jnp.array([0.6])]),
                       jnp.concatenate([c + d*w2, jnp.array([1.8])]), jnp.concatenate([c - d*w2, jnp.array([1.8])])]
            border = jnp.concatenate([seg(corners[0],corners[1],26), seg(corners[1],corners[2],16),
                                      seg(corners[2],corners[3],26), seg(corners[3],corners[0],16)],0)
            blob = jnp.concatenate([c[None].repeat(8,0) + jax.random.normal(jax.random.fold_in(kp[4], i),(8,2))*0.08,
                                    jnp.full((8,1),1.2) + jax.random.normal(jax.random.fold_in(kp[5], i),(8,1))*0.08],-1)
            blocks += [border, blob]; bvalid += [jnp.repeat(scr_on[i], 84+8)]
            pxy.append(c); pr.append(jnp.where(scr_on[i], 0.9, -1.0))
        # -- 2 truss slots (vertical lattice 0.6 m wide x 2.6 m tall)
        tr_on = jax.random.bernoulli(kp[6], 0.5, (2,))
        tr_c = jnp.stack([jax.random.uniform(kp[7], (2,), minval=x0+2, maxval=xg-2),
                          jax.random.uniform(kp[8], (2,), minval=-(hw-1.5), maxval=hw-1.5)], -1)
        for i in range(2):
            c = tr_c[i]
            L=[]
            for dx in (-0.3, 0.3):
                L.append(seg(jnp.array([c[0]+dx, c[1], 0.05]), jnp.array([c[0]+dx, c[1], 2.6]), 20))
            for z in (0.4, 0.95, 1.5, 2.05, 2.6):
                L.append(seg(jnp.array([c[0]-0.3, c[1], z]), jnp.array([c[0]+0.3, c[1], z]), 7))
            L.append(seg(jnp.array([c[0]-0.3, c[1], 0.4]), jnp.array([c[0]+0.3, c[1], 1.5]), 12))
            L.append(seg(jnp.array([c[0]+0.3, c[1], 1.5]), jnp.array([c[0]-0.3, c[1], 2.6]), 12))
            tp = jnp.concatenate(L,0)
            blocks.append(tp); bvalid.append(jnp.repeat(tr_on[i], tp.shape[0]))
            pxy.append(c); pr.append(jnp.where(tr_on[i], 0.35, -1.0))
        # -- 4 plant slots (stem + canopy ellipsoid cloud)
        pl_on = jax.random.bernoulli(kp[9], 0.5, (4,))
        kpl = jax.random.split(jax.random.fold_in(key, 313), 4)
        for i in range(4):
            kc1,kc2 = jax.random.split(kpl[i])
            c = jnp.stack([jax.random.uniform(kc1, (), minval=x0+2, maxval=xg-2),
                           jax.random.uniform(kc2, (), minval=-(hw-1.5), maxval=hw-1.5)])
            stem = seg(jnp.concatenate([c, jnp.array([0.05])]), jnp.concatenate([c, jnp.array([0.9])]), 10)
            kn = jax.random.fold_in(kpl[i], 7)
            can = jax.random.normal(kn, (80,3)) * jnp.array([0.35,0.35,0.45]) + jnp.concatenate([c, jnp.array([1.15])])
            keepc = EP.density_mask(jax.random.fold_in(kpl[i], 9), 80, 0.6)
            blocks += [stem, can]
            bvalid += [jnp.repeat(pl_on[i], 10), pl_on[i] & keepc]
            pxy.append(c); pr.append(jnp.where(pl_on[i], 0.45, -1.0))
        pts = jnp.concatenate([pts] + blocks, 0)
        valid = jnp.concatenate([valid] + bvalid, 0)
        prop_xy = jnp.stack(pxy); prop_r = jnp.stack(pr)
    if course.get('ceiling', False):
        # CEILING + HIGH-WALL TEXTURE DR (2026-08-26): light fixtures / beams as random straight
        # lines across the z=z_top plane (count 0..CEIL_N per episode -- some ceilings bare), and
        # sparse sign/conduit patches on the wall faces at z > 2. Ceiling lines are a strong yaw
        # reference: under pure yaw they sweep the upper FOV where the white room is otherwise silent.
        (kl1, kl2, kl3, kl4, kp1, kp2, kp3) = jax.random.split(jax.random.fold_in(key, 137), 7)
        CEIL_N, LPTS = 10, 60
        ang = jax.random.uniform(kl1, (CEIL_N,), minval=0.0, maxval=jnp.pi)
        cxm = jax.random.uniform(kl2, (CEIL_N, 2),
                                 minval=jnp.array([x0 + 2.0, -(hw - 2.0)]),
                                 maxval=jnp.array([xg - 2.0, hw - 2.0]))
        ln = jax.random.uniform(kl3, (CEIL_N,), minval=3.0, maxval=14.0)
        tline = jnp.linspace(-0.5, 0.5, LPTS)
        dirv = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
        cpts_xy = cxm[:, None, :] + dirv[:, None, :] * (ln[:, None, None] * tline[None, :, None])
        cpts = jnp.concatenate([cpts_xy.reshape(-1, 2),
                                jnp.full((CEIL_N * LPTS, 1), zt)], -1)
        n_lines = jax.random.randint(kl4, (), 0, CEIL_N + 1)
        c_alive = jnp.repeat(jnp.arange(CEIL_N) < n_lines, LPTS)
        # high-wall patches: PN slots, each a 5x3 grid on a random wall at z in [2, zt]
        PN = 8
        side = jax.random.randint(kp1, (PN,), 0, 4)
        u0 = jax.random.uniform(kp2, (PN,), minval=0.05, maxval=0.85)
        gp_u = u0[:, None] + jnp.linspace(0.0, 0.06, 5)[None, :]
        gp_z = jnp.linspace(2.0, zt - 0.1, 3)
        U = jnp.repeat(gp_u, 3, axis=1).reshape(PN, 15)
        Zp = jnp.tile(gp_z, (PN, 5))
        xa_, xb_ = x0 - 1.0, xg + 1.0
        wx = jnp.select([side[:, None] == 0, side[:, None] == 1, side[:, None] == 2, side[:, None] == 3],
                        [xa_ + U * (xb_ - xa_), xa_ + U * (xb_ - xa_),
                         jnp.full_like(U, xa_), jnp.full_like(U, xb_)])
        wy = jnp.select([side[:, None] == 0, side[:, None] == 1, side[:, None] == 2, side[:, None] == 3],
                        [jnp.full_like(U, -hw), jnp.full_like(U, hw),
                         -hw + U * 2 * hw, -hw + U * 2 * hw])
        ppts = jnp.stack([wx.reshape(-1), wy.reshape(-1), Zp.reshape(-1)], -1)
        p_alive = jnp.repeat(jax.random.uniform(kp3, (PN,)) < 0.5, 15)
        pts = jnp.concatenate([pts, cpts, ppts], 0)
        valid = jnp.concatenate([valid, c_alive, p_alive], 0)
    r_eff = jnp.where(alive, r, -1.0)
    # drp slots 3..6 = RENDER-STATISTICS DR (2026-09-01, sim2sim experiment): per-episode
    # jitter of the event RENDERER itself, recreating the multi-proxy-version diversity the
    # warm-start lineage saw. Defaults (flag off) reproduce the fixed constants exactly.
    kr = jax.random.split(jax.random.fold_in(key, 421), 4)
    if course.get('render_dr', False):
        gain_mul = jax.random.uniform(kr[0], (), minval=0.6, maxval=1.6)
        knee_mul = jax.random.uniform(kr[1], (), minval=0.6, maxval=1.4)
        pol_g = jnp.where(jax.random.uniform(kr[2], ()) < 0.2, 0.0,
                          jax.random.uniform(jax.random.fold_in(kr[2], 1), (),
                                             minval=0.4, maxval=1.5))
        dil = jnp.where(jax.random.uniform(kr[3], ()) < 0.5, GX_DILATE_W, 0.0)
    else:
        gain_mul = jnp.float32(1.0); knee_mul = jnp.float32(1.0)
        pol_g = jnp.float32(GX_POL_GRAIN); dil = jnp.float32(GX_DILATE_W)
    drp = jnp.stack([
        jax.random.uniform(k_d1, (), minval=DR_HORIZON_MAG[0], maxval=DR_HORIZON_MAG[1]),
        jax.random.uniform(k_d2, (), minval=DR_SALT_P[0], maxval=DR_SALT_P[1]),
        jnp.where(jax.random.uniform(k_d3, ()) < DR_P_SALT_HI, 2.0, 1.0),
        gain_mul, knee_mul, pol_g, dil])
    xy = jnp.concatenate([xy, prop_xy], 0)
    r_eff = jnp.concatenate([r_eff, prop_r], 0)
    return xy, r_eff, pts, valid, INERT_WALL, drp


def reset_sqroom(key, course, obs_xy):
    """Spawn anywhere in the room (decision: full-room random start), full-circle yaw,
    with a >= ~1 m clearance guarantee: draw 8 candidates, take the one with the largest
    min centre-distance to trees (conservative: SQROOM_CLEAR = 1.0 + max radius)."""
    kp, kpsi, kv = jax.random.split(key, 3)
    kx, ky = jax.random.split(kp)
    K = 8
    cx = jax.random.uniform(kx, (K,), minval=course['x0'] + 0.5,
                            maxval=course['x_goal'] - 5.0)
    cy = jax.random.uniform(ky, (K,), minval=-(course['half_w'] - 1.5),
                            maxval=course['half_w'] - 1.5)
    cand = jnp.stack([cx, cy], -1)                                   # [K,2]
    dmin = jnp.min(jnp.linalg.norm(cand[:, None, :] - obs_xy[None], axis=-1), axis=1)
    j = jnp.argmax(dmin)
    p = jnp.array([cand[j, 0], cand[j, 1], course['z']])
    psi0 = jax.random.uniform(kpsi, (), minval=-jnp.pi, maxval=jnp.pi)
    v0 = jax.random.uniform(kv, (), minval=course['vmin'], maxval=course['vmax'])
    v = v0 * jnp.array([jnp.cos(psi0), jnp.sin(psi0), 0.0])
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    return CState(p=p, v=v, q=_yaw_quat(psi0), w=jnp.zeros(3), Om=jnp.zeros(4),
                  prev_dist=jnp.linalg.norm(p - goal), target=jnp.array(0, jnp.int32))


def make_zoneforest(n_obs=18, x0=0.0, x_goal=SPEED_X_GOAL, half_w=2.0, z=1.2, z_top=3.0,
                    vmin=SPEED_VMIN, vmax=SPEED_VMAX, a_max=SPEED_AMAX):
    """Round-4 variable-speed course: EQUAL-WIDTH zones, density by tree count."""
    c = make_speedforest(n_obs=n_obs, x0=x0, x_goal=x_goal, half_w=half_w, z=z, z_top=z_top,
                         vmin=vmin, vmax=vmax, a_max=a_max)
    c['zone_counts'] = _zone_counts(n_obs)      # static tuple, e.g. 18 -> (1, 4, 13)
    return c


def sample_zoneforest_obs(key, course, n_obs, return_h=False):
    """Round-4 scene: 3 EQUAL-width zones; the zone a tree-group lands in is shuffled per episode.
    Linear density per zone = count / width, so the dense zone is genuinely dense AND long enough
    to be worth braking for. Round-2 DR profile throughout."""
    (k_perm, k_x, k_y, k_r, k_h, kb1, kb2, kb3,
     k_dr, k_td, k_ev) = jax.random.split(key, 11)
    counts = tuple(course['zone_counts'])                    # static
    Z = len(counts)
    xlo, xhi = course['x0'] + 1.5, course['x_goal'] - 1.0
    W = (xhi - xlo) / Z                                      # EQUAL widths (the round-4 fix)

    gi = np.repeat(np.arange(Z), counts)                     # static: group (=count class) per tree
    ki = np.concatenate([np.arange(c) for c in counts])      # static: index within its group
    cnt_of = np.asarray(counts, float)[gi]                   # static: how many trees share its zone

    perm = jax.random.permutation(k_perm, jnp.arange(Z))     # which zone each GROUP occupies
    z_lo = xlo + perm[jnp.asarray(gi)] * W
    band = W / jnp.asarray(cnt_of)
    xs = z_lo + band * (jnp.asarray(ki) + 0.5) + band * jax.random.uniform(
        k_x, (n_obs,), minval=-0.3, maxval=0.3)
    ymax = course['half_w'] - 0.3
    ys = jax.random.uniform(k_y, (n_obs,), minval=-ymax, maxval=ymax)
    r = jax.random.uniform(k_r, (n_obs,), minval=0.06, maxval=0.30)
    h_lo, h_hi = course.get('h_range', (1.2, None))
    h = jax.random.uniform(k_h, (n_obs,), minval=h_lo,
                           maxval=(h_hi if h_hi is not None else course['z_top']))
    xy = jnp.stack([xs, ys], -1)

    trunks = EP.cylinder_edge_points(xy, r, h, nv=FOREST_NV_DR, naz=FOREST_NAZ_DR)
    ang = jax.random.uniform(kb1, (n_obs, FOREST_NBR), minval=0.0, maxval=2 * jnp.pi)
    blen = jax.random.uniform(kb2, (n_obs, FOREST_NBR), minval=0.0, maxval=0.6)
    bz = jax.random.uniform(kb3, (n_obs, FOREST_NBR), minval=0.45, maxval=0.95) * h[:, None]
    cx = jnp.broadcast_to(xy[:, 0:1], (n_obs, FOREST_NBR))
    cy = jnp.broadcast_to(xy[:, 1:2], (n_obs, FOREST_NBR))
    P0 = jnp.stack([cx, cy, bz], -1)
    P1 = P0 + jnp.stack([blen * jnp.cos(ang), blen * jnp.sin(ang),
                         jnp.zeros((n_obs, FOREST_NBR))], -1)
    branches = EP.segment_edge_points(P0.reshape(-1, 3), P1.reshape(-1, 3), n=6)

    shadows, shadow_valid, wall_valid, trunk_keep = _shadow_and_density_dr(
        k_dr, xy, r, h, course['wall_pts'], TRUNK_KEEP_R2[0],
        wall_keep=DR_WALL_KEEP_R2, p_shadow=DR_P_SHADOW_R2, trunk_keep_max=TRUNK_KEEP_R2[1])
    trunk_valid = EP.density_mask(k_td, trunks.shape[0], trunk_keep)

    pts = jnp.concatenate([trunks, branches, shadows, course['wall_pts']], 0)
    valid = jnp.concatenate([trunk_valid, jnp.ones(branches.shape[0], bool),
                             shadow_valid, wall_valid], 0)
    if return_h:
        return xy, r, pts, valid, INERT_WALL, sample_event_dr(k_ev), h
    return xy, r, pts, valid, INERT_WALL, sample_event_dr(k_ev)


def reset_speedvar(key, course):
    """Reset with a RANDOM INITIAL SPEED v0 ~ U(vmin, vmax) — REQUIRED for both variable-speed
    variants (NEXT_TRAINING.md 2026-08-08, saturation-exploration risk #1).

    WHY (this is the fix for a MEASURED failure, not a precaution): a squashed actor's
    exploration noise is COMPRESSED at saturation. The forest13 seed emits |a| >= 1 about 80% of
    ticks, so v_cmd is vmax essentially always -> the policy never SAMPLES slow flight -> it
    never experiences that braking has value, and plateaus by steering harder at vmax instead.
    Measured on dv3_magforest1 (round-3 first attempt, without this fix): speed pinned at
    5.99 m/s, p95-p5 spread 0.06 m/s, corr(speed, local density) ~ 0 across every eval.

    Randomising the INTEGRATOR STATE sidesteps the actor entirely: the policy EXPERIENCES the
    whole speed range regardless of what it commands, so the world model learns speed-dependent
    dynamics and the actor can discover braking from data it actually visits. Structural fix —
    no reward shaping, which this project has twice found gets hacked."""
    k1, k2, k3 = jax.random.split(key, 3)
    y0 = jax.random.uniform(k1, (), minval=-course['half_w'] * 0.3, maxval=course['half_w'] * 0.3)
    psi0 = jax.random.uniform(k2, (), minval=-0.15, maxval=0.15)
    v0 = jax.random.uniform(k3, (), minval=course['vmin'], maxval=course['vmax'])
    p = jnp.array([course['x0'], y0, course['z']])
    v = v0 * jnp.array([jnp.cos(psi0), jnp.sin(psi0), 0.0])
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    return CState(p=p, v=v, q=_yaw_quat(psi0), w=jnp.zeros(3), Om=jnp.zeros(4),
                  prev_dist=jnp.linalg.norm(p - goal), target=jnp.array(0, jnp.int32))


# ===================== ROUND 7 (2026-08-17): RESET-YAW DR + FORCED ENCOUNTERS =====================
# Two separate problems, one reset function.
#
# (1) LARGE-YAW RECOVERY -- a first-order REAL-WORLD blocker. Measured on ratezone3: successful
#     episodes never leave ~+-30 deg of the corridor axis (p95 |yaw| 21-25 deg) because resets
#     sample only +-8.6 deg. Forced to psi0 = +-90 deg with a CORRECT goal bearing it recovers
#     0/10 -- it leans at ~20% authority and exits sideways; at 180 deg the bearing sine is ~0 and
#     it commits to neither side, flying backward. Physics is not the limit (turn radius at vmin
#     is 0.43 m inside a +-2 m corridor); the policy has simply never seen a large bearing error.
#     Fix: sample psi0 over the FULL circle. Zero architecture change -- it teaches "large bearing
#     error -> commit full-authority turn until it shrinks".
#
# (2) MODULATION, still missing after four attempts. Every variant converges to ~5 m/s everywhere
#     and wins by steering, because on a long approach steering usually suffices. Fix: start a
#     fraction of episodes INSIDE the densest zone already at vmax, so "arrived too fast" is
#     experienced from step one rather than only after an approach.
YAWDR_PSI_MAX = jnp.pi                 # full +-180 deg
YAWDR_P_FORCED = 0.3                   # fraction of episodes started inside the dense zone
YAWDR_FORCED_INSET = 0.15              # how far into the dense zone to drop in (fraction of width)


def reset_yawdr(key, course, obs_xy, psi_max=YAWDR_PSI_MAX, p_forced=YAWDR_P_FORCED):
    """Round-7 reset: full-circle initial yaw + occasional forced dense-zone encounter at vmax.

    `obs_xy` is the CURRENT episode's tree layout, so the forced start can be placed in whichever
    equal-width zone actually received the most trees (the zone assignment is shuffled per
    episode). Callers that do not need this may pass any [N,2] array; it is only read when the
    forced branch fires."""
    k_y, k_psi, k_v, k_f = jax.random.split(key, 4)
    y0 = jax.random.uniform(k_y, (), minval=-course['half_w'] * 0.3,
                            maxval=course['half_w'] * 0.3)
    psi0 = jax.random.uniform(k_psi, (), minval=-psi_max, maxval=psi_max)
    v_free = jax.random.uniform(k_v, (), minval=course['vmin'], maxval=course['vmax'])

    xlo = course['x0'] + 1.5
    xhi = course['x_goal'] - 1.0
    W = (xhi - xlo) / 3.0
    tx = obs_xy[:, 0]
    counts = jnp.stack([jnp.sum((tx >= xlo + k * W) & (tx < xlo + (k + 1) * W))
                        for k in range(3)])
    x_dense = xlo + jnp.argmax(counts).astype(jnp.float32) * W + YAWDR_FORCED_INSET * W

    forced = jax.random.uniform(k_f, ()) < p_forced
    x0 = jnp.where(forced, x_dense, course['x0'])
    v0 = jnp.where(forced, course['vmax'], v_free)     # arrive too fast, on purpose

    p = jnp.array([x0, y0, course['z']])
    v = v0 * jnp.array([jnp.cos(psi0), jnp.sin(psi0), 0.0])
    goal = jnp.array([course['x_goal'], 0.0, course['z']])
    return CState(p=p, v=v, q=_yaw_quat(psi0), w=jnp.zeros(3), Om=jnp.zeros(4),
                  prev_dist=jnp.linalg.norm(p - goal), target=jnp.array(0, jnp.int32))


def speed_proprio(s, course):
    """[4] = egocentric goal bearing (cos,sin) + normalised yaw rate + NORMALISED SPEED.
    Speed is appended because yaw authority depends on it (turn radius = v/yawrate_max); it is
    available on the real drone from the EKF. This is why speedforest cannot seed from forest13."""
    base = steer_proprio(s, course)                        # [3]
    spd = jnp.linalg.norm(s.v[:2]) / course['vmax']
    return jnp.concatenate([base, spd[None]])


def sample_speedforest_obs(key, course, n_obs):
    """Zoned-density forest: the corridor is split into SPEED_ZONES zones whose density factors are
    SHUFFLED per episode, so the policy must accelerate AND brake within a single flight (varying
    density only across episodes would teach one constant speed per episode instead).

    jit-safe trick: each zone holds the SAME number of trees, and the zone's x-EXTENT is scaled
    inversely to its density factor — so linear density ~ factor while every shape stays static.
    Uses the ROUND-2 rebalanced DR profile."""
    (k_perm, k_x, k_y, k_r, k_h, kb1, kb2, kb3,
     k_wall, k_dr, k_td, k_ev) = jax.random.split(key, 12)
    n_per = n_obs // SPEED_ZONES
    xlo, xhi = course['x0'] + 1.5, course['x_goal'] - 1.0

    factors = jax.random.permutation(k_perm, jnp.array(SPEED_ZONE_FACTORS))   # [3]
    inv = 1.0 / factors                                    # width ~ 1/density
    widths = inv / inv.sum() * (xhi - xlo)                 # [3], sums to the corridor span
    edges = xlo + jnp.concatenate([jnp.zeros(1), jnp.cumsum(widths)])         # [4]

    zi = jnp.repeat(jnp.arange(SPEED_ZONES), n_per)        # [n_obs] zone index per tree
    ki = jnp.tile(jnp.arange(n_per), SPEED_ZONES)          # [n_obs] index within its zone
    band = widths[zi] / n_per
    xs = edges[zi] + band * (ki + 0.5) + band * jax.random.uniform(
        k_x, (n_obs,), minval=-0.3, maxval=0.3)
    ymax = course['half_w'] - 0.3
    ys = jax.random.uniform(k_y, (n_obs,), minval=-ymax, maxval=ymax)
    r = jax.random.uniform(k_r, (n_obs,), minval=0.06, maxval=0.30)
    h_lo, h_hi = course.get('h_range', (1.2, None))
    h = jax.random.uniform(k_h, (n_obs,), minval=h_lo,
                           maxval=(h_hi if h_hi is not None else course['z_top']))
    xy = jnp.stack([xs, ys], -1)

    trunks = EP.cylinder_edge_points(xy, r, h, nv=FOREST_NV_DR, naz=FOREST_NAZ_DR)
    ang = jax.random.uniform(kb1, (n_obs, FOREST_NBR), minval=0.0, maxval=2 * jnp.pi)
    blen = jax.random.uniform(kb2, (n_obs, FOREST_NBR), minval=0.0, maxval=0.6)
    bz = jax.random.uniform(kb3, (n_obs, FOREST_NBR), minval=0.45, maxval=0.95) * h[:, None]
    cx = jnp.broadcast_to(xy[:, 0:1], (n_obs, FOREST_NBR))
    cy = jnp.broadcast_to(xy[:, 1:2], (n_obs, FOREST_NBR))
    P0 = jnp.stack([cx, cy, bz], -1)
    P1 = P0 + jnp.stack([blen * jnp.cos(ang), blen * jnp.sin(ang),
                         jnp.zeros((n_obs, FOREST_NBR))], -1)
    branches = EP.segment_edge_points(P0.reshape(-1, 3), P1.reshape(-1, 3), n=6)

    shadows, shadow_valid, wall_valid, trunk_keep = _shadow_and_density_dr(
        k_dr, xy, r, h, course['wall_pts'], TRUNK_KEEP_R2[0],
        wall_keep=DR_WALL_KEEP_R2, p_shadow=DR_P_SHADOW_R2, trunk_keep_max=TRUNK_KEEP_R2[1])
    trunk_valid = EP.density_mask(k_td, trunks.shape[0], trunk_keep)

    pts = jnp.concatenate([trunks, branches, shadows, course['wall_pts']], 0)
    valid = jnp.concatenate([trunk_valid, jnp.ones(branches.shape[0], bool),
                             shadow_valid, wall_valid], 0)
    return xy, r, pts, valid, INERT_WALL, sample_event_dr(k_ev)


# ---- Variant B: MAGFOREST — the 2-D head's DISCARDED MAGNITUDE becomes the speed command ----
# Heading is atan2(a1,a0) exactly as always (scale-invariant), so action shape stays (2,) and
# proprio stays [3] => SEEDS DIRECTLY FROM dv3_forest13 (same net shapes; N6 deploy graph
# unchanged). Two guards, both REQUIRED (NEXT_TRAINING.md round 3, Variant B):
#   - DEAD ZONE below |a|=MAG_DEADZONE: heading angular SNR degrades as 1/|a| under exploration
#     noise, and slow flight = dense clutter = exactly where heading precision matters most. The
#     dead zone puts "slow" at |a|~0.3 rather than ~0, keeping the angle usable.
#     WATCH FOR: dense-zone steering going erratic while sparse zones look fine -> that IS the
#     degeneracy this guards against; the response is to switch to Variant A (speedforest).
#   - CLIP |a| at 1 BEFORE the mapping: the [-1,1]^2 box reaches sqrt(2) on the diagonals, so
#     without this the top speed would depend on heading direction.
MAG_DEADZONE = 0.3
MAG_SPAN = 0.7                         # = 1 - MAG_DEADZONE

# Yaw-rate actuator time constant (s). ~60-100 ms mirrors the real CF attitude loop; 0.08 is the
# middle of that range. Used by step_ratezone only -- the heading-head tasks keep their original
# instantaneous rate so their checkpoints stay reproducible.
YAW_TAU = 0.08


def step_magforest(key, s: CState, a, course, obs_xy, obs_r, wall=INERT_WALL):
    """a:[2] — heading = atan2(a1,a0) (unchanged path), |a| -> commanded speed."""
    psi = _yaw_of(s.q)
    rel = jnp.arctan2(a[1], a[0])                       # heading: scale-invariant, as always
    yaw_rate = jnp.clip(course['kyaw'] * rel, -course['yawrate_max'], course['yawrate_max'])
    psi_n = psi + yaw_rate * STEER_DT
    fwd = jnp.array([jnp.cos(psi_n), jnp.sin(psi_n), 0.0])

    vmin, vmax, a_max = course['vmin'], course['vmax'], course['a_max']
    mag = jnp.clip(jnp.linalg.norm(a), 0.0, 1.0)        # clip BEFORE mapping (diagonals reach sqrt2)
    frac = jnp.clip((mag - MAG_DEADZONE) / MAG_SPAN, 0.0, 1.0)
    v_cmd = vmin + (vmax - vmin) * frac
    v_cur = jnp.linalg.norm(s.v[:2])
    dv = jnp.clip(v_cmd - v_cur, -a_max * STEER_DT, a_max * STEER_DT)
    v_spd = jnp.clip(v_cur + dv, vmin, vmax)

    p_n = (s.p + v_spd * fwd * STEER_DT).at[2].set(s.p[2])
    v_n = v_spd * fwd
    reward, dist, done, reached = _reward_steer(s, p_n, course, obs_xy, obs_r, wall)
    s_next = CState(p_n, v_n, _yaw_quat(psi_n), jnp.array([0.0, 0.0, yaw_rate]),
                    s.Om, dist, s.target)
    return s_next, reward, done, {"passed": reached, "all_done": reached, "target": s.target}


# ================= ROUND 5-REVISED (2026-08-12): DIRECT YAW-RATE HEAD — `ratezone` =================
# Recorded-output analysis of forest13 showed the trained policy is a 100 Hz DITHER controller:
# heading commands bimodal at +-120-180 deg (deep P-law saturation), lag-1 autocorr -0.73, with
# the actual steering carried in the DUTY CYCLE of hard-left/hard-right. The heading
# parametrization is being used as an awkward encoding of a rate signal, and it carries two
# pathologies:
#   * the atan2 branch cut at +-180 deg -- the policy LIVES on the discontinuity;
#   * the low-|a| angular degeneracy -- which is what sank the magnitude-speed coupling in
#     rounds 3-4 (braking required shrinking |a|, which destroyed the heading command).
# `ratezone` removes both: a0 IS the yaw rate (no atan2, no P-law) and a1 IS the speed. Two
# independent, monotone channels with no degeneracy anywhere.
#
# ACTION REPEAT IS NOT OPTIONAL HERE. A held dither phase is a crash, so at 33 Hz the policy must
# emit commands it actually means -- the repeat is what kills the dither. Rate head and repeat are
# synergistic; shipping one without the other misses the point (hence act_repeat defaults to 3).


def step_ratezone(key, s: CState, a, course, obs_xy, obs_r, wall=INERT_WALL):
    """a[0] -> yaw rate DIRECTLY (a0 * yawrate_max); a[1] -> commanded speed.

    Contrast with step_steer/step_magforest, which take a 2-D heading VECTOR through atan2 and a
    saturating P-law. Here there is no angle, no branch cut, and no dependence between the two
    channels.

    YAW-RATE ACTUATOR LAG (2026-08-13): the commanded rate is followed through a first-order lag
    (tau = YAW_TAU), mirroring the real Crazyflie attitude loop. This is DYNAMICS REALISM, not
    reward shaping -- the distinction matters, because this project has twice found that penalties
    suppress capability. Rationale: the rate head plus act-repeat did NOT kill the dither, because
    at 33 Hz a held +-3.5 rad/s phase swings only ~6 deg, cheap enough for the stochastic actor to
    keep alternating. A physical actuator cannot follow that alternation, so dithering now simply
    fails to steer -- the physics punishes it instead of a reward term."""
    psi = _yaw_of(s.q)
    yaw_cmd = jnp.clip(a[0], -1.0, 1.0) * course['yawrate_max']       # DIRECT rate command
    # first-order lag: the achieved rate chases the command with time constant tau
    alpha = STEER_DT / jnp.maximum(course.get('yaw_tau', YAW_TAU), 1e-6)
    yaw_rate = s.w[2] + (yaw_cmd - s.w[2]) * jnp.clip(alpha, 0.0, 1.0)
    psi_n = psi + yaw_rate * STEER_DT
    fwd = jnp.array([jnp.cos(psi_n), jnp.sin(psi_n), 0.0])

    vmin, vmax, a_max = course['vmin'], course['vmax'], course['a_max']
    v_cmd = vmin + 0.5 * (jnp.clip(a[1], -1.0, 1.0) + 1.0) * (vmax - vmin)
    v_cur = jnp.linalg.norm(s.v[:2])
    dv = jnp.clip(v_cmd - v_cur, -a_max * STEER_DT, a_max * STEER_DT)
    v_spd = jnp.clip(v_cur + dv, vmin, vmax)

    p_n = (s.p + v_spd * fwd * STEER_DT).at[2].set(s.p[2])
    v_n = v_spd * fwd
    reward, dist, done, reached = _reward_steer(s, p_n, course, obs_xy, obs_r, wall, psi_n=psi_n)
    s_next = CState(p_n, v_n, _yaw_quat(psi_n), jnp.array([0.0, 0.0, yaw_rate]),
                    s.Om, dist, s.target)
    return s_next, reward, done, {"passed": reached, "all_done": reached, "target": s.target}


def step_speedforest(key, s: CState, a, course, obs_xy, obs_r, wall=INERT_WALL):
    """a[0:2] = egocentric heading (unchanged from step_steer); a[2] -> commanded speed.
    Speed INTEGRATES toward the command under an accel clamp (a real drone cannot step its speed);
    heading stays raw — deliberately NOT filtered, per the action-low-pass lesson."""
    psi = _yaw_of(s.q)
    rel = jnp.arctan2(a[1], a[0])
    yaw_rate = jnp.clip(course['kyaw'] * rel, -course['yawrate_max'], course['yawrate_max'])
    psi_n = psi + yaw_rate * STEER_DT
    fwd = jnp.array([jnp.cos(psi_n), jnp.sin(psi_n), 0.0])

    vmin, vmax, a_max = course['vmin'], course['vmax'], course['a_max']
    v_cmd = vmin + 0.5 * (jnp.clip(a[2], -1.0, 1.0) + 1.0) * (vmax - vmin)
    v_cur = jnp.linalg.norm(s.v[:2])
    dv = jnp.clip(v_cmd - v_cur, -a_max * STEER_DT, a_max * STEER_DT)
    v_spd = jnp.clip(v_cur + dv, vmin, vmax)

    p_n = (s.p + v_spd * fwd * STEER_DT).at[2].set(s.p[2])
    v_n = v_spd * fwd
    reward, dist, done, reached = _reward_steer(s, p_n, course, obs_xy, obs_r, wall)
    s_next = CState(p_n, v_n, _yaw_quat(psi_n), jnp.array([0.0, 0.0, yaw_rate]),
                    s.Om, dist, s.target)
    return s_next, reward, done, {"passed": reached, "all_done": reached, "target": s.target}
