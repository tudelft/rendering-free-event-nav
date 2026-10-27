"""Closed-loop eval of a proxy-trained policy on physically rendered (Isaac + EVIS) events.

Replaces the analytic event proxy the policy was trained on with the full
Isaac-rendered warp-accelerated EVIS event stream, in the SAME task geometry
(forest12: 15 m corridor, half-width 2 m, 12 varied trees with branches,
3 m/s constant speed, 100 Hz heading-steering kinematics).

Per control tick (10 ms):
  render 1 RTX keyframe (HDR + motion vectors + depth) at the drone pose
  -> warp K=4 -> 400 Hz fine frames -> EVIS event model -> per-tick ON/OFF
  counts -> rolling 30 ms window (3 ticks) -> 64x64x2 sum-binned -> uint8 obs
  -> TCP to the policy process -> 2-D action back -> kinematics.

The policy side is training/isaac_client_ppo.py (JAX CPU). Start THIS script
first (it binds the socket), then the client; training/configs/isaac_eval.sh
launches both.

Protocol (little-endian):
  server->client per tick: b'O' + flags u8 (1=is_first, 2=is_last, 4=is_terminal)
                           + reward f32 + vector V*f32 + image 8192*u8
  client->server per tick: A*f32 action  (not read after is_last obs)
  server->client at end:   b'Q'
  V/A: 3/2 for fixed and mag2d speed modes, 4/3 for cmd3 (the client derives
  both from the policy's own obs/act spaces; the server from --speed-mode).

Run:
  PYTHONPATH=$REPO/isaac_eval:$REPO/isaac_eval/evis \
    $ISAAC_SIM_PATH/python.sh isaac_eval/forest_policy_eval.py \
      --episodes 30 --out /tmp/forest_eval
"""
import argparse
import math
import os
import socket
import struct
import sys
import time

import numpy as np

# ---------------- task constants (mirror proxy/quad_course_env.py) ------------
STEER_DT = 0.01
KYAW = 2.5
COLLIDE_CLEAR = 0.10

TASKS = {
    # forest12: 15 m corridor @ 3 m/s, 12 varied trees + branches, 30 ms window
    "forest12": dict(V0=3.0, YAWRATE_MAX=3.5, X0=0.0, X_GOAL=15.0, HALF_W=2.0,
                     Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.03, MAX_STEPS=900,
                     N_OBS=12, OBS_R=(0.06, 0.30), BRANCHES=True,
                     BAND_JIT=0.35, X_PAD=(1.5, 1.0), Y_PAD=0.3),
    # steer4: indoor 6 m corridor @ 1.2 m/s, 4 plain poles, 45 ms window
    "steer4": dict(V0=1.2, YAWRATE_MAX=3.0, X0=0.0, X_GOAL=6.0, HALF_W=1.3,
                   Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.045, MAX_STEPS=700,
                   N_OBS=4, OBS_R=(0.128, 0.208), BRANCHES=False,
                   BAND_JIT=0.22, X_PAD=(1.2, 0.8), Y_PAD=0.35),
    # magforest (round 3, variant B): 22 m corridor, 3 density zones with
    # SHUFFLED factors (0.3/1.0/2.0), equal trees per zone, zone width ~
    # 1/factor — mirrors quad_course_env.sample_speedforest_obs. Use with
    # --speed-mode mag2d --vmin 1.5.
    "magforest": dict(V0=1.5, YAWRATE_MAX=3.5, X0=0.0, X_GOAL=22.0, HALF_W=2.0,
                      Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.03, MAX_STEPS=900,
                      N_OBS=12, OBS_R=(0.06, 0.30), BRANCHES=True,
                      BAND_JIT=0.3, X_PAD=(1.5, 1.0), Y_PAD=0.3,
                      ZONES=(0.3, 1.0, 2.0)),
    # magzone (round 4): EQUAL ~6.5 m zones, density by TREE COUNT (1/4/13 of
    # 18), shuffled per episode — mirrors quad_course_env.sample_zoneforest_obs.
    "magzone": dict(V0=1.5, YAWRATE_MAX=3.5, X0=0.0, X_GOAL=22.0, HALF_W=2.0,
                    Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.03, MAX_STEPS=900,
                    N_OBS=18, OBS_R=(0.06, 0.30), BRANCHES=True,
                    BAND_JIT=0.3, X_PAD=(1.5, 1.0), Y_PAD=0.3,
                    ZONE_COUNTS=(1, 4, 13), ZONE_FACTORS=(0.3, 1.0, 3.5)),
    # sqroom (2026-08-26): 22x22 square arena mirroring quad_course_env sample_squareroom_obs —
    # WHITE walls all 4 sides (plain style forced), POINT goal at (22,0) r=1.0, trees over the
    # whole floor with per-episode density keep U(0.15,1.0), full-room spawns with clearance.
    "sqroom": dict(V0=1.5, YAWRATE_MAX=3.5, X0=0.0, X_GOAL=22.0, HALF_W=11.0,
                   Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.02, MAX_STEPS=3600,
                   N_OBS=99, OBS_R=(0.06, 0.30), BRANCHES=False,
                   BAND_JIT=0.35, X_PAD=(1.5, 0.5), Y_PAD=1.0, SQROOM=True),
    # sqroom5 (2026-08-27): 5x5 m room (real flight-cage scale), point goal on the far side,
    # spawn on the near side — side-to-side crossing test.
    "sqroom5": dict(V0=1.5, YAWRATE_MAX=3.5, X0=0.0, X_GOAL=5.0, HALF_W=2.5,
                    Z_FLY=1.2, Z_TOP=3.0, DT_EVT=0.02, MAX_STEPS=900,
                    N_OBS=0, OBS_R=(0.06, 0.30), BRANCHES=False,
                    BAND_JIT=0.35, X_PAD=(1.0, 0.5), Y_PAD=0.6, SQROOM=True,
                    SPAWN_X=(0.4, 1.2)),
}

# set from --task in main(); module-level so the helpers read them
FOREST_V0 = 3.0
FOREST_YAWRATE_MAX = 3.5
X0, X_GOAL, HALF_W = 0.0, 15.0, 2.0
Z_FLY, Z_TOP = 1.2, 3.0
DT_EVT = 0.03
MAX_STEPS = 900
OOB_W = None
GOAL_R = 1.0
PROPS_ACTIVE = False                       # --oob-w: walls+oob wider than the tree band

# proxy uint8 image stats measured on the TRAINING env (calibration targets):
#   active_frac=0.0805 median_nz=51 mean_nz=77 p90=194 sat=5.5%
PROXY_MEDIAN_NZ = 51.0


def sample_forest_layout(rng, task):
    """Numpy port of sample_forest_obs / sample_steer_obs obstacle distributions.

    Same bands/ranges (not bit-identical draws — eval is over the distribution,
    not a fixed seed pairing). Returns dict with trunks + branches.
    """
    n_obs = task["N_OBS"]
    xlo, xhi = X0 + task["X_PAD"][0], X_GOAL - task["X_PAD"][1]
    zone_meta = None
    if n_obs == 0:                                   # empty room: walls only
        if task.get("ZONE_COUNTS") or task.get("ZONES"):
            Z = len(task.get("ZONE_COUNTS") or task["ZONES"])
            zone_meta = {"edges": xlo + np.arange(Z + 1) * (xhi - xlo) / Z,
                         "factors": np.ones(Z)}
        e = np.zeros(0)
        return {"xy": np.zeros((0, 2)), "r": e, "h": e, "branches": [],
                "zones": zone_meta}
    if task.get("ZONE_COUNTS"):
        # round-4 magzone: EQUAL-width zones, density via shuffled tree counts
        counts = np.array(task["ZONE_COUNTS"])
        if counts.sum() != n_obs:
            # --n-obs override: rescale the counts proportionally
            # (largest-remainder, min 1 per zone) so all arrays stay consistent
            w = counts / counts.sum()
            raw = w * n_obs
            counts = np.maximum(np.floor(raw).astype(int), 1)
            while counts.sum() < n_obs:
                counts[np.argmax(raw - counts)] += 1
            while counts.sum() > n_obs:
                j = int(np.argmax(counts))
                counts[j] = max(1, counts[j] - 1)
        Z = len(counts)
        W = (xhi - xlo) / Z
        perm = rng.permutation(Z)                 # which zone each count lands in
        edges = xlo + np.arange(Z + 1) * W
        xs_list, factors = [], np.zeros(Z)
        for g in range(Z):
            z = perm[g]
            band = W / counts[g]
            xs_list.append(edges[z] + band * (np.arange(counts[g]) + 0.5)
                           + band * rng.uniform(-0.3, 0.3, counts[g]))
            factors[z] = task["ZONE_FACTORS"][g]
        xs = np.concatenate(xs_list)
        zone_meta = {"edges": edges, "factors": factors}
    elif task.get("ZONES"):
        # zoned density (magforest): equal trees per zone, zone x-extent
        # scaled inversely to its (shuffled) density factor
        factors = np.array(task["ZONES"], float)
        rng.shuffle(factors)
        inv = 1.0 / factors
        widths = inv / inv.sum() * (xhi - xlo)
        edges = xlo + np.concatenate([[0.0], np.cumsum(widths)])
        n_per = max(1, n_obs // len(factors))
        n_obs = n_per * len(factors)          # keep every array the same length
        zi = np.repeat(np.arange(len(factors)), n_per)
        ki = np.tile(np.arange(n_per), len(factors))
        band = widths[zi] / n_per
        xs = edges[zi] + band * (ki + 0.5) + band * rng.uniform(-0.3, 0.3, n_obs)
        zone_meta = {"edges": edges, "factors": factors}
    else:
        band = (xhi - xlo) / n_obs
        centres = xlo + band * (np.arange(n_obs) + 0.5)
        xs = centres + band * rng.uniform(-task["BAND_JIT"], task["BAND_JIT"],
                                          n_obs)
    ymax = HALF_W - task["Y_PAD"]
    ys = rng.uniform(-ymax, ymax, n_obs)
    r = rng.uniform(task["OBS_R"][0], task["OBS_R"][1], n_obs)
    if task.get("SQROOM"):
        keep = rng.uniform(0.15, 1.0)
        m = (rng.uniform(0, 1, n_obs) < keep)
        m &= np.hypot(xs - X_GOAL, ys) > 2.2          # keep the goal point clear
        xs, ys, r = xs[m], ys[m], r[m]
        n_obs = int(m.sum())
        props = []
        if globals().get('PROPS_ACTIVE', False):
            for kind, n_slot, p_on, prad in (("screen", 2, 0.6, 0.9),
                                             ("truss", 2, 0.5, 0.35),
                                             ("plant", 4, 0.5, 0.45)):
                for _ in range(n_slot):
                    if rng.uniform() < p_on:
                        px = rng.uniform(X0 + 2.0, X_GOAL - 2.0)
                        py = rng.uniform(-(HALF_W - 1.5), HALF_W - 1.5)
                        if math.hypot(px - X_GOAL, py) > 2.2:
                            props.append((kind, px, py, prad))
        zone_meta = zone_meta if 'zone_meta' in dir() else None
    if task["BRANCHES"]:
        h = rng.uniform(1.2, Z_TOP, n_obs)
    elif task.get("SQROOM"):
        h = rng.uniform(1.2, Z_TOP, n_obs)
    else:
        h = np.full(n_obs, Z_TOP)                    # steer: full-height poles
    branches = []
    _props = props if (task.get("SQROOM") and globals().get('PROPS_ACTIVE', False)) else []
    if task["BRANCHES"]:
        for i in range(n_obs):
            for _ in range(2):
                ang = rng.uniform(0, 2 * np.pi)
                blen = rng.uniform(0.0, 0.6)
                bz = rng.uniform(0.45, 0.95) * h[i]
                if blen < 0.05:
                    continue
                branches.append((xs[i], ys[i], bz, ang, blen))
    return {"xy": np.stack([xs, ys], -1), "r": r, "h": h, "branches": branches,
            "zones": zone_meta, "props": _props}


# ---------------- pure kinematics / observation logic (unit-tested) ----------
# Module-level and parameterized by an explicit cfg dict so tests can exercise
# them without Isaac; main() binds them to the CLI args via thin wrappers.
MAG_DEADZONE, MAG_SPAN = 0.3, 0.7      # match quad_course_env.step_magforest


def speed_cmd(action, mode, vmin, vmax):
    """Target speed for the given action under a speed mode; None = fixed."""
    if mode == "cmd3" and len(action) >= 3:
        a2 = max(-1.0, min(1.0, float(action[2])))
        return vmin + (a2 + 1.0) / 2.0 * (vmax - vmin)
    if mode == "cmd2rate" and len(action) >= 2:
        # ratezone (round 5-revised): a[1] is the speed channel
        a1 = max(-1.0, min(1.0, float(action[1])))
        return vmin + (a1 + 1.0) / 2.0 * (vmax - vmin)
    if mode in ("mag2d", "mag2d-inv"):
        mag = min(1.0, math.hypot(float(action[0]), float(action[1])))
        frac = max(0.0, min(1.0, (mag - MAG_DEADZONE) / MAG_SPAN))
        if mode == "mag2d-inv":
            frac = 1.0 - frac
        return vmin + frac * (vmax - vmin)
    return None


def kin_step(state, action, cfg):
    """One 100 Hz steering tick. cfg keys: v_const, yawrate_max, z_fly, mode,
    vmin, vmax, accel_max, steer ('heading'|'rate').
    Mirrors quad_course_env.step_steer/step_magforest (and step_ratezone)."""
    psi = state["psi"]
    if cfg.get("steer", "heading") == "rate":
        # direct rate head: a[0] IS the normalized yaw rate (no atan2, no P-law)
        a0 = max(-1.0, min(1.0, float(action[0])))
        yaw_cmd = a0 * cfg["yawrate_max"]
        tau = cfg.get("yaw_tau", 0.0)
        if tau > 0.0:
            # first-order actuator lag, EXACTLY as quad_course_env.step_ratezones
            # (2026-08-13): the achieved rate chases the command with time
            # constant tau; the LAGGED rate is what the state carries (and what
            # proprio reports), mirroring the real CF attitude loop.
            alpha = min(1.0, STEER_DT / max(tau, 1e-6))
            yaw_rate = state.get("yaw_rate", 0.0) \
                + (yaw_cmd - state.get("yaw_rate", 0.0)) * alpha
        else:
            yaw_rate = yaw_cmd
    else:
        rel = math.atan2(float(action[1]), float(action[0]))
        yaw_rate = max(-cfg["yawrate_max"],
                       min(cfg["yawrate_max"], KYAW * rel))
    psi_n = psi + yaw_rate * STEER_DT
    v = state.get("v", cfg["v_const"])
    v_cmd = speed_cmd(action, cfg["mode"], cfg["vmin"], cfg["vmax"])
    if v_cmd is not None:
        dv = max(-cfg["accel_max"] * STEER_DT,
                 min(cfg["accel_max"] * STEER_DT, v_cmd - v))
        v = v + dv                       # integrator + accel clamp, no filter
    p = state["p"]
    p_n = p + v * STEER_DT * np.array(
        [math.cos(psi_n), math.sin(psi_n), 0.0])
    p_n[2] = cfg["z_fly"]
    return {"p": p_n, "psi": psi_n, "yaw_rate": yaw_rate, "v": v}


def frozen_or_proprio(state, cfg):
    if cfg.get("freeze_vector") is not None:
        return np.array(cfg["freeze_vector"], np.float32)
    return proprio_vec(state, cfg)


def proprio_vec(state, cfg):
    """[3] (or [4] for cmd3): egocentric goal bearing, yaw rate norm, v/vmax.
    freeze_bearing emulates the N6 template bug: bearing pinned to (1, 0)."""
    if cfg.get("freeze_bearing"):
        ang = 0.0
    else:
        gv = np.array([cfg["x_goal"], 0.0]) - state["p"][:2]
        ang = math.atan2(gv[1], gv[0]) - state["psi"]
    base = [math.cos(ang), math.sin(ang),
            state["yaw_rate"] / cfg["yawrate_max"]]
    if cfg["mode"] in ("cmd3", "cmd2rate"):
        base.append(state.get("v", cfg["v_const"]) / cfg["vmax"])
    return np.array(base, np.float32)


def straight_action(steer_mode, act_floats):
    """The scripted 'fly straight' action (pre-roll / calibration): heading
    mode encodes straight as the +x unit vector; rate mode as zero yaw rate."""
    if steer_mode == "rate":
        return (0.0, 0.0, 0.0)[:act_floats]
    return (1.0, 0.0, 0.0)[:act_floats]


def done_flags(p, layout, half_w, x_goal, clear=COLLIDE_CLEAR, point_goal=False, x0=0.0):
    """(collided, oob, reached) for position p against trunk circles."""
    if len(layout["xy"]) == 0:                       # empty room: walls only
        collided = False
    else:
        d = np.linalg.norm(p[None, :2] - layout["xy"], axis=-1) - layout["r"]
        collided = bool(d.min() < clear)
    oob = bool(abs(p[1]) > half_w)
    for (_k, px, py, prad) in layout.get("props", []):
        if math.hypot(p[0] - px, p[1] - py) < prad + clear:
            collided = True
    if point_goal:
        reached = bool(math.hypot(p[0] - x_goal, p[1]) < GOAL_R)
        oob = oob or p[0] > x_goal + 1.0 or p[0] < x0 - 1.0
    else:
        reached = bool(p[0] > x_goal)
    return collided, oob, reached


def genx_contrast_for_p90(nets, target, h_out):
    """Binary-search the native-pixel contrast so that p90 of |px - 128| over
    ACTIVE output pixels (continuous block mean, pre-rounding) hits `target`.
    This is the training agent's gx calibration: the rec1.bin recording gives
    p90 of active |px-128| = 12 (GENX_1CH_PROXY_PLAN.md section 2). Pegging at
    the native stage is inside the search, so the fit is exact, not linear."""
    lo, hi = 1e-3, 400.0
    for _ in range(48):
        c = (lo * hi) ** 0.5
        devs = []
        for n in nets:
            H = n.shape[0]
            b = H // h_out
            px = np.clip(n.astype(np.float64) * c + 128.0, 0.0, 255.0)
            bm = px.reshape(h_out, b, h_out, b).mean((1, 3))
            d = np.abs(bm - 128.0)
            devs.append(d[d > 0.5])                    # would round to >= 1
        alld = (np.concatenate(devs) if any(len(d) for d in devs)
                else np.zeros(0))
        if alld.size == 0 or np.percentile(alld, 90) < target:
            lo = c
        else:
            hi = c
    return float((lo * hi) ** 0.5)


def genx_obs(net, contrast, h_out):
    """GenX320 DIFF3D observation (GENX_1CH_PROXY_PLAN.md): per NATIVE pixel
    ``px = USAT(net*contrast + 128)`` (pegs at |net|~8 for stock contrast 16),
    then the firmware's AREA downsample = integer block MEAN of those uint8s.
    net: [H, W] signed event counts (ON - OFF accumulated over the window).
    Returns [h_out, h_out, 1] uint8, 128-centred."""
    net = np.asarray(net, np.float64)
    H = net.shape[0]
    assert H % h_out == 0, (H, h_out)
    b = H // h_out
    px = np.clip(net * contrast + 128.0, 0.0, 255.0).astype(np.uint8)
    blocks = px.reshape(h_out, b, h_out, b).astype(np.uint64).sum((1, 3))
    out = (blocks // (b * b)).astype(np.uint8)          # integer mean, as draw_image AREA
    return out[..., None]


def pack_obs(flags, reward, vec, img):
    """Server->client observation frame (see module docstring protocol)."""
    return (b"O" + struct.pack("<Bf", flags, float(reward))
            + vec.astype("<f4").tobytes() + img.tobytes())


def unpack_obs(buf, vec_dim, img_shape=(64, 64, 2)):
    """Reference parser for pack_obs (mirrors isaac_bridge_client's reads)."""
    assert buf[:1] == b"O"
    flags, reward = struct.unpack("<Bf", buf[1:6])
    n = 4 * vec_dim
    vec = np.frombuffer(buf[6:6 + n], "<f4").copy()
    img = np.frombuffer(buf[6 + n:], np.uint8).reshape(img_shape).copy()
    return flags, reward, vec, img


class ForestStage:
    """Owns the persistent scene + per-episode obstacle prims.

    ``wall_style``: "checker" = alternating light/dark side-wall panels (strong
    hallway-flow events, MORE wall texture than the training proxy's sparse
    point grid); "plain" = uniform panels (silhouette-only wall events, closer
    to the proxy's sparseness).
    """

    def __init__(self, stage, realism="matched", wall_style="checker",
                 lighting="sun", scene_style="default"):
        self.wall_style = wall_style
        self.scene_style = scene_style
        self.sqroom = globals().get('SQROOM_ACTIVE', False)
        self.lighting = lighting
        from pxr import Gf, UsdGeom, UsdLux
        self.stage = stage
        self.Gf, self.UsdGeom = Gf, UsdGeom
        self._ep_prims = []
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        # ground: flat matte, LIGHT (bright floor = dark trunks contrast, and a
        # featureless surface makes near-zero events, matching the proxy which
        # renders NO ground points).
        self._cube("/World/ground", (X_GOAL / 2, 0, -0.05), (60, 40, 0.1),
                   (0.62, 0.60, 0.55))
        # corridor side walls: visible (training renders wall edge points), with
        # vertical shading edges every 1.5 m = sparse hallway-flow features.
        if self.wall_style != "none":
            for s, y in ((0, -(globals().get('OOB_W') or HALF_W) - 0.15),
                         (1, (globals().get('OOB_W') or HALF_W) + 0.15)):
                for i in range(int((X_GOAL + 4.0) / 1.5) + 2):
                    if self.wall_style == "checker":
                        c = 0.72 if (i + s) % 2 == 0 else 0.55
                    else:
                        c = 0.66
                    self._cube(f"/World/sidewall_{s}_{i}",
                               (-2.0 + 0.75 + i * 1.5, y, Z_TOP / 2),
                               (1.5, 0.12, Z_TOP), (c, c * 0.97, c * 0.9))
        # bright end wall far behind the goal (uniform -> few events)
        if self.sqroom:
            # square room: CLOSE end walls (plain white) at both x bounds
            hw_room = (globals().get('OOB_W') or HALF_W)
            self._cube("/World/endwall_far", (X_GOAL + 1.15, 0, Z_TOP / 2),
                       (0.2, 2 * hw_room + 0.5, Z_TOP), (0.7, 0.69, 0.66))
            self._cube("/World/endwall_near", (X0 - 1.15, 0, Z_TOP / 2),
                       (0.2, 2 * hw_room + 0.5, Z_TOP), (0.7, 0.69, 0.66))
        else:
            self._cube("/World/endwall", (X_GOAL + 6.0, 0, 2.0), (0.2, 40, 4.0),
                       (0.75, 0.73, 0.68))
        if self.scene_style == "lab":
            # FLIGHT-LAB DRESSING (2026-08-26): visual clutter of a real drone lab.
            import numpy as _np
            _rng = _np.random.RandomState(7)
            hw_lab = (globals().get('OOB_W') or HALF_W)
            span = X_GOAL + 4.0
            # 1. dark ceiling slab + rows of bright light tubes along the corridor
            self._cube("/World/lab_ceiling", (X_GOAL / 2, 0, Z_TOP + 0.25),
                       (span + 4, 2 * hw_lab + 2, 0.15), (0.22, 0.22, 0.24))
            n_rows = max(2, int(hw_lab))
            ys = _np.linspace(-hw_lab + 1.0, hw_lab - 1.0, n_rows)
            for ri, y in enumerate(ys):
                x = -1.0
                ti = 0
                while x < span - 1.0:
                    self._cube(f"/World/lab_tube_{ri}_{ti}",
                               (x + 2.0, y, Z_TOP + 0.12), (4.0, 0.15, 0.08),
                               (0.98, 0.98, 0.92))
                    x += 5.0; ti += 1
            # 2. geometric wallpaper: contrasting rectangles proud of the walls
            for side, wy in ((0, -hw_lab - 0.05), (1, hw_lab + 0.05)):
                for k in range(18):
                    wx = _rng.uniform(-1.0, span - 2.0)
                    wz = _rng.uniform(0.3, Z_TOP - 0.4)
                    w_ = _rng.uniform(0.4, 2.2); h_ = _rng.uniform(0.3, 1.4)
                    shade = _rng.choice([0.25, 0.45, 0.9])
                    self._cube(f"/World/lab_paper_{side}_{k}",
                               (wx, wy, wz), (w_, 0.06, h_),
                               (shade, shade * 0.95, shade * 0.9))
            # 3. curtain wall: one long section of alternating vertical slats with folds
            cx0 = _rng.uniform(X0 + 0.5, max(X0 + 1.0, X_GOAL - 8.0))
            n_slat = min(24, max(6, int((X_GOAL + 1.0 - cx0) / 0.25)))
            for k in range(n_slat):
                fold = 0.10 * (1 if k % 2 == 0 else -1)
                shade = 0.55 if k % 2 == 0 else 0.38
                self._cube(f"/World/lab_curtain_{k}",
                           (cx0 + k * 0.25, -hw_lab - 0.02 + fold, Z_TOP / 2 - 0.1),
                           (0.25, 0.10, Z_TOP - 0.2), (shade, shade * 0.9, shade * 1.05))
        if self.scene_style == "hangar":
            # HANGAR / GYM HALL (2026-09-11): fully enclosed realistic space.
            # Visual-only: collision contract (walls at +-HALF_W, Z_TOP done-check)
            # is untouched; everything here is dressing outside/above the arena.
            import numpy as _np
            _rng = _np.random.RandomState(11)
            hw_h = (globals().get('OOB_W') or HALF_W)
            span = X_GOAL - X0
            cxm = (X0 + X_GOAL) / 2.0
            CEIL = 5.2
            # -- gym floor: wood tone + painted court lines
            _fl = self._cube("/World/hg_floor", (cxm, 0, -0.02), (span + 6, 2*hw_h + 6, 0.05),
                       (0.55, 0.42, 0.28))
            self._matte(_fl, (0.55, 0.42, 0.28), roughness=0.9)
            for i, y in enumerate((-hw_h + 0.6, 0.0, hw_h - 0.6)):
                self._cube(f"/World/hg_line_x{i}", (cxm, y, 0.012), (span - 1.0, 0.09, 0.012),
                           (0.95, 0.95, 0.92))
            for i, x in enumerate(_np.linspace(X0 + 2, X_GOAL - 2, 5)):
                self._cube(f"/World/hg_line_y{i}", (x, 0, 0.012), (0.09, 2*hw_h - 1.5, 0.012),
                           (0.9, 0.35, 0.2) if i == 2 else (0.95, 0.95, 0.92))
            # -- full-height walls with panel seams + high window band
            wall_col = (0.62, 0.64, 0.66)
            for nm, cx, cy, sx, sy in (
                    ("n", cxm, -hw_h - 0.35, span + 6, 0.3),
                    ("s", cxm,  hw_h + 0.35, span + 6, 0.3),
                    ("w", X0 - 1.35, 0, 0.3, 2*hw_h + 6),
                    ("e", X_GOAL + 1.35, 0, 0.3, 2*hw_h + 6)):
                self._cube(f"/World/hg_wall_{nm}", (cx, cy, CEIL/2), (sx, sy, CEIL), wall_col)
            # panel seams (dark vertical strips) + window band on side walls
            for side, wy in (("n", -hw_h - 0.18), ("s", hw_h + 0.18)):
                for k, x in enumerate(_np.arange(X0 - 1.0, X_GOAL + 1.0, 2.4)):
                    self._cube(f"/World/hg_seam_{side}_{k}", (x, wy, CEIL/2),
                               (0.08, 0.08, CEIL - 0.4), (0.35, 0.36, 0.38))
                for k, x in enumerate(_np.arange(X0, X_GOAL - 0.5, 3.0)):
                    self._cube(f"/World/hg_win_{side}_{k}", (x + 1.2, wy, CEIL - 0.9),
                               (2.0, 0.06, 0.9), (0.93, 0.96, 1.0))
            # -- big hangar door on the goal wall (frame + horizontal ribs)
            self._cube("/World/hg_door", (X_GOAL + 1.15, 0, 1.9), (0.15, 6.0, 3.8), (0.5, 0.52, 0.55))
            for k in range(5):
                self._cube(f"/World/hg_door_rib_{k}", (X_GOAL + 1.05, 0, 0.55 + k*0.75),
                           (0.06, 5.8, 0.1), (0.32, 0.34, 0.36))
            # -- ceiling slab + corrugation ribs + roof trusses + hanging lights
            self._cube("/World/hg_ceil", (cxm, 0, CEIL + 0.15), (span + 6, 2*hw_h + 6, 0.2),
                       (0.42, 0.43, 0.45))
            for k, x in enumerate(_np.arange(X0 - 1.0, X_GOAL + 1.5, 2.0)):
                self._cube(f"/World/hg_rib_{k}", (x, 0, CEIL + 0.02), (0.12, 2*hw_h + 5, 0.1),
                           (0.30, 0.31, 0.33))
            for k, y in enumerate(_np.linspace(-hw_h + 1.5, hw_h - 1.5, 3)):
                # truss: bottom chord + verticals
                self._cube(f"/World/hg_truss_b_{k}", (cxm, y, CEIL - 0.55),
                           (span + 2, 0.12, 0.12), (0.55, 0.35, 0.2))
                for j, x in enumerate(_np.arange(X0, X_GOAL, 2.2)):
                    self._cube(f"/World/hg_truss_v_{k}_{j}", (x, y, CEIL - 0.28),
                               (0.09, 0.09, 0.45), (0.55, 0.35, 0.2))
                for j, x in enumerate(_np.arange(X0 + 1.0, X_GOAL, 4.4)):
                    self._cube(f"/World/hg_lamp_{k}_{j}", (x, y, CEIL - 0.85),
                               (0.9, 0.5, 0.12), (1.0, 0.98, 0.9))
                    from pxr import UsdLux as _UL
                    rl = _UL.RectLight.Define(self.stage, f"/World/hg_light_{k}_{j}")
                    rl.GetIntensityAttr().Set(9000.0)
                    rl.GetWidthAttr().Set(0.9); rl.GetHeightAttr().Set(0.5)
                    _xf = self.UsdGeom.XformCommonAPI(rl.GetPrim())
                    _xf.SetTranslate(self.Gf.Vec3d(float(x), float(y), CEIL - 0.95))
                    _xf.SetRotate(self.Gf.Vec3f(180.0, 0.0, 0.0))
        if self.lighting == "sun":
            sun = UsdLux.DistantLight.Define(stage, "/World/sun")
            sun.GetIntensityAttr().Set(2500.0)
            UsdGeom.XformCommonAPI(sun.GetPrim()).SetRotate(Gf.Vec3f(0, -55, 15))
            dome_i = 700.0
        else:               # "dome": shadowless ambient — the proxy renders no
            dome_i = 2200.0  # shadows, so this removes that whole event source
        dome = UsdLux.DomeLight.Define(stage, "/World/dome")
        dome.GetIntensityAttr().Set(dome_i)

    def _matte(self, prim, color, roughness=0.85):
        from pxr import UsdShade, Sdf
        mpath = str(prim.GetPrim().GetPath()) + "_mat"
        mat = UsdShade.Material.Define(self.stage, mpath)
        sh = UsdShade.Shader.Define(self.stage, mpath + "/s")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(self.Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        sh.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.04)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(mat)

    def _cube(self, path, pos, size, color):
        p = self.UsdGeom.Cube.Define(self.stage, path)
        p.GetSizeAttr().Set(1.0)
        xf = self.UsdGeom.XformCommonAPI(p.GetPrim())
        xf.SetTranslate(self.Gf.Vec3d(*[float(v) for v in pos]))
        xf.SetScale(self.Gf.Vec3f(*[float(v) for v in size]))
        p.GetDisplayColorAttr().Set([self.Gf.Vec3f(*color)])
        return p

    def _cyl(self, path, pos, r, h, color, yaw_deg=None):
        p = self.UsdGeom.Cylinder.Define(self.stage, path)
        p.GetRadiusAttr().Set(float(r))
        p.GetHeightAttr().Set(float(h))
        p.GetAxisAttr().Set("Z" if yaw_deg is None else "X")
        xf = self.UsdGeom.XformCommonAPI(p.GetPrim())
        xf.SetTranslate(self.Gf.Vec3d(*[float(v) for v in pos]))
        if yaw_deg is not None:
            xf.SetRotate(self.Gf.Vec3f(0, 0, float(yaw_deg)),
                         self.UsdGeom.XformCommonAPI.RotationOrderXYZ)
        p.GetDisplayColorAttr().Set([self.Gf.Vec3f(*color)])
        return p

    def build_episode(self, layout, rng):
        """(Re)create the per-episode tree prims; DARK trunks on the bright scene."""
        self.clear_episode()
        for i, ((x, y), r, h) in enumerate(zip(layout["xy"], layout["r"],
                                               layout["h"])):
            shade = rng.uniform(0.10, 0.22)
            path = f"/World/ep/tree{i}"
            self._cyl(path, (x, y, h / 2), r, h, (shade, shade * 0.9, shade * 0.7))
            self._ep_prims.append(path)
        for k, (kind, px, py, prad) in enumerate(layout.get("props", [])):
            base = f"/World/ep/prop{k}"
            if kind == "screen":
                self._cyl(base + "_pole", (px, py, 0.3), 0.03, 0.6, (0.3, 0.3, 0.32))
                self._cube(base + "_panel", (px, py, 1.2), (1.8, 0.05, 1.2),
                           (0.92, 0.92, 0.9))
                self._ep_prims += [base + "_pole", base + "_panel"]
            elif kind == "truss":
                for si, dx in enumerate((-0.3, 0.3)):
                    self._cube(f"{base}_rail{si}", (px + dx, py, 1.3), (0.06, 0.06, 2.6),
                               (0.55, 0.55, 0.58))
                    self._ep_prims.append(f"{base}_rail{si}")
                for zi, z in enumerate((0.4, 0.95, 1.5, 2.05, 2.6)):
                    self._cube(f"{base}_rung{zi}", (px, py, z), (0.66, 0.05, 0.05),
                               (0.55, 0.55, 0.58))
                    self._ep_prims.append(f"{base}_rung{zi}")
            else:  # plant
                self._cyl(base + "_stem", (px, py, 0.45), 0.03, 0.9, (0.25, 0.2, 0.12))
                self._ep_prims.append(base + "_stem")
                prng = np.random.RandomState(k * 77 + 5)
                for li in range(7):
                    off = prng.normal(0, 0.22, 2); oz = 1.05 + prng.normal(0, 0.28)
                    sz = prng.uniform(0.12, 0.28)
                    self._cube(f"{base}_leaf{li}", (px + off[0], py + off[1], oz),
                               (sz, sz, sz * 0.6), (0.13, 0.35, 0.14))
                    self._ep_prims.append(f"{base}_leaf{li}")
        for j, (bx, by, bz, ang, blen) in enumerate(layout["branches"]):
            path = f"/World/ep/br{j}"
            cx = bx + math.cos(ang) * blen / 2
            cy = by + math.sin(ang) * blen / 2
            self._cyl(path, (cx, cy, bz), 0.03, blen, (0.15, 0.12, 0.09),
                      yaw_deg=math.degrees(ang))
            self._ep_prims.append(path)

    def clear_episode(self):
        for p in self._ep_prims:
            self.stage.RemovePrim(p)
        self._ep_prims = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="forest12")
    ap.add_argument("--out", default="/tmp/forest_eval")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--stills", type=int, default=0, help="render N isometric scene stills to --out and exit")
    ap.add_argument("--room", type=float, default=0.0, help="square-room size override: sets X_GOAL=room, HALF_W=room/2 (density scaling)")
    ap.add_argument("--depth-obs", action="store_true", help="ground-truth depth image obs (baseline arm): distance_to_image_plane 320->obs_res pixel-center subsample, proxy inverse-depth mapping")
    ap.add_argument("--warp", type=int, default=4, help="fine frames per 10 ms tick")
    ap.add_argument("--event-c", type=float, default=0.15)
    ap.add_argument("--noise", choices=["none", "realistic"], default="none")
    ap.add_argument("--port", type=int, default=5560)
    ap.add_argument("--gain", type=float, default=0.0,
                    help="counts->proxy-units gain; 0 = auto-calibrate on a "
                         "scripted pre-flight so median nonzero uint8 matches "
                         "the proxy's (51)")
    ap.add_argument("--polarity", choices=["off_on", "on_off"], default="off_on",
                    help="map (image ch0, ch1) = (OFF, ON) counts [default: the "
                         "proxy's leading/trailing streak semantics for dark "
                         "obstacles on a bright background] or swapped")
    ap.add_argument("--save-frames", type=int, default=3,
                    help="save per-tick debug npz for the first N episodes")
    ap.add_argument("--steer-mode", choices=["heading", "rate"],
                    default="heading",
                    help="heading: a[0:2] unit-vector -> atan2 -> P-law (all "
                         "policies through magzone1); rate: a[0] IS the "
                         "normalized yaw rate directly (ratezone, round 5-rev)")
    ap.add_argument("--genx", action="store_true",
                    help="single-channel GenX320 obs: signed net (ON-OFF) per "
                         "native pixel -> USAT(net*contrast+128) -> AREA mean "
                         "to --obs-res, [H,H,1] uint8 128-centred. Pair with "
                         "--dt-evt 0.02 (the sensor integrates ~20 ms).")
    ap.add_argument("--genx-contrast", type=float, default=16.0,
                    help="stock firmware CONTRAST_DEFAULT. Auto-calibration "
                         "divides it by the measured median |net| per active "
                         "native pixel (real sensor: ~1 count per swept pixel); "
                         "pass --gain > 0 to use that as the contrast directly.")
    ap.add_argument("--genx-p90-target", type=float, default=12.0,
                    help="calibrate contrast so p90 of active |px-128| on the "
                         "output frames hits this (rec1.bin: 12). 0 = use the "
                         "median-count convention (contrast 16 / median |net|).")
    ap.add_argument("--y0", type=float, default=None,
                    help="fixed initial lateral offset (m) — lateral-recovery probe")
    ap.add_argument("--half-w", type=float, default=0.0,
                    help="override corridor half-width (wide-room tests)")
    ap.add_argument("--yawrate-max", type=float, default=0.0,
                    help="override yaw-rate cap in rad/s (low-yaw arms; also "
                         "normalizes the obs yawrate slot)")
    ap.add_argument("--oob-w", type=float, default=0.0,
                    help="widen ONLY the walls+out-of-bounds boundary to this "
                         "half-width; tree sampling keeps the task band "
                         "(recovery-space tests, 2026-08-25)")
    ap.add_argument("--freeze-vector", type=str, default="",
                    help="freeze the ENTIRE proprio vector to these comma-"
                         "separated values (e.g. '1,0,0,0.5' = dead-uplink "
                         "N6 defaults). Overrides --freeze-bearing.")
    ap.add_argument("--freeze-bearing", action="store_true",
                    help="pin the proprio goal bearing to (1,0) — emulates the "
                         "N6 template dummy-goal bug for sim reproduction")
    ap.add_argument("--no-done", action="store_true",
                    help="ignore collision/oob/reach termination; run to MAX_STEPS "
                         "(events still logged). For trajectory characterization.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="override the task's MAX_STEPS (100 Hz kinematic ticks)")
    ap.add_argument("--genx-diag", action="store_true",
                    help="log intra-window cancellation ratio statistics")
    ap.add_argument("--genx-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="net polarity sign (ablation hook)")
    ap.add_argument("--dt-evt", type=float, default=0.0,
                    help="override the task's event integration window (s). "
                         "The 10 Hz profile (yawzoneh32r10) uses 0.10 with "
                         "--calib-target 32; default 0 = task value.")
    ap.add_argument("--psi0-deg", type=float, default=None,
                    help="FORCE the initial heading (deg vs corridor axis, "
                         "+ = left) instead of the task's +-8.6 deg sample. "
                         "Large-yaw recovery probe: sim episodes never exceed "
                         "~+-30 deg yaw (measured 2026-08-17), real avoidance "
                         "exceeds 90 -- this tests the unvisited envelope.")
    ap.add_argument("--yaw-tau", type=float, default=0.0,
                    help="first-order yaw-rate actuator lag time constant (s), "
                         "rate steer mode only. ratezone3+ ('ratezones' task) "
                         "trains with 0.08 — MUST match or the dither-damping "
                         "the lag provides is absent at eval. 0 = instant "
                         "(pre-lag policies).")
    ap.add_argument("--speed-mode",
                    choices=["fixed", "mag2d", "mag2d-inv", "cmd3", "cmd2rate"],
                    default="fixed",
                    help="fixed: constant task speed, 2-float actions. "
                         "mag2d: 2-float actions, |a| (dead-zone 0.3, clip 1) "
                         "-> speed command; heading from the direction as "
                         "always; obs vector stays [3] (seedable policies). "
                         "cmd3: 3-float actions, a[2] -> speed command; obs "
                         "vector gains v/vmax as a 4th float.")
    ap.add_argument("--vmin", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=6.0)
    ap.add_argument("--accel-max", type=float, default=3.0,
                    help="speed integrator accel clamp (m/s^2)")
    ap.add_argument("--n-obs", type=int, default=-1,
                    help="override the task's obstacle count (0 = task default; "
                         "e.g. 24 = double-density forest12 stress test)")
    ap.add_argument("--obs-res", type=int, default=64,
                    help="policy image resolution (event counts are sum-binned "
                         "320 -> this; 32 for the N6 deploy-optimized nets; "
                         "must divide --res)")
    ap.add_argument("--act-hold-alt", type=str, default="",
                    help="alternating hold ticks, e.g. 7,8 -> exact 13.33 Hz decisions")
    ap.add_argument("--act-hold", type=int, default=1,
                    help="query the policy every Nth control tick and HOLD the "
                         "action in between (simulates the N6's low policy "
                         "rate: 3 ~ 33 Hz, 10 = 10 Hz at 100 Hz control). "
                         "Rewards are summed over the held span.")
    ap.add_argument("--flip-img", action="store_true",
                    help="DIAGNOSTIC: horizontally mirror the policy image. If "
                         "a policy that fails with a systematic one-sided veer "
                         "RECOVERS with this flag, its training proxy has a "
                         "mirrored horizontal axis (cf. the 2026-08-04 bug)")
    ap.add_argument("--min-count", type=int, default=0,
                    help="zero event bins below this raw count before the gain "
                         "(kills the single-event RTX-noise floor that inflates "
                         "the active fraction at coarse --obs-res; try 2 at 32)")
    ap.add_argument("--calib-target", type=float, default=PROXY_MEDIAN_NZ,
                    help="gain-calibration target: the TRAINING proxy's median "
                         "nonzero uint8 (51 for the 64x64 proxies; re-measure "
                         "and pass for other --obs-res)")
    ap.add_argument("--lighting", choices=["sun", "dome"], default="sun",
                    help="sun = directional light with sharp shadows (extra "
                         "event source the proxy never had); dome = shadowless")
    ap.add_argument("--third-person", action="store_true",
                    help="record a chase-cam RGB view (256px, every 2nd decision) "
                         "into each episode npz as 'tp'")
    ap.add_argument("--props", action="store_true",
                    help="sqroom: lab props lottery (screens, trusses, plants) "
                         "mirroring the proxy's course['props']")
    ap.add_argument("--goal-r", type=float, default=1.0,
                    help="point-goal capture radius (match training goal_r)")
    ap.add_argument("--scene-style", choices=["default", "lab", "hangar"], default="default",
                    help="lab: flight-lab dressing — geometric wallpaper, curtain "
                         "wall, dark ceiling with rows of bright light tubes")
    ap.add_argument("--wall-style", choices=["checker", "plain", "none"],
                    default="checker",
                    help="side walls: checker = strong hallway-flow events; "
                         "plain = silhouette-only (closer to the forest "
                         "proxy's sparse wall points); none = no wall prims "
                         "at all (STEER4-faithful: make_steer renders no wall "
                         "event points, the oob boundary is invisible)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # bind the task constants (module globals read by the helpers)
    global FOREST_V0, FOREST_YAWRATE_MAX, X0, X_GOAL, HALF_W
    global Z_FLY, Z_TOP, DT_EVT, MAX_STEPS
    T = TASKS[args.task]
    if args.n_obs >= 0:                    # 0 = explicit empty room (walls only)
        T = dict(T, N_OBS=args.n_obs)
    FOREST_V0, FOREST_YAWRATE_MAX = T["V0"], T["YAWRATE_MAX"]
    if args.yawrate_max > 0:
        FOREST_YAWRATE_MAX = args.yawrate_max
    global SQROOM_ACTIVE
    SQROOM_ACTIVE = T.get('SQROOM', False)
    global GOAL_R, PROPS_ACTIVE
    GOAL_R = args.goal_r
    PROPS_ACTIVE = args.props
    X0, X_GOAL, HALF_W = T["X0"], T["X_GOAL"], T["HALF_W"]
    if getattr(args, 'room', 0) and args.room > 0:      # square-room size override
        X_GOAL, HALF_W = args.room, args.room / 2.0
    if args.half_w > 0:
        HALF_W = args.half_w
    global OOB_W
    OOB_W = args.oob_w if args.oob_w > 0 else None
    Z_FLY, Z_TOP = T["Z_FLY"], T["Z_TOP"]
    DT_EVT, MAX_STEPS = T["DT_EVT"], T["MAX_STEPS"]
    if args.dt_evt > 0:
        DT_EVT = args.dt_evt
    if args.max_steps > 0:
        MAX_STEPS = args.max_steps
    elif args.vmax < 2.9:
        # task budgets assume ~3 m/s; slower caps made the goal unreachable
        # inside the old budget (caught 2026-08-24). Scale like skycourse.
        MAX_STEPS = int(MAX_STEPS * 3.0 / max(args.vmax, 0.2))
        print('[task] vmax %.1f < 3: MAX_STEPS scaled to %d' % (args.vmax, MAX_STEPS), flush=True)
    print(f"[task {args.task}] v0={FOREST_V0} corridor {X0}..{X_GOAL} "
          f"half_w={HALF_W} dt_evt={DT_EVT} n_obs={T['N_OBS']}", flush=True)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import omni.usd
    import torch
    from events.warp_events import (KeyframeCamera, WarpEventStream,
                                            downsample_counts)

    stage = omni.usd.get_context().get_stage()
    scene = ForestStage(stage, scene_style=args.scene_style,
                        wall_style=args.wall_style,
                        lighting=args.lighting)
    cam = KeyframeCamera(res=(args.res, args.res), fov_deg=90.0,
                         event_source="hdr").build()
    tp_cam = None
    if args.third_person:
        # HDR + own tone-mapping: the LDR annotator renders the sun-lit scene
        # nearly black (measured mean 6.7/255). Autoexpose per frame instead.
        tp_cam = KeyframeCamera(res=(256, 256), fov_deg=70.0, event_source="hdr",
                                prim_path="/World/tp_cam").build()

        def tp_grab():
            rgb = tp_cam.capture()[0].cpu().numpy()          # linear HDR float
            hi = max(float(np.percentile(rgb, 99.0)), 1e-6)
            img = np.clip(rgb / hi, 0.0, 1.0) ** (1.0 / 2.2)  # autoexpose+gamma
            return (img * 255).astype(np.uint8)
    NOISE = dict(sigma_threshold=0.03, refractory_s=1e-3, leak_rate_hz=0.1,
                 shot_rate_hz=0.5, hot_pixel_frac=2e-4, hot_pixel_rate_hz=300.0,
                 intensity_scale=1.0) if args.noise == "realistic" else None

    K = args.warp
    WIN_TICKS = int(round(DT_EVT / STEER_DT))          # 3 ticks = 30 ms
    rng_master = np.random.RandomState(args.seed)

    # ---------------- event->obs machinery -----------------------------------
    stream = WarpEventStream(args.res, args.res, threshold=args.event_c,
                             noise=NOISE)
    stream.window_frames = K                            # one window per tick

    def tick_counts():
        """Drain the per-tick windows accumulated by the last push (list of 1)."""
        wins = stream.windows
        stream.windows = []
        return wins

    class RollingObs:
        def __init__(self):
            self.buf = []

        def add(self, on_off):
            self.buf.append(on_off)
            if len(self.buf) > WIN_TICKS:
                self.buf.pop(0)

        def cancel_ratio(self):
            """Intra-window cancellation: 1 - |signed sum| / unsigned sum over
            the window's fine frames (active pixels only). The proxy renders
            one displacement per window (no cancellation), the DIFF3D
            accumulator and this bridge cancel — this measures how much."""
            nets = [(b[0] - b[1]) for b in self.buf]
            signed = torch.stack(nets).sum(0).abs()
            unsigned = torch.stack([n.abs() for n in nets]).sum(0)
            m = unsigned > 0
            if not m.any():
                return 0.0
            return float(1.0 - (signed[m].sum() / unsigned[m].sum()))

        def frame_net(self):
            """Signed native-res net (ON - OFF) over the window -> numpy int32.
            Models the DIFF3D accumulator exactly, including intra-window
            oscillation cancellation (which the training proxy cannot)."""
            on = sum(b[0] for b in self.buf)
            off = sum(b[1] for b in self.buf)
            return (on - off).cpu().numpy().astype(np.int32)

        def frame64(self):
            """Sum last WIN_TICKS ticks -> (obs_res, obs_res, 2) float counts
            (OFF, ON order per --polarity)."""
            on = sum(b[0] for b in self.buf)
            off = sum(b[1] for b in self.buf)
            hw = (args.obs_res, args.obs_res)
            on64 = downsample_counts(on, hw).float()
            off64 = downsample_counts(off, hw).float()
            if args.polarity == "off_on":
                return torch.stack([off64, on64], -1)
            return torch.stack([on64, off64], -1)

    # ---------------- kinematics (thin wrappers over module-level logic) -------
    args.act_floats = 3 if args.speed_mode == "cmd3" else 2
    VARIABLE_V = args.speed_mode != "fixed"
    if args.speed_mode == "cmd2rate" and args.steer_mode != "rate":
        ap.error("--speed-mode cmd2rate requires --steer-mode rate "
                 "(a[0]=yaw rate, a[1]=speed)")
    KIN_CFG = dict(v_const=FOREST_V0, yawrate_max=FOREST_YAWRATE_MAX,
                   z_fly=Z_FLY, x_goal=X_GOAL, mode=args.speed_mode,
                   steer=args.steer_mode, yaw_tau=args.yaw_tau,
                   vmin=args.vmin, vmax=args.vmax, accel_max=args.accel_max,
                   freeze_bearing=args.freeze_bearing,
                   freeze_vector=([float(x) for x in args.freeze_vector.split(",")]
                                  if args.freeze_vector else None))

    def step_kin(state, action):
        return kin_step(state, action, KIN_CFG)

    def proprio(state):
        return frozen_or_proprio(state, KIN_CFG)

    def check_done(prev_p, p, layout):
        return done_flags(p, layout, (OOB_W or HALF_W), X_GOAL,
                          point_goal=T.get('SQROOM', False), x0=X0)

    last_depth = [None]

    def render_tick(state, t):
        cam.set_pose(tuple(state["p"]), yaw_deg=math.degrees(state["psi"]))
        KeyframeCamera.render()
        rgb, mv, depth = cam.capture()
        if args.depth_obs:
            last_depth[0] = depth.cpu().numpy()
        stream.push(rgb, mv, depth, t=t, K=K)
        return tick_counts()

    # ---------------- gain calibration (scripted pre-flight) ------------------
    layout0 = sample_forest_layout(rng_master, T)
    scene.build_episode(layout0, rng_master)
    if args.stills > 0:
        import numpy as _np, math as _m
        os.makedirs(args.out, exist_ok=True)
        iso = KeyframeCamera(res=(960, 720), fov_deg=55.0, event_source="hdr",
                             prim_path="/World/iso_cam").build()
        cxm = (X0 + X_GOAL) / 2.0
        # interior poses: the hall is enclosed, so overviews shoot from inside,
        # below the ceiling (CEIL 5.2 in hangar style)
        poses = [((X0 + 1.5, -HALF_W + 1.5, 4.6), 38.0, -28.0),
                 ((X_GOAL - 1.5, HALF_W - 1.5, 4.6), 218.0, -28.0),
                 ((cxm, -HALF_W + 1.0, 4.3), 90.0, -22.0),
                 ((X0 + 1.0, 0.0, 3.4), 0.0, -13.0),
                 ((cxm, 0.0, 4.95), 90.0, -80.0)]
        for i in range(min(args.stills, len(poses))):
            (px, py, pz), yaw, pit = poses[i]
            iso.set_pose((px, py, pz), yaw_deg=yaw, pitch_deg=-pit if False else -pit)
            for _ in range(24):
                KeyframeCamera.render()
            rgb = iso.capture()[0].cpu().numpy()
            hi = max(float(_np.percentile(rgb, 99.2)), 1e-6)
            img = (_np.clip(rgb / hi, 0, 1) ** (1 / 2.2) * 255).astype(_np.uint8)
            from PIL import Image as _Im
            _Im.fromarray(img).save(os.path.join(args.out, f"iso_{i}.png"))
            print(f"[stills] saved iso_{i}.png", flush=True)
        sys.exit(0)
    gain = args.gain
    if gain <= 0:
        st = {"p": np.array([X0, 0.0, Z_FLY]), "psi": 0.0, "yaw_rate": 0.0}
        roll = RollingObs()
        stream.reset()
        vals = []
        genx_nets = []
        t = 0.0
        for _ in range(3):                              # settle + reference
            cam.set_pose(tuple(st["p"]), yaw_deg=0.0)
            KeyframeCamera.render()
        for i in range(120):                            # 1.2 s straight flight
            t += STEER_DT
            st = step_kin(st, straight_action(args.steer_mode, args.act_floats))
            for w in render_tick(st, t):
                roll.add(w)
            if len(roll.buf) == WIN_TICKS:
                if args.genx:
                    net = roll.frame_net()
                    genx_nets.append(net)
                    a = np.abs(net)
                    nz = a[a > 0]
                else:
                    f = roll.frame64().cpu().numpy()
                    if args.min_count > 0:
                        f = f * (f >= args.min_count)
                    nz = f[f > 0]
                if nz.size:
                    vals.append(nz)
        allv = np.concatenate(vals)
        med = float(np.median(allv))
        # proxy: image_u8 = clip(frame*32, 0, 255); frame units ~= proxy counts.
        # choose gain so median nonzero count -> PROXY_MEDIAN_NZ/32 proxy units.
        if args.genx:
            if args.genx_p90_target > 0:
                gain = genx_contrast_for_p90(
                    genx_nets, args.genx_p90_target, args.obs_res)
                px90 = args.genx_p90_target
                print(f"[calib-genx] p90-target {px90:.0f} -> contrast "
                      f"{gain:.3f} (median |net|/native-px was {med:.1f})",
                      flush=True)
            else:
                # median convention: ~1 count per pixel the edge sweeps
                gain = args.genx_contrast / max(med, 1e-6)
                print(f"[calib-genx] median |net| per active native px = "
                      f"{med:.1f} -> effective contrast {gain:.2f} "
                      f"(stock 16 / med)", flush=True)
        else:
            gain = (args.calib_target / 32.0) / max(med, 1e-6)
        act = float(np.mean([(v > 0).mean() for v in
                             [roll.frame64().cpu().numpy()]]))
        print(f"[calib] median nonzero 64-bin count={med:.1f} -> gain={gain:.4f} "
              f"(last-frame active={act*100:.1f}%, proxy active: mean 25%, p10-p90 4-57% at dt=0.10)", flush=True)

    diag_cancel = []

    def obs_image(roll):
        if args.depth_obs:
            d = last_depth[0]
            b = 320 // args.obs_res
            sub = d[b // 2::b, b // 2::b][:args.obs_res, :args.obs_res]
            inv = (1.0 / np.clip(sub, 0.3, 25.0) - 1.0 / 25.0) / (1.0 / 0.3 - 1.0 / 25.0)
            return np.clip(1020.0 * inv, 0.0, 255.0).astype(np.uint8)[..., None]
        if args.genx:
            if args.genx_diag:
                diag_cancel.append(roll.cancel_ratio())
                if len(diag_cancel) % 50 == 0:
                    a = np.array(diag_cancel)
                    print(f"[diag-cancel] n={len(a)} mean={a.mean():.3f} "
                          f"p90={np.percentile(a,90):.3f} max={a.max():.3f}",
                          flush=True)
            net = roll.frame_net() * args.genx_sign
            return genx_obs(net, gain, args.obs_res)
        f = roll.frame64()
        if args.min_count > 0:
            f = f * (f >= args.min_count)               # cut the noise floor
        f = f * gain                                    # -> proxy frame units
        img = (f * 32.0).clamp(0, 255).to(torch.uint8)  # proxy uint8 mapping
        out = img.cpu().numpy()
        return out[:, ::-1, :].copy() if args.flip_img else out

    # ---------------- socket ---------------------------------------------------
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[server] waiting for policy client on :{args.port} ...", flush=True)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("[server] client connected", flush=True)

    def send_obs(flags, reward, vec, img):
        conn.sendall(pack_obs(flags, reward, vec, img))

    ACT_BYTES = 4 * args.act_floats

    def recv_action():
        buf = b""
        while len(buf) < ACT_BYTES:
            chunk = conn.recv(ACT_BYTES - len(buf))
            if not chunk:
                raise ConnectionError("client closed")
            buf += chunk
        return struct.unpack(f"<{args.act_floats}f", buf)

    # ---------------- episodes -------------------------------------------------
    results = []
    for ep in range(args.episodes):
        rng = np.random.RandomState(args.seed * 1000 + ep)
        layout = layout0 if ep == 0 else sample_forest_layout(rng, T)
        if ep > 0:
            scene.build_episode(layout, rng)
        v0_ep = args.vmin if VARIABLE_V else FOREST_V0
        if T.get('SQROOM'):
            # full-room spawn + best-of-8 clearance (mirrors QC.reset_sqroom)
            sxlo, sxhi = T.get('SPAWN_X', (X0 + 0.5, X_GOAL - 5.0))
            cand = np.stack([rng.uniform(sxlo, sxhi, 8),
                             rng.uniform(-(HALF_W - 1.5), HALF_W - 1.5, 8)], -1)
            if len(layout['xy']):
                dmin = np.linalg.norm(cand[:, None] - layout['xy'][None], axis=-1).min(1)
            else:
                dmin = np.ones(8)
            j = int(np.argmax(dmin))
            psi0 = rng.uniform(-math.pi, math.pi)
            state = {"p": np.array([cand[j, 0], cand[j, 1], Z_FLY]), "psi": psi0,
                     "yaw_rate": 0.0, "v": v0_ep}
        else:
            # reset like reset_steer: y0 in +-0.3*half_w, psi0 in +-0.15
            y0 = (args.y0 if args.y0 is not None
                  else rng.uniform(-0.3 * HALF_W, 0.3 * HALF_W))
            if args.psi0_deg is not None:
                psi0 = math.radians(args.psi0_deg) + rng.uniform(-0.05, 0.05)
            else:
                psi0 = rng.uniform(-0.15, 0.15)
            state = {"p": np.array([X0, y0, Z_FLY]), "psi": psi0, "yaw_rate": 0.0,
                     "v": v0_ep}
        psi0 = state["psi"]
        stream.reset()
        roll = RollingObs()
        # pre-roll: fly the 3 ticks BEFORE x0 scripted so the first policy obs
        # has a full 30 ms live window (the analytic proxy is never empty)
        pre = {"p": state["p"] - np.array(
            [math.cos(psi0), math.sin(psi0), 0.0]) * v0_ep * STEER_DT * WIN_TICKS,
            "psi": psi0, "yaw_rate": 0.0, "v": v0_ep}
        cam.set_pose(tuple(pre["p"]), yaw_deg=math.degrees(psi0))
        for _ in range(3):
            KeyframeCamera.render()
        t = ep * 100.0                                   # monotonic across episodes
        # reference push (first push after reset() only latches the reference
        # frame and emits no window) so all WIN_TICKS pre-roll ticks count
        stream.push(*cam.capture(), t=t, K=1)
        for _ in range(WIN_TICKS):
            t += STEER_DT
            pre = step_kin(pre, straight_action(args.steer_mode, args.act_floats))
            for w in render_tick(pre, t):
                roll.add(w)
        state["p"][:2] = pre["p"][:2]                    # exact continuity
        state["v"] = pre["v"]        # carry the pre-roll's integrated speed too
        # (in mag modes the scripted (1,0) pre-roll accelerates slightly; the
        # first policy tick must continue from that speed, not snap back)

        traj = [state["p"].copy()]
        speeds = []
        dbg_frames = []
        tp_frames = []
        reward_prev = 0.0            # summed over the held-action span
        prev_dist = float(np.linalg.norm(state["p"][:2] - [X_GOAL, 0.0]))
        status = "timeout"
        a = straight_action(args.steer_mode, args.act_floats)
        t_ep0 = time.perf_counter()
        for stepi in range(MAX_STEPS):
            if args.act_hold_alt:
                if stepi == 0:
                    _alt = [int(x) for x in args.act_hold_alt.split(',')]
                    _next_dec, _alt_i = 0, 0
                _is_dec = stepi == _next_dec
                if _is_dec:
                    _next_dec += _alt[_alt_i % len(_alt)]
                    _alt_i += 1
            else:
                _is_dec = stepi % args.act_hold == 0
            if _is_dec:                      # policy query tick
                img = obs_image(roll)
                vec = proprio(state)
                flags = 1 if stepi == 0 else 0
                send_obs(flags, reward_prev, vec, img)
                reward_prev = 0.0
                a = recv_action()
                if args.save_frames and ep < args.save_frames:
                    dbg_frames.append((img.copy(), vec.copy(),
                                       np.array(a, np.float32),
                                       state["p"].copy(), state["psi"]))
            if tp_cam is not None:
                # chase pose EVERY tick (cheap USD write) so the last render is
                # <=1 tick stale; capture every 2nd decision only.
                cp = state["p"]; ps = state["psi"]
                tp_cam.set_pose((cp[0] - 2.6 * math.cos(ps),
                                 cp[1] - 2.6 * math.sin(ps), cp[2] + 1.1),
                                yaw_deg=math.degrees(ps), pitch_deg=18.0)
                if stepi % ((args.act_hold if not args.act_hold_alt else 15) * 2) == 0:
                    tp_frames.append(tp_grab())
            prev_p = state["p"].copy()
            state = step_kin(state, a)
            t += STEER_DT
            for w in render_tick(state, t):
                roll.add(w)
            collided, oob, reached = check_done(prev_p, state["p"], layout)
            dist = float(np.linalg.norm(state["p"][:2] - [X_GOAL, 0.0]))
            reward_prev += 5.0 * (prev_dist - dist) - 10.0 * (collided or oob) \
                + 30.0 * reached
            prev_dist = dist
            traj.append(state["p"].copy())
            speeds.append(state["v"])
            if args.no_done:
                collided = oob = reached = False
            if collided or oob or reached:
                status = "reached" if reached else ("collided" if collided else "oob")
                break
        # terminal obs so the client can close the episode cleanly
        flags = 2 | (4 if status != "timeout" else 0)
        send_obs(flags, reward_prev, proprio(state), obs_image(roll))
        wall = time.perf_counter() - t_ep0
        results.append((status, state["p"][0]))
        vtxt = (f" v_mean={np.mean(speeds):.2f} v_max={np.max(speeds):.2f}"
                if VARIABLE_V and speeds else "")
        print(f"[ep {ep:02d}] {status:9s} x_final={state['p'][0]:5.2f}/{X_GOAL:.0f} "
              f"steps={len(traj)-1} wall={wall:.1f}s "
              f"({wall/max(len(traj)-1,1)*1e3:.0f} ms/tick){vtxt}", flush=True)
        zm = layout.get("zones")
        np.savez(os.path.join(args.out, f"ep{ep:03d}.npz"),
                 traj=np.array(traj), status=status,
                 speeds=np.array(speeds, np.float32),
                 zone_edges=np.array(zm["edges"]) if zm else np.zeros(0),
                 zone_factors=np.array(zm["factors"]) if zm else np.zeros(0),
                 obs_xy=layout["xy"], obs_r=layout["r"], obs_h=layout["h"],
                 branches=np.array(layout["branches"]) if layout["branches"]
                 else np.zeros((0, 5)),
                 frames=np.array([f[0] for f in dbg_frames], dtype=np.uint8)
                 if dbg_frames else np.zeros((0,), np.uint8),
                 vecs=np.array([f[1] for f in dbg_frames]) if dbg_frames else 0,
                 acts=np.array([f[2] for f in dbg_frames]) if dbg_frames else 0,
                 tp=np.array(tp_frames, dtype=np.uint8) if tp_frames
                 else np.zeros((0,), np.uint8))

    conn.sendall(b"Q")
    n_reach = sum(1 for s, _ in results if s == "reached")
    mean_x = np.mean([x for _, x in results])
    print(f"[RESULT] reach {n_reach}/{len(results)} = "
          f"{100*n_reach/len(results):.0f}%  mean final x = {mean_x:.2f}/{X_GOAL:.0f}",
          flush=True)
    with open(os.path.join(args.out, "results.txt"), "w") as fh:
        for i, (s, x) in enumerate(results):
            fh.write(f"{i} {s} {x:.3f}\n")
        fh.write(f"reach {n_reach}/{len(results)}\n")
    conn.close()
    srv.close()
    app.close()


if __name__ == "__main__":
    main()
