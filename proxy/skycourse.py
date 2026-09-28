"""SkyDreamer-CF multi-obstacle COURSE env for official DreamerV3.

Observation (two keys, as designed):
  image  : per-class detection heatmap  [64,64,8] uint8  (Option A)
  vector : per-class metric slots [40] + proprio [7] = [47] float32  (Option B)
The detections are SIMULATED from ground-truth geometry (fast, no rendering) with
detector-style domain randomization (box jitter / dropout) as the sim2real bridge.
Action = 4-D CTBR in [-1,1]. Course + traversal reward from quad_course_env.

Runs the sim on CPU under a transfer guard.
"""
import re
import sys
import pathlib

import numpy as np
import jax
import jax.numpy as jnp

_ENVS = pathlib.Path(__file__).resolve().parent
if str(_ENVS) not in sys.path:
    sys.path.insert(0, str(_ENVS))
import spaces as elements          # noqa: E402  (Space / Env stand-ins)
import quad_ctbr_env as E          # noqa: E402
import quad_course_env as QC       # noqa: E402
import course_sim as CS            # noqa: E402
import mask_render as MR           # noqa: E402


class SkyCourse(elements.Env):

    def __init__(self, task='course', mask_H=64, fov_deg=120.0, max_steps=None,
                 seed=0, k_prog=5.0, jitter=0.02, dropout=0.08):
        self._cpu = jax.devices('cpu')[0]
        self.H = mask_H
        self.C = CS.N_CLASSES
        # curriculum: task 'courseN' -> first N traversal targets (full course = 7).
        # e.g. skycourse_course2 trains only window + red bar.
        n = 7
        # 'landonly' = LANDING PRE-TRAINING: full geometry, but each episode starts at the
        # tube exit already on the LAND target, so the drone drills ONLY the final 1.9 m
        # (the segment where it tumbles). Short episodes => many attempts per hour.
        self.landonly = (task == 'landonly')
        # 'coursem2' = MISSION 1 + MISSION 2: the 6 M1 obstacles then the dark room
        # (fly IN red window -> deep inside -> turn ~180 -> fly OUT blue window). No landing.
        # 9 targets; needs extra steps for the confined turn.
        self.mission2 = (task == 'coursem2')
        # CIRCLE = HIL milestone: orbit a 2 m circle past 4 distinct landmarks (classes 0-3).
        # Always vector-only (it's the deployable NPU-friendly obs). No traversal/curriculum.
        self.circle = task.startswith('circle')
        # AVOID = obstacle avoidance via LPTC wide-field optic-FLOW summary (insect-style).
        # obs = 6 flow scalars [div,L,R,U,D,mean] + proprio; reactive, no localisation.
        self.avoid = task.startswith('avoid')
        self.flow_dim = QC.FLOW_N_SECT + 2   # 8 azimuth sectors + div + loom_max = 10
        # STEER = 2D fixed-height HEADING-STEERING avoidance with EVENT-FRAME obs (event_proxy).
        # action = 2D unit-circle heading (egocentric); obs = [64,64,2] event frame + [3] proprio
        # (egocentric goal bearing + yaw rate). Kinematic, 100 Hz, sparse poles in a narrow corridor.
        # 'steerN' -> N poles (default 4). See quad_course_env.make_steer/step_steer.
        # FOREST = denser/varied STEER (poles+trees+branches+visible WALLS, 15m @3m/s). Same obs/act.
        # SPEEDFOREST (round 3, 2026-08-07) = forest + VARIABLE SPEED as a third action.
        # Fixed 3 m/s is the binding constraint in dense clutter (turn radius 0.86 m vs ~1 m gaps);
        # at 2x density both fixed-speed models collapse to ~10-13%. Zoned densities within each
        # episode force accelerate-AND-brake in one flight. Action [3], proprio [4] (speed appended
        # -> cannot seed from forest13, hence from scratch). See quad_course_env SPEEDFOREST block.
        # Variant A `speedforest` = 3-D action (explicit speed), proprio[4], from scratch.
        # Variant B `magforest`   = 2-D action, |a| -> speed, proprio[3] => SEEDS FROM forest13.
        # Both share the same 22 m zoned-density course and round-2 DR; only the action mapping
        # differs (QC.step_speedforest vs QC.step_magforest).
        # EVENT-FRAME RESOLUTION suffix 'h<N>' (2026-08-11 deploy lever): `speedzoneh32`,
        # `magzone24h32`, `forest13h32`. The 64x64 policy runs at 2.4 Hz on the N6 and the CNN
        # encoder is the bottleneck; 32x32 cuts first-layer conv work ~4x. Stripped BEFORE any
        # other parsing so the obstacle-count suffix still works. Absent -> 64 (unchanged).
        # ACTION REPEAT suffix 'r<N>' + the DEPLOY PROFILE alias trailing 's' (round D,
        # 2026-08-11). On the N6 the CNN policy runs at ~4-10 Hz while the Crazyflie's inner
        # loops run far faster; training at 100 Hz and deploying at 4-10 Hz is a train/deploy
        # rate mismatch. 'r3' holds each action for 3 kinematic ticks (policy sees every 3rd
        # tick ~= 33 Hz) and SUMS the rewards over the span, so returns stay comparable.
        # `forest13s` == `forest13h32r3`: the deploy-optimized profile in one name.
        # A suffix may only be stripped when what REMAINS is still a well-formed task name.
        # Without this guard `steer5` parses as base 'stee' + repeat 5 -- it silently loses the
        # steer branch entirely (image becomes [64,64,8], DR off). Any 'r'/'h' immediately
        # before trailing digits is ambiguous, so anchor on the known task prefixes.
        _PREFIXES = ('steer', 'forest', 'magforest', 'magzone', 'speedforest', 'speedzone',
                     'ratezone', 'yawzone', 'course', 'circle', 'avoid', 'landonly')
        _wellformed = lambda t: any(t.startswith(p) for p in _PREFIXES)

        _deploy = False
        if task.endswith('s') and _wellformed(task[:-1]):
            _deploy, task = True, task[:-1]
        self.act_repeat = 1
        self.event_h = QC.STEER_H
        self.event_dt_ms = None          # explicit integration window, 'w<N>' in ms
        self.genx = False                 # `gx` suffix: single-channel GenX320 DIFF3D obs (see below)
        self.scenedr = False              # `sc` suffix: per-episode scene-type DR (corridor/open/half/messy)
        self.sqroom = False               # `sq` suffix: 22x22 square room, white walls, point goal
        self.vmax_ovr = None              # `v<N>` suffix: max commanded speed in m/s
        _repeat_explicit = False
        while True:                       # suffixes may be combined in either order
            if task.endswith('sc') and _wellformed(task[:-2]):
                self.scenedr, task = True, task[:-2]
            if task.endswith('sq') and _wellformed(task[:-2]):
                self.sqroom, task = True, task[:-2]
            if task.endswith('gx') and _wellformed(task[:-2]):
                self.genx, task = True, task[:-2]
                continue
            m = re.match(r'^(?P<base>.+?)v(?P<n>\d+)$', task)
            if m and _wellformed(m.group('base')):
                self.vmax_ovr, task = float(m.group('n')), m.group('base')
                continue
            m = re.match(r'^(?P<base>.+?)r(?P<n>\d+)$', task)
            if m and _wellformed(m.group('base')):
                self.act_repeat, task = int(m.group('n')), m.group('base')
                _repeat_explicit = True
                continue
            m = re.match(r'^(?P<base>.+?)h(?P<n>\d+)$', task)
            if m and _wellformed(m.group('base')):
                self.event_h, task = int(m.group('n')), m.group('base')
                continue
            m = re.match(r'^(?P<base>.+?)w(?P<n>\d+)$', task)
            if m and _wellformed(m.group('base')):
                self.event_dt_ms, task = int(m.group('n')), m.group('base')
                continue
            break
        if _deploy:                       # alias defaults, only where not set explicitly
            if self.event_h == QC.STEER_H:
                self.event_h = 32
            if not _repeat_explicit:
                self.act_repeat = 3
        if (task.startswith('ratezone') or task.startswith('yawzone')) and not _repeat_explicit:
            # The rate head and the action repeat are SYNERGISTIC, not independent: the repeat is
            # what kills the dither (a held dither phase is a crash, so the policy must emit
            # commands it means). The directive says do not ship one without the other, so
            # ratezone defaults to 3 -- still overridable with an explicit r<N>.
            self.act_repeat = 3
        # ROUND 4 (`magzone`/`speedzone`): EQUAL-width zones, density by tree count. Round 3's
        # `magforest`/`speedforest` scaled zone WIDTH by 1/density, making the dense zone ~2 m
        # (0.37 s) — too brief to make braking worth learning; it failed the Isaac dense gate at
        # 10%. Kept as separate task names so magforest1/2 stay reproducible.
        # ROUND 5-REVISED `ratezone`: a0 IS the yaw rate (no atan2/P-law), a1 IS the speed. Action
        # stays [2] but proprio is [4] (bearing, yawrate, v/vmax) -- a combination no earlier task
        # had. Replaces the heading parametrization, which the trained policies were only using as
        # a dithered encoding of a rate signal. See quad_course_env.step_ratezone.
        # ROUND 7 (`yawzone`): identical to ratezone (rate head, zone course, act-repeat 3) but
        # with round-7 resets -- full-circle initial yaw and forced dense-zone starts at vmax.
        # Separate name so ratezone1/3 stay reproducible.
        self.yawdr = task.startswith('yawzone')
        self.ratevar = task.startswith('ratezone') or self.yawdr
        self.zone4 = (task.startswith('magzone') or task.startswith('speedzone')
                      or self.ratevar)
        self.speedvar = task.startswith('speedforest') or task.startswith('speedzone')
        self.magvar = task.startswith('magforest') or task.startswith('magzone')
        _pre = ('speedforest' if task.startswith('speedforest') else
                'speedzone' if task.startswith('speedzone') else
                'yawzone' if self.yawdr else
                'ratezone' if self.ratevar else
                'magforest' if task.startswith('magforest') else 'magzone')
        _sbase = (task[len(_pre):].replace('wall', '')
                  if (self.speedvar or self.magvar or self.ratevar) else '')
        self.speed_n = int(_sbase) if _sbase.isdigit() else (18 if self.zone4 else 12)
        if self.sqroom:
            self.speed_n = QC.SQROOM_N    # trees over the whole 22x22 floor
        self.forest = task.startswith('forest')
        _fbase = task[6:].replace('wall', '') if self.forest else ''
        self.forest_n = int(_fbase) if _fbase.isdigit() else 12
        self.forest_pwall = 1.0 if ('wall' in task) else 0.1   # 'forestwall' -> a wall EVERY episode
        # forest AND speedforest reuse the steer machinery (event obs, heading action, reward)
        self.steer = (task.startswith('steer') or self.forest or self.speedvar
                      or self.magvar or self.ratevar)
        self.steer_n = int(task[5:]) if (task.startswith('steer') and task[5:].isdigit()) else 4
        if self.steer:
            # The event frame IS the image observation, so obs_space and the renderer must agree
            # on one size. Previously obs_space used mask_H while the renderer used QC.STEER_H --
            # equal only by coincidence at 64.
            self.H = self.event_h
        # REAL-EVENT NUISANCE DR (2026-08-06) — tasks `steer5` / `forest13`(+`wall`).
        # The Isaac closed-loop eval on physically-rendered events showed the proxy-trained
        # policies fail on event sources the proxy never renders (trunk ground shadows, horizon
        # flicker, dense textured-wall event mass, RTX count noise). These tasks add all four as
        # analytic DR, and steer5 additionally makes the corridor WALLS visible (steer4 had none
        # and scored 0/30 on real events). The OLD tasks steer4/forest12 are deliberately left
        # untouched so their checkpoints/evals stay reproducible for the comparison tables.
        self.nuisance_dr = ((task == 'steer5') or (self.forest and self.forest_n == 13)
                            or self.speedvar or self.magvar or self.ratevar)
        # VECTOR-ONLY MODE (task suffix 'vec', e.g. 'course3vec' / 'coursem2vec').
        # The heatmap is DRAWN FROM the same detection slots as `vector`, so it carries no extra
        # information (position is quantised onto a 64x64 grid; the vector holds exact floats).
        # Dropping it removes 99.4% of per-step replay bytes (32,768 -> 188), the CNN encoder and
        # the image decoder. Default stays 'both' so every existing task/checkpoint is unaffected.
        # ACTION LOW-PASS (task suffix 'filt', e.g. 'course2vecfilt'). The DreamerV3 actor learns
        # near-white-noise CTBR commands (autocorr ~0.19) that the motor lag averages into flight;
        # that jitter is load-bearing but UN-DISTILLABLE (every behaviour-cloning student flew 0/20)
        # and un-trackable by the real CF rate loop. Filtering the action INSIDE the env before
        # `_ctbr_to_u` makes the policy learn to drive a filtered actuator -> its command becomes a
        # SMOOTH, imitable function of state. alpha=0.3 => tau~26ms (near the 50ms motor constant).
        self.action_lp = 0.0
        if task.endswith('filt'):
            self.action_lp = 0.3
            task = task[:-4]
        if task.endswith('vec'):
            self.obs_mode = 'vector'
            task = task[:-3]
            self.mission2 = (task == 'coursem2')
        else:
            self.obs_mode = 'both'
        if task.startswith('course') and task[6:].isdigit():
            n = int(task[6:])
        if self.circle or self.avoid:
            self.obs_mode = 'vector'          # deployable obs; no heatmap for circle/avoid
        if self.steer:
            self.obs_mode = 'event'           # [H,W,2] event frame + [3] proprio (handled specially)
        self.n_targets = 9 if self.mission2 else n
        # ~180 steps of headroom per obstacle (they're ~1-2 m apart) + margin;
        # the M2 turn is slow, so give the mission2 course a generous cap.
        # circle: continuing orbit task ~600 steps; avoid: short corridor traversal ~300.
        self.max_steps = max_steps if max_steps is not None else (
            QC.SPEED_MAX_STEPS if (self.speedvar or self.magvar or self.ratevar) else  # 22m/900
            (900 if self.forest else 700) if self.steer else  # forest 15m@3m/s~500; steer 6m@1.2~500
            300 if self.avoid else 600 if self.circle else
            300 if self.landonly else (1900 if self.mission2 else 250 + 130 * n))
        # self._steps counts POLICY steps, so an action repeat must shorten the cap or the
        # episode would cover act_repeat x the simulated time (and the timeout, which is the
        # only time pressure in the variable-speed tasks, would stop biting).
        if self.act_repeat > 1:
            self.max_steps = max(1, self.max_steps // self.act_repeat)
        if self.scenedr:
            # worst case: 50 m at vmin ~0.5 m/s = 100 s = 10000 kinematic ticks
            self.max_steps = max(self.max_steps, 10500 // self.act_repeat)
        if self.sqroom:
            # room diagonal ~24 m at vmin 1.0 + avoidance detours -> 36 s budget
            self.max_steps = max(self.max_steps, 3600 // self.act_repeat)
        with jax.transfer_guard('allow'), jax.default_device(self._cpu):
            self.pm = E.Params()
            self.M_inv = E._mixer_inv(self.pm)
            self.course = (QC.make_zoneforest(n_obs=self.speed_n) if self.zone4 else
                           QC.make_speedforest(n_obs=self.speed_n)
                           if (self.speedvar or self.magvar or self.ratevar) else
                           QC.make_forest(n_obs=self.forest_n, p_wall=self.forest_pwall,
                                          **(dict(wall_nx=QC.FOREST_WALL_NX, wall_nz=QC.FOREST_WALL_NZ)
                                             if self.nuisance_dr else {})) if self.forest else
                           QC.make_steer(n_obs=self.steer_n, walls=self.nuisance_dr) if self.steer else
                           QC.make_avoid() if self.avoid else
                           QC.make_circle() if self.circle else
                           QC.make_course(n_targets=n, mission2=self.mission2))
            # `v<N>` MAX SPEED OVERRIDE. The time budget is the only pressure in the
            # variable-speed tasks, so it must be rescaled with vmax or a slower course becomes
            # timeout-bound rather than obstacle-bound: at 22 m the corridor needs 3.7 s at
            # 6 m/s against a 9 s cap, but 7.3 s at 3 m/s -- nearly the whole budget. Holding the
            # slack RATIO fixed keeps the task the same problem, just flown slower.
            if self.sqroom:
                self.course = dict(self.course)
                self.course['half_w'] = QC.SQROOM_HW
                self.course['point_goal'] = True
            if self.vmax_ovr is not None:
                assert 'vmax' in self.course, f'task {task!r} has no vmax to override'
                base_vmax = float(self.course['vmax'])
                self.course['vmax'] = jnp.float32(self.vmax_ovr)
                # vmin must stay meaningfully below vmax or the speed action has nothing to do
                # and the task silently becomes fixed-speed. Cap it at half of vmax: for the
                # trained v3 course that is min(1.5, 1.5) = 1.5, i.e. EXACTLY what
                # artifacts/GX2_15hz_3ms was trained on, so this cannot alter a banked task.
                vmin_new = min(float(self.course['vmin']), 0.5 * self.vmax_ovr)
                if vmin_new != float(self.course['vmin']):
                    self.course['vmin'] = jnp.float32(vmin_new)
                    self.course['v0'] = jnp.float32(vmin_new)
                self.max_steps = int(round(self.max_steps * base_vmax / self.vmax_ovr))

            # FRAME CONTRACT (round 8, 2026-08-17): the event integration window must equal the
            # POLICY PERIOD, because that is what the firmware will do -- one event read per
            # decision. Getting this wrong is the forest13s lesson: a frame-generation mismatch
            # that looks fine in-proxy and scores 0/30 on Isaac.
            #   act_repeat 3  -> 0.03 s  (exactly the value every existing task already uses,
            #                             so this is a no-op for them)
            #   act_repeat 10 -> 0.10 s  (the ~10 Hz board rate)
            # Applied only when act_repeat > 1, so the 100 Hz tasks keep their hand-tuned 0.03.
            # ROUND 8.1 (2026-08-18): the DECISION PERIOD and the INTEGRATION WINDOW are
            # INDEPENDENT -- the N6 histogram window is configurable separately from the policy
            # rate, and DEPLOYMENT.md specifies a 30 ms integration. Round 8 tied them together
            # (dt_evt = act_repeat * dt), which at 10 Hz produced 0.10 s frames that proved
            # structurally untransferable: Isaac frames at 100 ms sit at the p90 edge of that
            # training distribution (zoned 23% vs 67% at 33 Hz, level-1 gap -50). A 30 ms window
            # keeps Isaac frames near the CENTRE of the validated distribution and discards 70 ms
            # of motion smear that blurs obstacle bearing.
            #   'w<N>' sets the window explicitly in ms and WINS.
            #   Otherwise the round-8 default (window == policy period) still applies.
            if 'dt_evt' in self.course:
                if self.event_dt_ms is not None:
                    self.course = dict(self.course)
                    self.course['dt_evt'] = jnp.float32(self.event_dt_ms / 1000.0)
                elif self.act_repeat > 1:
                    self.course = dict(self.course)
                    self.course['dt_evt'] = jnp.float32(self.act_repeat * QC.STEER_DT)
            self._fov_h = float(np.deg2rad(fov_deg))
            self._f = float(MR.focal_from_fov(fov_deg, mask_H))
            self._key = jax.random.PRNGKey(seed)
        pm, M_inv, course = self.pm, self.M_inv, self.course
        H, f = self.H, self._f
        dr = {"jitter": jitter, "dropout": dropout}

        landonly = self.landonly
        circle = self.circle
        avoid = self.avoid
        steer = self.steer
        speedvar_r, magvar_r = self.speedvar, self.magvar
        fov_h = self._fov_h
        yawdr_r = self.yawdr
        sqroom_r = self.sqroom
        @jax.jit
        def _reset(key, obs_xy=None):
            if sqroom_r:
                return QC.reset_sqroom(key, course, obs_xy)  # full-room spawn + clearance
            if yawdr_r:
                return QC.reset_yawdr(key, course, obs_xy)   # full-circle yaw + forced encounters
            if speedvar_r or magvar_r:
                return QC.reset_speedvar(key, course)   # RANDOM initial speed (required, see QC)
            if steer:
                return QC.reset_steer(key, course)
            if avoid:
                return QC.reset_avoid(key, pm, course)
            if circle:
                return QC.reset_circle(key, pm, course)
            if landonly:
                # start where tube#2 spits it out, with realistic exit speed, target = LAND
                return QC.reset_at(key, pm, course, (11.0, 3.5, 0.55), (1.3, 0.0, 0.0), 6)
            return QC.reset(key, pm, course)

        @jax.jit
        def _step(s, a, key):
            if circle:
                return QC.step_circle(key, s, a, pm, M_inv, course, k_prog)
            return QC.step(key, s, a, pm, M_inv, course, k_prog)

        @jax.jit
        def _obs(s, key):
            R = E._quat_to_R(s.q)
            proprio = jnp.concatenate([s.w, s.Om / pm.K_motor])  # [7]
            priv = QC.observe_priv(s, course)                    # [20]
            slots = CS.simulate_detections(s.p, R, course['det_cls'], course['det_corners'],
                                           f, H, key, dr)
            hm = CS.heatmap_from_slots(slots, H)                 # [C,H,W]
            vec = CS.vector_from_slots(slots)                    # [C*5]
            return hm, vec, proprio, priv

        if avoid:
            n_obs_static = int(course['n_obs'])
            @jax.jit
            def _sample_obs(key):
                return QC.sample_avoid_obs(key, course, n_obs_static)
            @jax.jit
            def _step_av(s, a, key, oxy, orr):
                return QC.step_avoid(key, s, a, pm, M_inv, course, oxy, orr, k_prog)
            @jax.jit
            def _obs_av(s, key, oxy, orr):
                R = E._quat_to_R(s.q)
                proprio = jnp.concatenate([s.w, s.Om / pm.K_motor])
                priv = QC.observe_priv(s, course)
                flow = QC.simulate_flow_field(s.p, R, s.v, oxy, orr, course['z_top'], fov_h=fov_h)
                return jnp.zeros((self.C, H, H)), flow, proprio, priv
            self._sample_obs, self._step_av, self._obs_av = _sample_obs, _step_av, _obs_av
            self._obs_xy = self._obs_r = None

        if steer:
            n_obs_static = int(course['n_obs'])
            z_top = float(course['z_top'])
            dt_evt = float(course['dt_evt'])
            is_forest = self.forest
            nuisance_dr = self.nuisance_dr
            speedvar = self.speedvar
            magvar = self.magvar
            zone4 = self.zone4
            ratevar = self.ratevar
            ev_h = int(self.event_h)                 # event-frame resolution ('h<N>' suffix)
            ev_f = QC.steer_focal(ev_h)              # focal scales with resolution
            ev_gain = QC.steer_gain(ev_h, dt_evt)    # calibrated per (resolution, event window)
            _genx = self.genx                        # single-channel GenX320 DIFF3D obs ('gx' suffix)
            tiltdr = 'tilt' in task                  # camera-tilt DR ('tilt' token, round 8.1)
            @jax.jit
            def _sample_st(key):
                # -> xy, r, pts, valid, wall, drp   (drp = per-tick event-noise params [3])
                if zone4:                            # round 4: equal widths, count-based density
                    return QC.sample_zoneforest_obs(key, course, n_obs_static)
                if speedvar or magvar or ratevar:    # round 3: width-scaled zones
                    return QC.sample_speedforest_obs(key, course, n_obs_static)
                if nuisance_dr:
                    return (QC.sample_forest_obs_dr if is_forest else QC.sample_steer_obs_dr)(
                        key, course, n_obs_static)
                if is_forest:
                    return QC.sample_forest_obs(key, course, n_obs_static) + (QC.INERT_DR,)
                xy, r = QC.sample_steer_obs(key, course, n_obs_static)
                pts = QC.steer_edge_points(xy, r, z_top)
                return xy, r, pts, jnp.ones(pts.shape[0], bool), QC.INERT_WALL, QC.INERT_DR
            _scenedr = self.scenedr
            _sqroom = self.sqroom
            if _genx:
                # gx scene enrichment (round gx-2): OBSERVATION-only ground/clutter/wall points.
                # Appended to pts/valid, never to obs_xy/obs_r, so dynamics and collision are
                # untouched -- see QC.gx_scene_points. With the 'sc' token the enrichment is the
                # SCENE-TYPE LOTTERY instead (corridor/open/half-open/messy) and the per-episode
                # arena half-width rides as wall[6] (read by _reward_steer).
                _sample_base_gx = _sample_st
                @jax.jit
                def _sample_st(key):
                    # fold_in, not split: the base scene draw must see the ORIGINAL key so a gx
                    # task and its 2-channel twin generate the SAME tree layout for a given seed.
                    if _sqroom:
                        return QC.sample_squareroom_obs(key, course, n_obs_static)
                    if _scenedr:
                        return QC.sample_zoneforest_obs_sc(key, course, n_obs_static)
                    xy, r, pts, valid, wall, drp = _sample_base_gx(key)
                    gpts, gvalid = QC.gx_scene_points(jax.random.fold_in(key, 91), course)
                    return (xy, r, jnp.concatenate([pts, gpts], 0),
                            jnp.concatenate([valid, gvalid], 0), wall, drp)
            if tiltdr:
                _sample_base = _sample_st
                @jax.jit
                def _sample_st(key):
                    # append per-episode (pitch, roll) -> drp[3:5]; frames of non-tilt
                    # tasks are untouched (they never reach this wrapper).
                    k_s, k_t = jax.random.split(key)
                    out = _sample_base(k_s)
                    return out[:-1] + (jnp.concatenate([out[-1], QC.sample_tilt_dr(k_t)]),)
            @jax.jit
            def _step_st(s, a, key, oxy, orr, wall):
                if speedvar:
                    return QC.step_speedforest(key, s, a, course, oxy, orr, wall)
                if magvar:                           # |a| -> speed, heading = atan2(a1,a0)
                    return QC.step_magforest(key, s, a, course, oxy, orr, wall)
                if ratevar:                          # a0 = yaw rate, a1 = speed (no atan2)
                    return QC.step_ratezone(key, s, a, course, oxy, orr, wall)
                return QC.step_steer(key, s, a, course, oxy, orr, wall)
            @jax.jit
            def _obs_st(s, key, oxy, orr, pts, valid, drp):
                if _scenedr:                      # solid trunk occlusion (audit fix)
                    h_trees = drp[4:4 + n_obs_static]
                    valid = valid & QC.cylinder_occlusion_mask(
                        s.p, pts, oxy, jnp.abs(orr), h_trees)
                    # CLEAR-SPACE FRAME-CORRUPTION DR (gx-6b): when no obstacle
                    # is near, sometimes corrupt the frame (ghost echo + noise
                    # burst + gain wobble) so "unfamiliar frame + goal ahead"
                    # teaches HOLD COURSE, not turn-to-investigate — the
                    # measured circling mechanism on Isaac/real empty stretches.
                    key, kc = jax.random.split(key)
                # gx ground texture needs its own key. Split ONLY on the gx branch: splitting
                # unconditionally shifts the render key for every existing task and silently
                # changes their frames (caught by the frame-checksum test, twice now).
                if _genx:
                    key, gkey = jax.random.split(key)
                R = E._quat_to_R(s.q)
                if tiltdr:                           # body-frame camera tilt, drp[3:5]
                    R = R @ QC.tilt_R(drp[3], drp[4])
                # BINNED PROXY: below the native render size the frame is produced by rendering
                # at STEER_H and SUM-POOLING, so training frames share the Isaac/hardware binning
                # microstructure (fat blobs) instead of thin native streaks. At ev_h == STEER_H
                # this is a plain render, bit-identical to before.
                # (frame corruption applied below, after render, for scenedr)
                # `gx` deposits EVENT COUNTS with a contrast-triggered (saturating) response and
                # scales the nuisance DR into the same units -- see the GENX_* block in
                # quad_course_env.py. Everything else about the frame is unchanged.
                frame = QC.render_event_frame(
                    s.p, R, s.v, s.w, pts, valid, ev_h,
                    dt=dt_evt, key=key,
                    gain=((QC.genx_evt(ev_h) * (drp[3] if (_scenedr or _sqroom) else 1.0))
                          if _genx else ev_gain),
                    horizon_mag=drp[0] * (QC.GENX_NOISE_COUNTS if _genx else 1.0),
                    salt_p=drp[1],
                    salt_val=drp[2] * (QC.GENX_NOISE_COUNTS if _genx else 1.0),
                    channels=(1 if _genx else 2),   # single-channel GenX320 net for 'gx' tasks
                    pol_hash=_genx,                 # per-edge contrast polarity (gx only)
                    pol_grain=((drp[5] if _sqroom else QC.GX_POL_GRAIN) if _genx else None),
                    dilate_w=((drp[6] if _sqroom else QC.GX_DILATE_W) if _genx else None),
                    sat_knee=((QC.GENX_KNEE * (drp[4] if _sqroom else 1.0)) if _genx else None),
                    ground=(QC.gx_ground_dr(gkey, ev_h) if _genx else None))
                if _scenedr:
                    dd = jnp.min(jnp.linalg.norm(s.p[:2][None, :] - oxy, axis=-1)
                                 - jnp.abs(orr))
                    k1, k2, k3, k4 = jax.random.split(kc, 4)
                    p_cor = jnp.where(dd > 3.0, 0.30, 0.05)
                    do = jax.random.uniform(k1, ()) < p_cor
                    dyx = jax.random.randint(k2, (2,), -5, 6)
                    ghost = jnp.roll(frame, (dyx[0], dyx[1]), axis=(0, 1)) \
                        * jax.random.uniform(k3, (), minval=-0.7, maxval=0.7)
                    wob = jax.random.uniform(k4, (), minval=0.8, maxval=1.25)
                    frame = jnp.where(do, (frame + ghost) * wob, frame)
                # [3] goal bearing + yaw rate; speedforest appends normalised speed -> [4]
                proprio = (QC.speed_proprio if (speedvar or ratevar) else QC.steer_proprio)(s, course)
                priv = QC.observe_priv_steer(s, course, oxy, orr)  # [15]
                return frame, proprio, priv
            self._sample_st, self._step_st, self._obs_st = _sample_st, _step_st, _obs_st
            self._obs_xy = self._obs_r = self._pts = self._valid = self._wall = self._drp = None

        self._reset_fn = _reset
        self._step_fn = _step
        self._obs_fn = _obs
        self._state = None
        self._steps = 0
        self._done = True
        self._af = None

    @property
    def obs_space(self):
        if self.steer:
            return {
                'image': elements.Space(np.uint8, (self.H, self.H, 1 if self.genx else 2)),  # [H,W,1] genx / [H,W,2] event
                # goal bearing (cos,sin) + yaw rate; speedforest appends normalised speed
                'vector': elements.Space(np.float32, (4 if (self.speedvar or self.ratevar) else 3,)),
                'priv': elements.Space(np.float32, (15,)),                # informed decoder target
                'log/gate': elements.Space(np.float32),
                'log/target': elements.Space(np.float32),
                'log/landed': elements.Space(np.float32),
                'reward': elements.Space(np.float32),
                'is_first': elements.Space(bool),
                'is_last': elements.Space(bool),
                'is_terminal': elements.Space(bool),
            }
        vdim = (self.flow_dim + 7) if self.avoid else (self.C * 5 + 7)
        sp = {
            'image': elements.Space(np.uint8, (self.H, self.H, self.C)),
            'vector': elements.Space(np.float32, (vdim,)),
            # INFORMED DECODER (SkyDreamer's "Informed"): privileged state, TRAIN-ONLY.
            # agent.py puts this in dec_space but NOT enc_space, so the world model must
            # RECONSTRUCT true pos/vel/vector-to-goal from its latent (making the RSSM an
            # explicit state estimator) while the policy can never observe it.
            'priv': elements.Space(np.float32, (20,)),
            'log/gate': elements.Space(np.float32),      # 1.0 on an obstacle pass
            'log/target': elements.Space(np.float32),    # current obstacle index
            # DIRECT course-completion metric. Do NOT infer landings from episode/score:
            # the 6-pass and 7-pass return distributions OVERLAP (a 6-pass can score 232),
            # and the scale shifts every time the reward changes. Measure it, don't proxy it.
            'log/landed': elements.Space(np.float32),    # 1.0 on the final LAND pass
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        if self.obs_mode == 'vector':
            sp.pop('image')
        return sp

    @property
    def act_space(self):
        # steer/forest = 2D unit-circle heading; speedforest adds a 3rd = commanded speed; else CTBR
        adim = 3 if self.speedvar else 2 if self.steer else 4
        return {
            'reset': elements.Space(bool),
            'action': elements.Space(np.float32, (adim,), -1.0, 1.0),
        }

    def _build_obs(self, reward, is_first, is_last, is_terminal, passed=0.0, landed=0.0):
        with jax.transfer_guard('allow'), jax.default_device(self._cpu):
            self._key, k = jax.random.split(self._key)
            if self.steer:
                frame, proprio, priv = self._obs_st(self._state, k, self._obs_xy, self._obs_r,
                                                    self._pts, self._valid, self._drp)
                if self.genx:            # GenX320 DIFF3D grayscale: USAT(net*contrast + brightness)
                    img = np.clip(np.asarray(frame) * QC.genx_contrast(self.H) + QC.GENX_BRIGHTNESS,
                                  0, 255).astype(np.uint8)                        # [H,W,1] net frame
                else:
                    img = np.clip(np.asarray(frame) * 32.0, 0, 255).astype(np.uint8)  # [H,W,2] event frame
                out = {
                    'image': img,
                    'vector': np.asarray(proprio, np.float32),
                    'priv': np.asarray(priv, np.float32),
                    'log/gate': np.float32(passed),
                    'log/target': np.float32(float(np.asarray(self._state.target))),
                    'log/landed': np.float32(landed),
                    'reward': np.float32(reward),
                    'is_first': is_first, 'is_last': is_last, 'is_terminal': is_terminal,
                }
                return out
            if self.avoid:
                hm, vec, proprio, priv = self._obs_av(self._state, k, self._obs_xy, self._obs_r)
            else:
                hm, vec, proprio, priv = self._obs_fn(self._state, k)
            img = (np.asarray(hm).transpose(1, 2, 0) * 255).astype(np.uint8)  # [H,W,C]
            vector = np.concatenate([np.asarray(vec), np.asarray(proprio)]).astype(np.float32)
            target = float(np.asarray(self._state.target))
        out = {
            'image': img,
            'vector': vector,
            'priv': np.asarray(priv, np.float32),
            'log/gate': np.float32(passed),
            'log/target': np.float32(target),
            'log/landed': np.float32(landed),
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
        if self.obs_mode == 'vector':
            out.pop('image')          # vector-only: the heatmap carries no extra information
        return out

    def step(self, action):
        with jax.transfer_guard('allow'), jax.default_device(self._cpu):
            if action['reset'] or self._done:
                # NOTE the key draws below are in the SAME order as before (reset key first, then
                # the obstacle key). Only the reset CALL is deferred, so that the round-7 reset can
                # place a forced start inside whichever zone actually got the trees. Reordering the
                # draws would shift every existing task's RNG stream.
                self._key, k = jax.random.split(self._key)
                if self.avoid:
                    self._key, ko = jax.random.split(self._key)
                    self._obs_xy, self._obs_r = self._sample_obs(ko)   # NEW random obstacles each episode
                if self.steer:
                    self._key, ko = jax.random.split(self._key)
                    (self._obs_xy, self._obs_r, self._pts,
                     self._valid, self._wall, self._drp) = self._sample_st(ko)
                self._state = (self._reset_fn(k, self._obs_xy) if self.yawdr
                               else self._reset_fn(k))
                if self.scenedr:                 # per-episode flight height (wall[8])
                    z_ep = self._wall[8]
                    self._state = self._state._replace(
                        p=self._state.p.at[2].set(z_ep))
                self._steps = 0
                self._done = False
                self._af = np.zeros(4, np.float32)      # reset the filtered-action memory
                return self._build_obs(0.0, True, False, False)
            self._key, k = jax.random.split(self._key)
            raw = np.asarray(action['action'], np.float32)
            if self.action_lp > 0.0:                    # first-order low-pass on the ACTION
                self._af = self.action_lp * raw + (1.0 - self.action_lp) * self._af
                raw = self._af
            a = jnp.asarray(raw, jnp.float32)
            # ACTION REPEAT: hold `a` for act_repeat kinematic ticks and SUM the rewards, so the
            # policy runs at the deploy rate while the kinematics stay at 100 Hz (mirrors the real
            # CF: fast inner loops, slow policy). Summing keeps returns on the same scale as
            # act_repeat=1 -- the progress term telescopes, so a held action earns exactly the
            # progress it actually covers. Breaks out early on termination.
            r_sum, done, info = 0.0, False, None
            for _ in range(self.act_repeat):
                self._key, k = jax.random.split(self._key)
                if self.steer:
                    s2, r, done, info = self._step_st(
                        self._state, a, k, self._obs_xy, self._obs_r, self._wall)
                elif self.avoid:
                    s2, r, done, info = self._step_av(self._state, a, k, self._obs_xy, self._obs_r)
                else:
                    s2, r, done, info = self._step_fn(self._state, a, k)
                self._state = s2
                r_sum += float(r)
                if bool(done):
                    break
            r = r_sum
            self._steps += 1
            crash_or_finish = bool(done)
            timeout = self._steps >= self.max_steps
            is_last = crash_or_finish or timeout
            is_terminal = crash_or_finish and not timeout
            self._done = is_last
            passed = float(np.asarray(info['passed']))
            landed = float(np.asarray(info['all_done']))
        return self._build_obs(float(r), False, is_last, is_terminal, passed=passed, landed=landed)
