"""Fast ANALYTIC EVENT PROXY (JAX, jittable) — the deployable event-camera observation.

Real events fire on MOVING brightness edges. Given the camera pose (R, cam_pos), its linear
velocity v_world and body angular velocity omega, and a set of static 3D edge points, we compute
WITHOUT rendering, per point:
  1. transform world edge point -> camera frame (camera looks along body +x),
  2. its image velocity from ego-motion (point is static in the world),
  3. deposit a short ON/OFF streak (leading half -> ch0, trailing half -> ch1),
     magnitude ~ image speed (so nearer / faster edges = more events = looming).
Arithmetic over a few hundred points -> microseconds, RL-loop fast. Class-agnostic: works on ANY
3D edge (tree trunks, branches, poles, window rims, bars), which is why it generalises to a forest.

Camera convention MATCHES simulate_flow_field: camera +x = forward (into scene), +y horizontal,
+z up; R is body->world so world->camera is R^T. Output frame is [H,W,2] float (event counts).
"""
import numpy as np
import jax
import jax.numpy as jnp


def focal_from_fov(fov_deg, H):
    return float((H / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0))


# ------------------------- edge-point builders (world frame) -------------------------
def cylinder_edge_points(xy, r, h, nv=8, naz=3):
    """Vertical cylinders (tree trunks / poles) -> surface points.
    xy:[N,2] centres, r:[N] radius, h:[N] height. Returns [N*naz*nv, 3] world points.
    naz azimuths around the trunk (gives it width in the image) x nv heights."""
    az = jnp.linspace(0.0, 2 * jnp.pi, naz, endpoint=False)   # [naz]
    zf = jnp.linspace(0.05, 1.0, nv)                          # height fractions [nv]

    def one(c_xy, c_r, c_h):
        ax = c_xy[0] + c_r * jnp.cos(az)                      # [naz]
        ay = c_xy[1] + c_r * jnp.sin(az)
        X = jnp.repeat(ax, nv)                                # a0*nv, a1*nv, ...
        Y = jnp.repeat(ay, nv)
        Z = jnp.tile(zf * c_h, naz)
        return jnp.stack([X, Y, Z], -1)                       # [naz*nv, 3]

    return jax.vmap(one)(xy, r, h).reshape(-1, 3)             # [N*naz*nv, 3]


def pillar_edge_points(xy, r, h, shape, lean_vec, nv=16, naz=8, with_normals=False):
    """Shape-lottery pillars (2026-08-23): per-obstacle edge points, FIXED count
    nv*naz per pillar (matches cylinder_edge_points(nv,naz) shapes).
    shape [N] int: 0=cylinder, 1=square pillar, 2=rectangular pillar (2:1),
    3=thin tube (r/3). lean_vec [N,2] = tan(lean)*direction, horizontal offset
    per metre of height; 0 = upright. Boxes concentrate points on their 4
    vertical corner edges — what an event camera sees of a flat column."""
    az = jnp.linspace(0.0, 2 * jnp.pi, naz, endpoint=False)
    zf = jnp.linspace(0.05, 1.0, nv)

    def one(c_xy, c_r, c_h, c_shape, c_lean):
        cx = jnp.cos(az) * c_r
        cy = jnp.sin(az) * c_r
        corner = (jnp.floor(az / (jnp.pi / 2)) * (jnp.pi / 2) + jnp.pi / 4)
        sqx = jnp.cos(corner) * c_r * 1.4142
        sqy = jnp.sin(corner) * c_r * 1.4142
        rex, rey = sqx * 1.4, sqy * 0.7
        tbx, tby = cx / 3.0, cy / 3.0
        px = jnp.select([c_shape == 0, c_shape == 1, c_shape == 2, c_shape == 3],
                        [cx, sqx, rex, tbx])
        py = jnp.select([c_shape == 0, c_shape == 1, c_shape == 2, c_shape == 3],
                        [cy, sqy, rey, tby])
        X = jnp.repeat(c_xy[0] + px, nv)
        Y = jnp.repeat(c_xy[1] + py, nv)
        Z = jnp.tile(zf * c_h, naz)
        X = X + c_lean[0] * Z
        Y = Y + c_lean[1] * Z
        P = jnp.stack([X, Y, Z], -1)
        # outward radial normal (horizontal; lean tilt <=15 deg neglected). For boxes this is the
        # corner direction -- the outward side of the vertical edge an event camera actually sees.
        nn = jnp.hypot(px, py) + 1e-6
        NX = jnp.repeat(px / nn, nv)
        NY = jnp.repeat(py / nn, nv)
        N = jnp.stack([NX, NY, jnp.zeros_like(NX)], -1)
        return P, N

    P, N = jax.vmap(one)(xy, r, h, shape, lean_vec)
    P, N = P.reshape(-1, 3), N.reshape(-1, 3)
    return (P, N) if with_normals else P


def hashed_plane_normals(pts, plane, grain=0.8):
    """Pseudo-random but world-anchored outward normals for surface MARKINGS (posters, ground
    patches, clutter): a marking's edge points in a random direction, but the same 0.8 m patch
    keeps the same direction forever. plane='ground' -> in the x-y plane; 'wallx' -> in the x-z
    plane (corridor walls run along x at fixed y)."""
    cell = jnp.floor(pts[:, :3] / grain)
    th = jnp.mod(jnp.sin(cell @ jnp.array([45.233, 91.187, 27.719], jnp.float32))
                 * 15731.743, 1.0) * (2 * jnp.pi)
    if plane == 'wallx':
        return jnp.stack([jnp.cos(th), jnp.zeros_like(th), jnp.sin(th)], -1)
    return jnp.stack([jnp.cos(th), jnp.sin(th), jnp.zeros_like(th)], -1)


def segment_edge_points(p0, p1, n=8):
    """Thin 3D segments (branches) -> sampled points. p0,p1:[M,3]. Returns [M*n,3]."""
    t = jnp.linspace(0.0, 1.0, n)[None, :, None]             # [1,n,1]
    pts = p0[:, None, :] * (1 - t) + p1[:, None, :] * t      # [M,n,3]
    return pts.reshape(-1, 3)


def shadow_edge_points(xy, r, h, sun_az, sun_elev, n=6, max_len=4.0):
    """Ground-shadow streaks cast by vertical trunks — the #1 real-event source
    MISSING from this proxy (measured 2026-08-06: Isaac closed-loop eval of
    steer4 went 0% -> 23% just by removing scene shadows; see
    runs/isaac_eval/ + robosim CLAUDE.md).

    A trunk at ``xy`` with height ``h`` under a sun at azimuth ``sun_az``
    (direction the light TRAVELS, i.e. shadow extends along +az) and elevation
    ``sun_elev`` casts a ground line of length ``min(h/tan(elev), max_len)``.
    Returns [N*n, 3] points on the ground plane (z=0.02). Sample sun_az/elev
    per episode; gate presence with the caller's ``valid`` mask (overcast =
    all False) so jit shapes stay static.
    """
    L = jnp.clip(h / jnp.tan(jnp.clip(sun_elev, 0.2, 1.4)), 0.2, max_len)  # [N]
    t = jnp.linspace(0.05, 1.0, n)                                        # [n]
    dx = jnp.cos(sun_az); dy = jnp.sin(sun_az)
    X = xy[:, 0:1] + (L[:, None] * t[None, :]) * dx                       # [N,n]
    Y = xy[:, 1:2] + (L[:, None] * t[None, :]) * dy
    Z = jnp.full(X.shape, 0.02)
    return jnp.stack([X, Y, Z], -1).reshape(-1, 3)


def density_mask(key, n, keep_frac):
    """Random keep-mask for edge-point *density* DR (jit-safe: shape stays [n],
    density varies via the mask). Use on walls so the policy sees anything from
    the sparse training grid up to dense texture-like coverage — the Isaac eval
    showed dense unexpected wall mass reads as an obstacle to a sparse-trained
    policy."""
    return jax.random.uniform(key, (n,)) < keep_frac


def wall_edge_points(y, x0, x1, z_top, nx=30, nz=3):
    """A vertical wall plane at fixed y (corridor boundary) -> textured grid of edge points.
    Under forward motion these stream backward in the periphery (hallway flow) -> the policy SEES
    the wall. Returns [nx*nz, 3]."""
    xs = jnp.linspace(x0, x1, nx)
    zs = jnp.linspace(0.2, z_top, nz)
    X = jnp.repeat(xs, nz)
    Z = jnp.tile(zs, nx)
    Y = jnp.full(X.shape, y)
    return jnp.stack([X, Y, Z], -1)                          # [nx*nz, 3]


# ------------------------------- the per-tick renderer -------------------------------
def simulate_event_frame(cam_pos, R, v_world, omega_body, pts, valid, f, H,
                         dt=0.05, key=None, dr=True, mag_clip=16.0, gain=3.0,
                         occlude=True, occ_tol=0.2,
                         horizon_mag=0.0, salt_p=0.0012, salt_val=1.0, channels=2,
                         pol_hash=False, sat_knee=None, ground=None,
                         pol_grain=None):
    """cam_pos:[3], R:[3,3] body->world, v_world:[3], omega_body:[3] (rad/s),
    pts:[M,3] world edge points, valid:[M] bool (padding mask). f focal px, H frame size.
    occlude: z-buffer depth test so only the NEAREST surface per pixel fires (a solid wall hides
    what's behind it; background shows only through gaps).

    Real-event nuisance DR (2026-08-06, from the Isaac closed-loop eval; defaults keep the
    ORIGINAL behavior so old checkpoints eval unchanged):
      horizon_mag > 0 : sparse event band at the horizon row (level flight: image centre) —
        the ground/background boundary flickers under motion in the real stream.
      salt_p / salt_val : background count-noise rate and amplitude (RTX-denoiser-like floor;
        randomize per episode, e.g. salt_p in [0.001, 0.006], salt_val in {1, 2}).
    Returns [H,W,2] float32 event frame."""
    C = (H - 1) / 2.0
    # pts may carry per-point outward NORMALS in columns 3:6 (motion-consistent polarity,
    # 2026-08-24); plain [M,3] inputs keep the fully-hashed polarity (legacy tasks bit-identical).
    nrm = pts[:, 3:6] if pts.shape[-1] >= 6 else None
    pts = pts[:, :3]
    Pc = (pts - cam_pos[None]) @ R              # world->cam: (R^T (x-o))^T == (x-o) @ R
    v_cam = R.T @ v_world                       # [3]
    # velocity of a WORLD-STATIC point seen in the moving camera frame:
    #   Pdot = -v_cam - omega x Pc
    Pdot = -v_cam[None] - jnp.cross(jnp.broadcast_to(omega_body, Pc.shape), Pc)
    Pc_end = Pc + Pdot * dt

    def proj(P):
        x = jnp.maximum(P[:, 0], 1e-3)
        u = C - f * (P[:, 1] / x)              # +y = drone-LEFT -> LEFT of image (was mirrored)
        v = C - f * (P[:, 2] / x)              # +z up -> smaller row (top of image)
        return u, v, P[:, 0] > 0.05            # in front of camera

    u0, v0, if0 = proj(Pc)
    u1, v1, if1 = proj(Pc_end)
    speed = jnp.hypot(u1 - u0, v1 - v0)
    vis = valid & if0 & if1
    if occlude:                                # depth test: keep only the front surface per pixel
        depth = Pc[:, 0]                        # camera-forward distance
        ui0 = jnp.clip(jnp.round(u0), 0, H - 1).astype(jnp.int32)
        vi0 = jnp.clip(jnp.round(v0), 0, H - 1).astype(jnp.int32)
        inb0 = (u0 >= 0) & (u0 < H) & (v0 >= 0) & (v0 < H)
        dsrc = jnp.where(vis & inb0, depth, 1e6)
        dbuf = jnp.full((H, H), 1e6).at[vi0, ui0].min(dsrc)   # nearest depth per pixel
        vis = vis & (depth <= dbuf[vi0, ui0] + occ_tol)       # this point is the front surface
    if sat_knee is None:
        base = jnp.clip(speed * gain, 0.0, mag_clip) * vis.astype(jnp.float32)
    else:
        # CONTRAST-TRIGGERED RESPONSE (`gx` / GenX320 net path, 2026-08-18).
        # A DIFF3D pixel fires when the log-intensity STEP crosses its threshold, so an edge that
        # sweeps a pixel emits ~one event whether it crossed at 1 px/window or 20; magnitude is set
        # by contrast, not by how far the edge travelled. The default response above is linear in
        # displacement, which is a fair proxy at a 45-100 ms window but collapses at the sensor's
        # real 20 ms one: measured against rec1.bin, a linear 20 ms frame is 87% background while
        # the recording is 39%. Here `gain` is EVENT COUNTS deposited per edge point and `sat_knee`
        # the displacement (render px) above which the pixel reliably fires.
        base = jnp.clip(speed / sat_knee, 0.0, 1.0) * gain * vis.astype(jnp.float32)

    # PER-EDGE CONTRAST POLARITY (`gx` / GenX320 net path, 2026-08-18).
    # A real DIFF3D pixel signs its events by the direction of the BRIGHTNESS step, so whether the
    # leading side of a moving edge reads ON or OFF depends on that edge's albedo contrast -- half
    # the edges in a scene are dark-on-light and produce the opposite doublet. The proxy has only
    # geometry, so without this every sim edge carries the same leading-is-positive convention and
    # the net image gains a global sign cue that the hardware does not have (visible in rec1.bin:
    # neighbouring edges of one scene appear with opposite polarity). The sign is hashed from the
    # point's WORLD position so it is fixed for the lifetime of an obstacle -- an edge does not
    # change contrast sign between frames -- and costs no extra state or key draws.
    if pol_hash:
        if pol_grain is None:
            hp = pts
        else:
            _pg = jnp.maximum(pol_grain, 1e-6)
            # grain ~0 = LEGACY per-point polarity (pre-2026-08-24 regime; used by render-DR)
            hp = jnp.where(pol_grain > 0.01, jnp.floor(pts / _pg) * _pg, pts)
        h = jnp.sin(hp @ jnp.array([12.9898, 78.233, 37.719], jnp.float32)) * 43758.5453
        pos = (jnp.mod(h, 1.0) < 0.5).astype(jnp.float32)     # [M] 1 = normal, 0 = flipped
        if nrm is not None:
            # MOTION-CONSISTENT POLARITY (2026-08-24): a real pixel fires OFF when the object
            # COVERS it (edge advancing along its outward normal) and ON when it REVEALS it --
            # so polarity = (object contrast class, hashed) x (covering vs revealing, physics).
            # Forward flight -> a whole silhouette expands -> one polarity around the object;
            # yaw -> leading edges one sign, trailing the other, flipping with yaw direction.
            n_cam = nrm @ R
            un, vn, _ = proj(Pc + 0.05 * n_cam)
            dot = (u1 - u0) * (un - u0) + (v1 - v0) * (vn - v0)
            cover = dot > 0.0
            # FORESHORTENING (2026-08-24, caught by the vertical-split review): an edge whose
            # normal points along the VIEW RAY has no visible contrast step -- a trunk's front
            # face is uniform (only the silhouette fires) and an along-corridor marking edge is
            # seen edge-on. Without this, those degenerate points get their sign from tiny
            # perspective-curvature terms, which correlates with image height and painted a fake
            # top/bottom polarity split. Weight = sin(angle(normal, ray)).
            ray = Pc / (jnp.linalg.norm(Pc, axis=-1, keepdims=True) + 1e-6)
            nn = n_cam / (jnp.linalg.norm(n_cam, axis=-1, keepdims=True) + 1e-6)
            fore = jnp.sqrt(jnp.clip(1.0 - jnp.sum(ray * nn, -1) ** 2, 0.0, 1.0))
            base = base * fore
            h2 = jnp.mod(jnp.sin(hp[:, :3] @ jnp.array([93.9898, 47.233, 61.719], jnp.float32))
                         * 24634.6345, 1.0)
            flip = h2 < 0.05                    # per-cell threshold-mismatch noise
            pos = jnp.where(cover != flip, pos, 1.0 - pos)
    else:
        pos = None

    frame = jnp.zeros((H, H, 2), jnp.float32)
    for a in (0.0, 0.34, 0.67, 1.0):           # short streak so fast edges smear
        uu = u0 * (1 - a) + u1 * a
        vv = v0 * (1 - a) + v1 * a
        ch = 0 if a > 0.5 else 1               # leading half -> ch0, trailing -> ch1
        inb = (uu >= 0) & (uu < H) & (vv >= 0) & (vv < H)
        m = base * inb.astype(jnp.float32)
        ui = jnp.clip(jnp.round(uu), 0, H - 1).astype(jnp.int32)
        vi = jnp.clip(jnp.round(vv), 0, H - 1).astype(jnp.int32)
        if pos is None:
            frame = frame.at[vi, ui, ch].add(m)
        else:
            # SINGLE-POLARITY STREAK (`gx`, corrected 2026-08-19). A moving step edge does NOT
            # produce an ON/OFF doublet: the pixels the edge sweeps all see the brightness step in
            # the SAME direction, so they all fire the same polarity, and the opposite polarity
            # comes from the object's OTHER edge (separated by the object's width, not by the
            # displacement). Depositing leading-vs-trailing into opposite channels made the net a
            # +/- pair whose separation IS the displacement -- under 20 ms motion that pair lands
            # inside one pooled cell and CANCELS, which is why the enriched scene could not raise
            # the frame's occupancy past ~77% no matter how many ground points or how low the
            # firing knee. Here the whole streak carries the point's own contrast sign.
            frame = frame.at[vi, ui, 0].add(m * pos)
            frame = frame.at[vi, ui, 1].add(m * (1.0 - pos))

    if dr and key is not None:                 # sim2real: LIGHT sensor background noise + a hot pixel
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        frame = frame + salt_val * (jax.random.uniform(k1, (H, H, 2)) < salt_p).astype(jnp.float32)
        hy = jax.random.randint(k2, (2,), 0, H)
        hx = jax.random.randint(k3, (2,), 0, H)
        frame = frame.at[hy, hx, 0].add(3.0)
        if isinstance(horizon_mag, jnp.ndarray) or horizon_mag > 0.0:
            # sparse flicker band where flat ground meets the background: at
            # level flight that is the image-centre row (+/- 1 px jitter)
            hrow = jnp.clip(
                jnp.round(C + jax.random.uniform(k4, (), minval=-1.5, maxval=1.5)),
                0, H - 1).astype(jnp.int32)
            cols = jax.random.uniform(k5, (H, 2))
            band = jnp.clip(horizon_mag, 0.0, None) * (cols < 0.35).astype(jnp.float32) \
                * (0.5 + cols)                          # varied magnitudes
            frame = frame.at[hrow, :, :].add(band)
    if ground is not None:
        # ---- ANALYTIC GROUND-TEXTURE EVENTS (`gx`, round gx-2, 2026-08-19) --------------------
        # The ground cannot be represented by sampled points: the z-buffer keeps one point per
        # pixel, so coverage saturates at the pixels the ground subtends and extra points merely
        # compete for them (measured: 1500 -> 8000 ground points moved frame occupancy by <2 pt).
        # A real sensor sees a CONTINUOUS textured surface -- every pixel below the horizon looks
        # at ground and fires as the projection sweeps across the texture. So shade it per pixel
        # instead: intersect each pixel's ray with z=0, take the flow of that world point, and let
        # a world-ANCHORED hash decide whether the patch is textured and with which contrast sign
        # (anchored so the texture does not flicker between frames as a per-pixel random would).
        evt, knee, grain, fill, zmax = (
            ground['evt'], ground['knee'], ground['grain'], ground['fill'], ground['zmax'])
        gy, gx_ = jnp.meshgrid(jnp.arange(H), jnp.arange(H), indexing='ij')
        dir_cam = jnp.stack([jnp.ones_like(gx_, jnp.float32),
                             (C - gx_) / f, (C - gy) / f], -1)          # [H,H,3]
        dir_world = dir_cam @ R.T                                        # body->world
        denom = jnp.where(jnp.abs(dir_world[..., 2]) < 1e-6, -1e-6, dir_world[..., 2])
        t = -cam_pos[2] / denom                                          # ray param to z = 0
        below = (t > 0.05) & (t < zmax) & (denom < 0)                    # below the horizon only
        W = cam_pos[None, None, :] + t[..., None] * dir_world            # world hit point
        P = (W - cam_pos[None, None, :]) @ R                             # camera frame
        Pd = -v_cam[None, None, :] - jnp.cross(
            jnp.broadcast_to(omega_body, P.shape), P)
        P1 = P + Pd * dt
        x0c = jnp.maximum(P[..., 0], 1e-3)
        x1c = jnp.maximum(P1[..., 0], 1e-3)
        du = (C - f * (P1[..., 1] / x1c)) - (C - f * (P[..., 1] / x0c))
        dv = (C - f * (P1[..., 2] / x1c)) - (C - f * (P[..., 2] / x0c))
        flow = jnp.hypot(du, dv)
        # world-anchored texture: quantise the hit point to a grain, hash it for presence + sign
        cell = jnp.floor(W[..., :2] / grain)
        hh = jnp.sin(cell[..., 0] * 12.9898 + cell[..., 1] * 78.233) * 43758.5453
        tex = jnp.mod(hh, 1.0)
        present = below & (tex < fill)
        sign_pos = jnp.mod(tex * 7.13, 1.0) < 0.5
        mag = jnp.clip(flow / knee, 0.0, 1.0) * evt * present.astype(jnp.float32)
        if occlude:                          # a trunk in front hides the ground behind it
            mag = mag * (P[..., 0] <= dbuf + occ_tol).astype(jnp.float32)
        frame = frame.at[..., 0].add(mag * sign_pos.astype(jnp.float32))
        frame = frame.at[..., 1].add(mag * (1.0 - sign_pos.astype(jnp.float32)))

    if channels == 1:
        # GenX320 DIFF3D single-channel: the sensor holds a signed net accumulator per pixel over
        # the integration window. ch0 (leading half of the streak) plays the arriving-edge polarity
        # and ch1 (trailing) the departing one, so net = ch0 - ch1 is the same signed quantity --
        # with pol_hash on, per-edge contrast sign is applied at deposit time. The 128-centre map
        # USAT(net*contrast + 128) is applied downstream in the wrapper's _build_obs. Returns
        # [H,W,1]. NOTE the scale there is NOT the firmware's CONTRAST_DEFAULT=16: that constant
        # multiplies a NATIVE-resolution int8 net, while this frame is already a pooled aggregate
        # (see GENX_CONTRAST in quad_course_env.py).
        return frame[..., 0:1] - frame[..., 1:2]
    return frame


# --------------------------------- self-test / viz ---------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    H = 64
    f = focal_from_fov(90.0, H)
    R = jnp.eye(3)                              # facing +x (world forward)
    omega = jnp.zeros(3)

    # one tree dead-ahead at increasing proximity -> looming (more/faster events)
    dists = [4.0, 3.0, 2.0, 1.2]
    v = jnp.array([1.6, 0.0, 0.0])             # flying forward at 1.6 m/s
    fig, ax = plt.subplots(2, len(dists), figsize=(3 * len(dists), 6))
    for j, d in enumerate(dists):
        pts = cylinder_edge_points(jnp.array([[d, 0.0]]), jnp.array([0.15]), jnp.array([2.0]),
                                   nv=12, naz=6)
        valid = jnp.ones(pts.shape[0], bool)
        fr = simulate_event_frame(jnp.array([0., 0., 1.0]), R, v, omega, pts, valid, f, H,
                                  key=jax.random.PRNGKey(j), dr=True)
        fr = np.asarray(fr)
        rgb = np.zeros((H, H, 3)); rgb[..., 2] = np.clip(fr[..., 0] / (fr.max() + 1e-6), 0, 1)
        rgb[..., 0] = np.clip(fr[..., 1] / (fr.max() + 1e-6), 0, 1)
        ax[0, j].imshow(rgb); ax[0, j].axis('off'); ax[0, j].set_title(f'tree @ {d:.1f} m')
        ax[1, j].imshow(fr.sum(-1), cmap='gray'); ax[1, j].axis('off')
        print(f'd={d:.1f}m  total events={fr.sum():.0f}  peak={fr.max():.1f}')
    fig.suptitle('JAX event proxy: approaching a tree -> more/faster events = looming')
    fig.tight_layout(); fig.savefig('event_proxy_demo/jax_proxy_looming.png', dpi=90)
    print('saved event_proxy_demo/jax_proxy_looming.png')

    # left vs right tree -> asymmetric frame (steering signal)
    fig2, ax2 = plt.subplots(1, 3, figsize=(11, 4))
    for k, (name, oy) in enumerate([('tree LEFT', 1.2), ('tree AHEAD', 0.0), ('tree RIGHT', -1.2)]):
        pts = cylinder_edge_points(jnp.array([[2.5, oy]]), jnp.array([0.15]), jnp.array([2.0]),
                                   nv=12, naz=6)
        valid = jnp.ones(pts.shape[0], bool)
        fr = np.asarray(simulate_event_frame(jnp.array([0., 0., 1.0]), R, v, omega, pts, valid, f, H,
                                             dr=False))
        ax2[k].imshow(fr.sum(-1), cmap='inferno'); ax2[k].axis('off'); ax2[k].set_title(name)
        L = fr[:, :H // 2].sum(); Rr = fr[:, H // 2:].sum()
        print(f'{name:11s}  left-half={L:.0f}  right-half={Rr:.0f}')
    fig2.suptitle('lateral position -> which side of the frame lights up (steering)')
    fig2.tight_layout(); fig2.savefig('event_proxy_demo/jax_proxy_lateral.png', dpi=90)
    print('saved event_proxy_demo/jax_proxy_lateral.png')
