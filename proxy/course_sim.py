"""Simulate detector observations for RL training (fast, geometry-only).

Instead of rendering RGB/event frames and running a real YOLO detector in the RL
loop (slow), we project the ground-truth 3D obstacles into the drone camera to
get synthetic (class + 2D bbox) detections directly, then build the two Dreamer
observation representations:

  * heatmap  [C, H, W]  — one channel per class; each detection painted as a soft
                          Gaussian blob (encodes type=channel, bearing=position,
                          distance=blob size). Generalizes SkyDreamer's mask.
  * vector   [C * 5]     — per-class slot [present, cx, cy, w, h] (nearest instance).
                          Exact geometry for the metric task; deploy = detector->slot.

Domain randomization (bbox jitter / dropout / false positives) makes the synthetic
detections look like a real detector's noisy output -> sim2real for perception.

C = 8 classes: window_blue, window_red, bar_red, bar_blue, tube_gate,
                marker_takeoff, marker_landing, person.
"""
from functools import partial
import jax
import jax.numpy as jnp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mask_render as MR   # reuse the pinhole _project
import quad_ctbr_env as E

N_CLASSES = 8
CLASS_NAMES = ["window_blue", "window_red", "bar_red", "bar_blue",
               "tube_gate", "marker_takeoff", "marker_landing", "person"]


def box_corners(center, half):
    """8 corners of an axis-aligned 3D box. center[3], half[3] -> [8,3]."""
    cx, cy, cz = center
    hx, hy, hz = half
    signs = jnp.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                      dtype=float)
    return jnp.array(center) + signs * jnp.array(half)


def _project_box(cam_pos, R, corners, f, H):
    """Project a 3D box (8 corners) to a 2D bbox. Returns (present, cx,cy,w,h)
    with pixel coords normalized to [0,1]. present=obstacle is in front & in view."""
    c = H / 2.0
    uv, depth = jax.vmap(lambda P: MR._project(P, cam_pos, R, f, c))(corners)  # [8,2],[8]
    in_front = depth > 0.1
    u, v = uv[:, 0], uv[:, 1]
    # only use corners that are in front; if none, mark absent
    big = 1e6
    u_f = jnp.where(in_front, u, big)
    v_f = jnp.where(in_front, v, big)
    u_b = jnp.where(in_front, u, -big)
    v_b = jnp.where(in_front, v, -big)
    umin, umax = u_f.min(), u_b.max()
    vmin, vmax = v_f.min(), v_b.max()
    # clip to image
    umin = jnp.clip(umin, 0, H); umax = jnp.clip(umax, 0, H)
    vmin = jnp.clip(vmin, 0, H); vmax = jnp.clip(vmax, 0, H)
    w = (umax - umin) / H
    h = (vmax - vmin) / H
    cx = 0.5 * (umin + umax) / H
    cy = 0.5 * (vmin + vmax) / H
    present = (in_front.sum() >= 4) & (w > 0.01) & (h > 0.01) & \
              (cx > 0.0) & (cx < 1.0) & (cy > 0.0) & (cy < 1.0)
    return present.astype(jnp.float32), cx, cy, w, h, depth.mean()


def simulate_detections(cam_pos, R, classes, corners_M, f, H, key=None, dr=None):
    """Project M obstacles -> per-class detection slots [C,5]=(present,cx,cy,w,h).
    classes[M] int, corners_M[M,8,3]. Optional DR noise via `dr` dict + key."""
    pres, cx, cy, w, h, depth = jax.vmap(
        lambda cor: _project_box(cam_pos, R, cor, f, H))(corners_M)   # each [M]

    if dr is not None and key is not None:
        k1, k2, k3 = jax.random.split(key, 3)
        j = dr.get("jitter", 0.0)
        cx = cx + j * jax.random.normal(k1, cx.shape)
        cy = cy + j * jax.random.normal(jax.random.fold_in(k1, 1), cy.shape)
        w = w * (1 + j * jax.random.normal(k2, w.shape))
        h = h * (1 + j * jax.random.normal(jax.random.fold_in(k2, 1), h.shape))
        drop = jax.random.uniform(k3, pres.shape) < dr.get("dropout", 0.0)
        pres = pres * (1 - drop.astype(jnp.float32))

    # aggregate to per-class slots: nearest present instance of each class
    def slot(c):
        m = (classes == c) & (pres > 0.5)
        d = jnp.where(m, depth, 1e6)
        i = jnp.argmin(d)
        any_c = m.any().astype(jnp.float32)
        return jnp.array([any_c, cx[i], cy[i], w[i], h[i]])
    slots = jax.vmap(slot)(jnp.arange(N_CLASSES))          # [C,5]
    return slots


@partial(jax.jit, static_argnums=(1,))
def heatmap_from_slots(slots, H=64):
    """slots [C,5] -> heatmap [C,H,W] of soft Gaussian blobs (one per class)."""
    ys = (jnp.arange(H) + 0.5) / H
    xs = (jnp.arange(H) + 0.5) / H
    gx, gy = jnp.meshgrid(xs, ys)   # [H,W]

    def chan(s):
        present, cx, cy, w, h = s
        sx = jnp.clip(w, 0.02, 1.0) / 2.0
        sy = jnp.clip(h, 0.02, 1.0) / 2.0
        g = jnp.exp(-(((gx - cx) ** 2) / (2 * sx ** 2 + 1e-6)
                      + ((gy - cy) ** 2) / (2 * sy ** 2 + 1e-6)))
        return present * g
    return jax.vmap(chan)(slots)     # [C,H,W]


def vector_from_slots(slots):
    """slots [C,5] -> flat [C*5] metric vector."""
    return slots.reshape(-1)


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = float(MR.focal_from_fov(90.0, 64))
    # a small course along +x: window_red, bar_blue(low), window_blue, bar_red(high), tube, marker
    obstacles = [
        (1, (1.5, 0.0, 1.4), (0.03, 0.25, 0.25)),   # window_red (opening)
        (3, (2.6, 0.0, 0.7), (0.03, 0.8, 0.05)),    # bar_blue  (fly under; low, wide)
        (0, (3.7, 0.3, 1.5), (0.03, 0.30, 0.30)),   # window_blue
        (2, (4.8, 0.0, 1.8), (0.03, 0.8, 0.05)),    # bar_red   (fly over; high, wide)
        (4, (6.0, -0.2, 1.2), (0.05, 0.15, 0.15)),  # tube_gate
        (6, (7.2, 0.0, 0.1), (0.2, 0.2, 0.01)),     # marker_landing (on floor)
    ]
    classes = jnp.array([o[0] for o in obstacles])
    corners_M = jnp.stack([box_corners(o[1], o[2]) for o in obstacles])

    cam_pos = jnp.array([0.0, 0.0, 1.3])
    R = jnp.eye(3)   # level, facing +x
    slots = simulate_detections(cam_pos, R, classes, corners_M, f, 64)
    hm = heatmap_from_slots(slots, 64)
    vec = vector_from_slots(slots)
    print("per-class slots [present,cx,cy,w,h]:")
    for c in range(N_CLASSES):
        p, x, y, w, h = np.array(slots[c])
        if p > 0.5:
            print(f"  {CLASS_NAMES[c]:16s}: cx={x:.2f} cy={y:.2f} w={w:.2f} h={h:.2f}")
    print("vector dim:", vec.shape, "heatmap:", hm.shape)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(np.array(hm[c]), cmap="magma", vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES[c], fontsize=8); ax.axis("off")
    fig.suptitle("Simulated detector obs — per-class heatmap channels (drone at x=0 facing +x)")
    out = os.path.join(os.path.dirname(__file__), "..", "runs", "m2", "course_obs_test.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
