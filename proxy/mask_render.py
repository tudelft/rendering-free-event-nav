"""Differentiable-friendly gate MASK renderer (JAX), for SkyDreamer-CF M2.

Given the drone pose and a gate, projects the gate opening into a drone-mounted
forward-facing pinhole camera and rasterizes a HxW binary mask — a stand-in for
the deploy-time perception (GenX320 event camera -> GateNet U-Net -> 64x64 mask).

Camera: mounted on the body, optical axis = body +x (forward), right = body -y,
down = body -z. Pinhole with square pixels, FOV set by focal length.

Kept jittable and vmap-friendly so masks render inside the batched sim.
Simplification (v1): if any gate corner is behind the image plane the mask is
empty (fine — by then the drone is essentially at the gate and crosses).
"""
from functools import partial
import jax
import jax.numpy as jnp


def gate_corners(center, normal, half):
    """4 world-space corners of the gate opening (CCW)."""
    center = jnp.asarray(center, float)
    n = jnp.asarray(normal, float)
    n = n / (jnp.linalg.norm(n) + 1e-9)
    up = jnp.array([0.0, 0.0, 1.0])
    u = jnp.cross(up, n)
    u = u / (jnp.linalg.norm(u) + 1e-9)          # lateral in-plane axis
    v = jnp.cross(n, u)                          # vertical in-plane axis
    hy, hz = half
    return jnp.stack([
        center - hy * u - hz * v,
        center + hy * u - hz * v,
        center + hy * u + hz * v,
        center - hy * u + hz * v,
    ])                                           # [4, 3]


def focal_from_fov(fov_deg, W):
    return (W / 2.0) / jnp.tan(jnp.deg2rad(fov_deg) / 2.0)


def _project(P, cam_pos, R, f, c):
    fwd = R @ jnp.array([1.0, 0.0, 0.0])         # body +x
    right = R @ jnp.array([0.0, -1.0, 0.0])      # body -y
    down = R @ jnp.array([0.0, 0.0, -1.0])       # body -z
    rel = P - cam_pos
    zc = rel @ fwd                               # depth
    xc = rel @ right
    yc = rel @ down
    zsafe = jnp.where(zc > 1e-3, zc, 1e-3)
    u = f * xc / zsafe + c
    v = f * yc / zsafe + c
    return jnp.array([u, v]), zc


@partial(jax.jit, static_argnums=(3,))
def render_mask(cam_pos, R, corners, H=64, f=32.0):
    """cam_pos [3], R [3,3] body->world, corners [4,3] -> mask [H,H] in {0,1}."""
    c = H / 2.0
    pts, zc = jax.vmap(lambda P: _project(P, cam_pos, R, f, c))(corners)  # [4,2],[4]
    valid = jnp.all(zc > 0.05)

    cols = jnp.arange(H) + 0.5                   # u (x) coords
    rows = jnp.arange(H) + 0.5                   # v (y) coords
    px = jnp.tile(cols[None, :], (H, 1)).reshape(-1)   # [H*W] u per pixel
    py = jnp.tile(rows[:, None], (1, H)).reshape(-1)   # [H*W] v per pixel

    def edge(i):
        a = pts[i]
        b = pts[(i + 1) % 4]
        return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])

    crosses = jnp.stack([edge(i) for i in range(4)], 0)          # [4, H*W]
    inside = jnp.all(crosses >= 0, 0) | jnp.all(crosses <= 0, 0)
    return (inside & valid).astype(jnp.float32).reshape(H, H)


@partial(jax.jit, static_argnums=(3,))
def render_masks(cam_pos, R, corners_G, H=64, f=32.0):
    """Union mask of G gates. corners_G [G,4,3] -> mask [H,H] in {0,1}."""
    masks = jax.vmap(lambda c: render_mask(cam_pos, R, c, H, f))(corners_G)  # [G,H,W]
    return masks.max(0)


def gate_corners_multi(centers, normals, halves):
    """centers [G,3], normals [G,3], halves [G,2] -> corners [G,4,3]."""
    return jax.vmap(gate_corners)(centers, normals, halves)


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import quad_ctbr_env as E

    pm = E.Params()  # spec gate at (2,0,1.5), half (0.25,0.25)
    corners = gate_corners(pm.gate_center, pm.gate_normal, pm.gate_half)
    f = float(focal_from_fov(90.0, 64))
    R_level = jnp.eye(3)

    # a) approach along x (level)
    poses = [(0.0, 0, 1.5), (0.7, 0, 1.5), (1.2, 0, 1.5), (1.6, 0, 1.5)]
    # b) lateral / vertical offsets at fixed range
    offs = [(1.0, -0.4, 1.5), (1.0, 0.4, 1.5), (1.0, 0.0, 1.1), (1.0, 0.0, 1.9)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax, p in zip(axes[0], poses):
        m = render_mask(jnp.array(p), R_level, corners, 64, f)
        ax.imshow(np.array(m), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"approach x={p[0]}"); ax.axis("off")
    for ax, p in zip(axes[1], offs):
        m = render_mask(jnp.array(p), R_level, corners, 64, f)
        ax.imshow(np.array(m), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"offset y={p[1]},z={p[2]}"); ax.axis("off")
    fig.suptitle("Gate mask renderer — spec gate (2,0,1.5), FOV 90, 64x64")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "runs", "m2", "mask_test.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print("saved", out, "| mask coverage (approach):",
          [round(float(render_mask(jnp.array(p), R_level, corners, 64, f).mean()), 3)
           for p in poses])
