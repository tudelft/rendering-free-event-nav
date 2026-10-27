"""Warp-accelerated EVIS event generation (the paper's Path B, no Isaac Lab).

How it works:
render sparse RTX *keyframes* (rgb/HDR + motion_vectors + depth), synthesise the
K-1 in-between frames on the GPU with EVIS's bidirectional motion-vector warp
(`dvs_gen.warp.bidir_warp_gap`, pure torch), and feed every fine frame to EVIS's
log-intensity event model (`BatchedMultiCamProcessor`). Events therefore run at
``f_render x K`` Hz while the renderer runs at ``f_render``.

MEASURED on this install (Isaac 4.2, RTX 4090, 320x320, examples/diag_warp_*):
  * `motion_vectors` annotator returns (H, W, 4) float32; channels [0:2] are the
    screen-space displacement in PIXELS, row-down, with the warp's exact
    convention ``earlier_pos = pos + mv`` (verified against analytic camera
    motion, both axes; NO sign flips needed).
  * `HdrColor` is available: (H, W, 4) float16 LINEAR radiance -> physically
    correct ``log(I)`` for the event model (the paper's HDR source).
  * render+step+read = ~28 ms/keyframe with the camera moved via direct USD
    xform writes (never `rep.modify.pose`, whose graph rebuilds grow unbounded).
  * bidir_warp_gap K=8 @320x320: ~11 ms/gap first-call JIT excluded (~1.4 ms/
    subframe), events ~0.5 ms/frame.

Two classes, split so the torch side is testable without Isaac:

* :class:`KeyframeCamera` (Isaac-side) — owns a plain USD camera prim, a render
  product and the annotators; ``capture()`` returns GPU torch tensors.
* :class:`WarpEventStream` (pure torch) — consumes keyframes, warps the gaps,
  runs the EVIS event model, accumulates per-pixel ON/OFF counts. Pop the
  accumulated counts whenever your sensor's integration window closes (GenX320:
  every 20 ms) — the window is decoupled from the keyframe rate.

Typical use (inside a running SimulationApp)::

    cam = KeyframeCamera(res=(320, 320), event_source="hdr").build()
    stream = WarpEventStream(320, 320, threshold=0.15)
    for pose in trajectory_at_render_hz:          # e.g. 50 Hz
        cam.set_pose(pose.pos, yaw_deg=pose.yaw, pitch_deg=pose.pitch)
        KeyframeCamera.render()
        stream.push(*cam.capture(), t=pose.t, K=8)  # 400 Hz effective events
        if pose.t >= next_frame_t:                # 20 ms sensor frame closed
            on, off = stream.pop_counts()
            gray = net_to_gray_genx320((on - off).cpu().numpy())
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np

_DEFAULT_EVIS = os.path.join(os.path.dirname(__file__), "..", "evis")   # the submodule


def _ensure_evis_on_path(evis_path: Optional[str] = None) -> str:
    path = evis_path or os.environ.get("EVIS_PATH") or os.path.abspath(_DEFAULT_EVIS)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"EVIS repo not found at {path}; clone isaac-sim-event-camera-plugin "
            f"or set EVIS_PATH")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


class _NullRecorder:
    """Recorder stub: we consume events as per-frame masks / counts, not HDF5.

    `BatchedMultiCamProcessor.record()` appends every event to in-memory lists,
    which grows without bound on long runs; for count/frame outputs we never
    need the list, so this swallows them. Use a real `GeneralDVSRecorder` (pass
    ``recorder=``) when you want the raw (x, y, t, p) stream on disk.
    """

    def record(self, *a, **k):  # noqa: D401
        pass

    def flush_episode(self, *a, **k):
        pass


class WarpEventStream:
    """EVIS event model + motion-vector warp, GPU-resident, single camera.

    Feed one keyframe per render via :meth:`push`; every gap is expanded to
    ``K`` fine frames (1 real + K-1 warped) and each is run through the EVIS
    log-intensity model. Per-pixel ON/OFF counts accumulate until
    :meth:`pop_counts`.

    Parameters
    ----------
    threshold : contrast threshold C (log-intensity units), paper default 0.15.
    noise : None (ideal) or a dict of ``DVSNoiseCfg`` fields (threshold
        mismatch, refractory, leak/shot/hot pixels...).
    mv_dilate : grow each mv field over anti-aliased silhouette fringes
        (px radius). With AA off 0 is fine; use 1 if AA must stay on.
    hole_fill : warp double-occlusion fill; None = black (no fabrication).
    """

    def __init__(self, height: int, width: int, threshold: float = 0.15,
                 noise: Optional[dict] = None, device: Optional[str] = None,
                 mv_dilate: int = 0, hole_fill=None, evis_path: Optional[str] = None,
                 recorder=None, camera_name: str = "cam"):
        _ensure_evis_on_path(evis_path)
        import torch
        from dvs_gen.dvs.processor import BatchedMultiCamProcessor
        from dvs_gen.warp.interpolation import bidir_warp_gap

        self.torch = torch
        self._warp = bidir_warp_gap
        self.height, self.width = int(height), int(width)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mv_dilate = int(mv_dilate)
        self.hole_fill = hole_fill

        noise_model = None
        if noise:
            from dvs_gen.dvs.noise import DVSNoiseCfg, DVSNoiseModel
            noise_model = DVSNoiseModel(DVSNoiseCfg(**dict(noise)), float(threshold))
        self._proc = BatchedMultiCamProcessor(
            recorder if recorder is not None else _NullRecorder(),
            camera_name, threshold=float(threshold), noise=noise_model)
        self._proc.stash_events = True

        self._on = torch.zeros(self.height, self.width, dtype=torch.int32,
                               device=self.device)
        self._off = torch.zeros_like(self._on)
        self._prev = None            # (rgb, mv, depth, t) of the last keyframe
        # optional fixed-length integration windows, counted in FINE frames so
        # window boundaries land on the same fine-time grid regardless of K
        # (a 20 ms window at 240 Hz fine rate = 5 fine frames in EVERY config).
        # When set, every completed window's (on, off) counts are appended to
        # ``self.windows`` and the accumulators reset.
        self.window_frames: Optional[int] = None
        self._win_count = 0
        self.windows = []
        # stats for benchmarking
        self.frames_fed = 0
        self.t_warp = 0.0
        self.t_event = 0.0

    # -- internals ---------------------------------------------------------
    def _feed(self, frame, t: float):
        """One fine frame (H, W, 3) float -> event model -> count accumulators."""
        self._proc(frame[None], float(t))
        masks = self._proc.last_masks
        self.frames_fed += 1
        if masks is None:
            return
        pm, nm = masks[0][0], masks[1][0]
        self._on += pm.to(self.torch.int32)
        self._off += nm.to(self.torch.int32)
        if self.window_frames:
            self._win_count += 1
            if self._win_count >= self.window_frames:
                self.windows.append((self._on.clone(), self._off.clone()))
                self._on.zero_()
                self._off.zero_()
                self._win_count = 0

    # -- public API ---------------------------------------------------------
    def reset(self):
        """New reference frame next push (episode/window reset); keep the clock."""
        self._proc.ref_log_intensity = None
        self._proc.needs_reset_mask = None
        self._proc.last_masks = None
        self._prev = None
        self._on.zero_()
        self._off.zero_()
        # window bookkeeping restarts too: a stale mid-window count would
        # phase-shift every window of the next episode by up to K-1 fine frames
        self._win_count = 0
        self.windows = []

    def push(self, rgb, mv, depth, t: float, K: int = 8):
        """Consume one rendered keyframe; emit K fine frames into the model.

        rgb   : (H, W, 3) float tensor (linear HDR or LDR/255) on ``device``.
        mv    : (H, W, 2) float tensor, pixels, ``earlier = pos + mv``.
        depth : (H, W) float tensor (metres) or None (no occlusion resolve).
        t     : keyframe timestamp (s); MUST be monotonic across pushes.
        K     : warp multiplier; K=1 disables interpolation (dense baseline).

        The K-1 intermediates are synthesised between the PREVIOUS keyframe and
        this one, then this keyframe is fed last, so after every push the model
        has seen everything up to time ``t``.
        """
        torch = self.torch
        rgb = rgb.to(self.device)
        mv = mv.to(self.device)
        depth = depth.to(self.device) if depth is not None else None

        if self._prev is None:
            self._feed(rgb, t)                       # reference frame
            self._prev = (rgb, mv, depth, float(t))
            return

        pA, mvA, dA, tA = self._prev
        K = max(1, int(K))
        if K > 1:
            if torch.cuda.is_available():
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            mids = self._warp(pA, rgb, mvA, mv, K,
                              depthA=dA, depthB=depth,
                              hole_fill=self.hole_fill, mv_dilate=self.mv_dilate)
            if torch.cuda.is_available():
                ev1.record(); ev1.synchronize()
                self.t_warp += ev0.elapsed_time(ev1) / 1e3
            dt = (float(t) - tA) / K
            for i, f in enumerate(mids):
                self._feed(f, tA + (i + 1) * dt)
        self._feed(rgb, float(t))
        self._prev = (rgb, mv, depth, float(t))

    def pop_counts(self):
        """Return (on, off) int32 (H, W) accumulated since the last pop; reset."""
        on, off = self._on.clone(), self._off.clone()
        self._on.zero_()
        self._off.zero_()
        self._win_count = 0          # a pop empties the current window too
        return on, off

    def peek_net(self):
        """Net polarity (ON - OFF) without resetting the window."""
        return (self._on - self._off)


class KeyframeCamera:
    """A plain-USD camera + render product + the three warp annotators.

    Built for tight render loops: the pose is written straight to the USD
    xform (constant cost), never through ``rep.modify.pose`` (whose
    accumulated graph rebuilds reach 30-60 s/frame after a few hundred calls
    -- measured, see imav2026_course/gen_yolo_dataset.py).

    ``event_source="hdr"`` captures the LINEAR ``HdrColor`` buffer for the
    event model (physically correct log response); ``"ldr"`` captures the
    tone-mapped ``rgb`` buffer scaled to [0, 1].
    """

    #: euler XYZ (deg) that points the USD camera's -Z axis at world +X with
    #: +X_cam = world -Y (image right = drone right), verified by centre-depth
    #: and mv-sign diagnostics.
    _BASE_ROT = (90.0, 0.0, -90.0)

    def __init__(self, res: Tuple[int, int] = (320, 320), fov_deg: float = 90.0,
                 event_source: str = "hdr", prim_path: str = "/World/warp_cam",
                 antialiasing: str = "Off", device: Optional[str] = None,
                 rgb_annotator: bool = False):
        self.res = (int(res[0]), int(res[1]))
        self.fov_deg = float(fov_deg)
        if event_source not in ("hdr", "ldr"):
            raise ValueError(f"event_source must be hdr|ldr, got {event_source!r}")
        self.event_source = event_source
        self.prim_path = prim_path
        self.antialiasing = antialiasing
        self.device = device  # resolved in build()
        self.rgb_annotator = rgb_annotator or event_source == "ldr"
        self._annots = {}
        self._cam_xf = None
        self._torch = None

    def build(self) -> "KeyframeCamera":
        import omni.replicator.core as rep
        import omni.usd
        import torch
        from pxr import Gf, UsdGeom

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        stage = omni.usd.get_context().get_stage()
        W, H = self.res

        cam = UsdGeom.Camera.Define(stage, self.prim_path)
        # square aperture; horizontal FOV = fov_deg
        hap = 24.0
        focal = hap / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        cam.GetFocalLengthAttr().Set(focal)
        cam.GetHorizontalApertureAttr().Set(hap)
        cam.GetVerticalApertureAttr().Set(hap * H / W)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 500.0))
        self._cam_xf = UsdGeom.XformCommonAPI(cam.GetPrim())
        self.set_pose((0.0, 0.0, 1.0))

        rp = rep.create.render_product(self.prim_path, (W, H))
        names = ["motion_vectors", "distance_to_image_plane"]
        names.append("HdrColor" if self.event_source == "hdr" else "rgb")
        if self.rgb_annotator and "rgb" not in names:
            names.append("rgb")
        for n in names:
            a = rep.AnnotatorRegistry.get_annotator(n)
            a.attach(rp)
            self._annots[n] = a

        if self.antialiasing is not None:
            try:
                rep.settings.set_render_rtx_realtime(antialiasing=self.antialiasing)
            except Exception as ex:  # non-fatal; mv_dilate can compensate
                print(f"[KeyframeCamera] antialiasing={self.antialiasing!r} "
                      f"failed: {ex}", flush=True)
        rep.orchestrator.set_capture_on_play(False)
        return self

    # -- pose ----------------------------------------------------------------
    def set_pose(self, pos, yaw_deg: float = 0.0, pitch_deg: float = 0.0):
        """Place the camera; yaw about world +Z (0 = facing +X), pitch down +."""
        from pxr import Gf, UsdGeom
        self._cam_xf.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]),
                                           float(pos[2])))
        rx, ry, rz = self._BASE_ROT
        self._cam_xf.SetRotate(Gf.Vec3f(rx - float(pitch_deg), ry,
                                        rz + float(yaw_deg)),
                               UsdGeom.XformCommonAPI.RotationOrderXYZ)

    @staticmethod
    def render(rt_subframes: int = 0):
        import omni.replicator.core as rep
        rep.orchestrator.step(rt_subframes=rt_subframes, pause_timeline=False)

    # -- capture ---------------------------------------------------------------
    def capture(self):
        """Read the last render -> (rgb, mv, depth) torch tensors on ``device``.

        rgb is (H, W, 3) float32: linear HDR radiance (hdr) or LDR/255 (ldr).
        mv is (H, W, 2) float32 pixels; depth (H, W) float32 metres.
        """
        torch = self._torch
        if self.event_source == "hdr":
            raw = np.asarray(self._annots["HdrColor"].get_data())[..., :3]
            rgb = torch.as_tensor(raw.astype(np.float32), device=self.device)
        else:
            raw = np.asarray(self._annots["rgb"].get_data())[..., :3]
            rgb = torch.as_tensor(raw.astype(np.float32), device=self.device) / 255.0
        mv_np = np.nan_to_num(
            np.asarray(self._annots["motion_vectors"].get_data())[..., :2])
        mv = torch.as_tensor(mv_np.astype(np.float32), device=self.device)
        d_np = np.nan_to_num(
            np.asarray(self._annots["distance_to_image_plane"].get_data()),
            posinf=1e4, neginf=1e4)
        depth = torch.as_tensor(d_np.astype(np.float32), device=self.device)
        return rgb, mv, depth

    def capture_rgb_u8(self):
        """Tone-mapped uint8 rgb for videos/debug (needs rgb_annotator=True)."""
        if "rgb" in self._annots:
            return np.asarray(self._annots["rgb"].get_data())[..., :3].copy()
        raw = np.asarray(self._annots["HdrColor"].get_data())[..., :3]
        return (np.clip(raw.astype(np.float32), 0, 1) * 255).astype(np.uint8)


def downsample_counts(counts, out_hw: Tuple[int, int]):
    """Sum-pool an (H, W) count tensor to ``out_hw`` (must divide evenly)."""
    H, W = counts.shape[-2], counts.shape[-1]
    oh, ow = out_hw
    if H % oh or W % ow:
        raise ValueError(f"{(H, W)} not divisible by {out_hw}")
    return (counts.reshape(*counts.shape[:-2], oh, H // oh, ow, W // ow)
            .sum(dim=(-3, -1)))
