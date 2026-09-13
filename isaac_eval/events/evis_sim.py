"""EVIS event backend -- wrap the JHU SPIKE Lab event model as a drop-in.

The full EVIS plugin (isaac-sim-event-camera-plugin) targets Isaac Lab, but its
*pure core* (physics-grounded log-intensity model + sensor-noise models) runs
standalone on torch. This wraps `BatchedMultiCamProcessor` behind the same
interface as :class:`~robosim.events.esim.EventSimulator` (`update(frame, t)`
-> `(N,4)` [t,x,y,polarity]) so `EventProcessor(backend="evis")` swaps it in.

Needs the cloned repo on the path (EVIS_PATH env var, or pass ``evis_path``) and
torch (Isaac's python has it). More faithful than the built-in ESIM: linear/HDR
log response, per-pixel threshold mismatch, leak/shot/hot-pixel noise, refractory.
"""

from __future__ import annotations

import os

import numpy as np

# events/ -> robosim/robosim/events ; two levels up = the robosim repo root
_DEFAULT_EVIS = os.path.join(os.path.dirname(__file__), "..", "..",
                             "isaac-sim-event-camera-plugin")


class EvisEventSimulator:
    def __init__(self, height, width, c_pos=0.15, c_neg=0.15, evis_path=None,
                 noise=None, device=None):
        import sys
        path = evis_path or os.environ.get("EVIS_PATH") or os.path.abspath(_DEFAULT_EVIS)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"EVIS repo not found at {path}; clone isaac-sim-event-camera-plugin "
                f"or set EVIS_PATH")
        if path not in sys.path:
            sys.path.insert(0, path)
        import torch
        from dvs_gen import GeneralDVSRecorder, BatchedMultiCamProcessor

        self.torch = torch
        self.height, self.width = int(height), int(width)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # noise may be None (ideal), a dict of DVSNoiseCfg fields, a DVSNoiseCfg,
        # or an already-built DVSNoiseModel.
        noise_model = None
        if noise is not None:
            from dvs_gen.dvs import DVSNoiseCfg, DVSNoiseModel
            if isinstance(noise, dict):
                noise = DVSNoiseCfg(**noise)
            noise_model = noise if isinstance(noise, DVSNoiseModel) else \
                DVSNoiseModel(noise, float(c_pos))

        self._rec = GeneralDVSRecorder(output_dir="/tmp/evis_robosim")
        self._proc = BatchedMultiCamProcessor(
            self._rec, "cam", threshold=float(c_pos), noise=noise_model)
        self._proc.stash_events = True     # expose per-frame (pos,neg) masks
        # Monotonic clock fed to the model. The caller's per-frame timestamps may
        # restart at 0 each capture window, but the noise model's refractory /
        # leak / shot timers require ever-increasing time -- otherwise a backwards
        # time jump leaves every pixel stuck in refractory and events vanish.
        self._mono_t = 0.0
        self._dt = 0.01

    def reset(self):
        # New reference frame for the next window; the monotonic clock keeps going.
        self._proc.ref_log_intensity = None
        self._proc.needs_reset_mask = None
        self._proc.last_masks = None

    def update_net(self, frame, t):
        """Accumulate this sub-frame's net polarity (ON-OFF) straight on the GPU.

        The event *list* path (`update`) pulls every event's (x, y) back to the
        host: at c=0.06 roughly 70% of a 320x320 array fires per sub-step, i.e.
        ~570k coordinates per sensor frame. That marshalling is single-threaded
        numpy and became the whole bottleneck -- 30-40 s/frame with the GPU at
        3%. For histogram/GenX320 output we only ever need the net count per
        pixel, so sum the masks on-device and move one (H,W) int16 array instead.

        Returns None for the reference frame (no masks yet).
        """
        pm, nm = self._masks_for(frame)
        if pm is None:
            return None
        return (pm.to(self.torch.int16) - nm.to(self.torch.int16))

    def _masks_for(self, frame):
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        batch = self.torch.as_tensor(arr[None], dtype=self.torch.float32,
                                     device=self.device)
        self._mono_t += self._dt                                   # monotonic clock
        self._proc(batch, self._mono_t)
        masks = self._proc.last_masks
        if masks is None:
            return None, None
        return masks[0][0], masks[1][0]

    def update(self, frame, t) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        batch = self.torch.as_tensor(arr[None], dtype=self.torch.float32,
                                     device=self.device)          # (1,H,W,3)
        self._mono_t += self._dt                                   # monotonic clock
        self._proc(batch, self._mono_t)
        masks = self._proc.last_masks
        if masks is None:                  # first (reference) frame
            return np.empty((0, 4), dtype=np.float64)
        pm, nm = masks[0][0], masks[1][0]  # (H,W) bool
        ys_p, xs_p = self.torch.nonzero(pm, as_tuple=True)
        ys_n, xs_n = self.torch.nonzero(nm, as_tuple=True)
        xs = self.torch.cat([xs_p, xs_n]).float().cpu().numpy()
        ys = self.torch.cat([ys_p, ys_n]).float().cpu().numpy()
        pol = np.concatenate([np.ones(len(xs_p)), -np.ones(len(xs_n))])
        n = len(xs)
        ev = np.empty((n, 4), dtype=np.float64)
        ev[:, 0] = float(t)
        ev[:, 1] = xs
        ev[:, 2] = ys
        ev[:, 3] = pol
        return ev
