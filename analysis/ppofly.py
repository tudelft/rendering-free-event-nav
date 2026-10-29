"""Reader for PPOFLY01 onboard flight recordings.

Layout (verified against all 30 recordings in analysis/data/onboard_events):

    header  14 B : b'PPOFLY01' + uint16 grid G + uint16 vector length + uint16 reserved
    record 4126 B: uint16 tick
                   G*G   uint8    DIFF3D event frame (128 = no event)
                   4      float32 observation vector
                   2      float32 action (yaw rate, speed), both in [-1, 1]
                   4 B             reserved

NOTE the 14-byte header. An earlier reader in this project assumed 16, which shifted every
frame by two pixels in raster order, pulled two bytes of the vector field in as pixels, and
dropped the final record. Aggregate statistics barely move under a two-pixel shift, but the
frames themselves were wrong; use this reader.
"""
import struct
import numpy as np

MAGIC, HDR, REC = b'PPOFLY01', 14, 4126


def read(path):
    """-> dict(frames [N,G,G] uint8, dev [N,G,G] int16 signed, vec [N,4], act [N,2], tick [N])."""
    d = open(path, 'rb').read()
    assert d[:8] == MAGIC, f'{path}: not PPOFLY01 ({d[:8]!r})'
    G, vlen, _ = struct.unpack('<3H', d[8:HDR])
    body = len(d) - HDR
    assert body % REC == 0, f'{path}: {body} B is not a whole number of {REC} B records'
    n = body // REC
    a = np.frombuffer(d[HDR:], np.uint8).reshape(n, REC)
    frames = a[:, 2:2 + G * G].reshape(n, G, G)
    tail = a[:, 2 + G * G:].copy()
    vec = tail[:, :vlen * 4].view(np.float32).reshape(n, vlen)
    act = tail[:, vlen * 4:vlen * 4 + 8].view(np.float32).reshape(n, 2)
    return dict(frames=frames, dev=frames.astype(np.int16) - 128,
                vec=vec, act=act, tick=a[:, :2].copy().view(np.uint16).ravel(), G=G, n=n)
