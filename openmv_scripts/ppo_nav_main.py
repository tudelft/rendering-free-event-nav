# ppo_nav_main.py -- run the event-nav policy on the OpenMV N6 (feedforward, 13.3 Hz).
#
# Implements deploy/contracts.md:
#   obs   : image[1,64,64,3] float (3-frame DIFF3D stack, oldest->newest) + vector[4]
#   net   : the int8 Neural-ART .bin from deploy/convert/ (outputs the raw action MEAN)
#   action: a = tanh(mean)  -- the tanh is TWO scalar ops here on the Cortex-M55, NOT in the graph
#   link  : N6->CF 0xAA55 | a0,a1 int16*1e4 | XOR  ;  CF->N6 0xCF55 | vector int16*1e4 | XOR
#
# CONTRACT CONSTANTS (champion, vmax=3.5): decision rate 13.3 Hz (PERIOD=75 ms); yaw rate = a0*1.0 rad/s;
# forward speed v = (a1+1)/2 * 3.5 m/s -- both MAPPED ON THE CF (firmware), set there, not here. NO
# output filtering anywhere. The 3-frame stack is flattened NHWC (ch0=oldest, ch2=newest).
#
# Copy this to the board as main.py; put the compiled policy at /sdcard/policy.bin.
import csi, image, time, ml, struct, gc, math
from machine import UART
from ulab import numpy as np
try:
    import framestack                      # custom N6 fw: zero-alloc C frame-stack build (push/fill do the
    _HAS_FS = True                          # channel shift + uint8->float in C). Kills the in-flight GC
except ImportError:                         # freeze (no ~260B/frame ulab churn). See firmware/README.md.
    _HAS_FS = False                         # stock fw: ulab fallback works, but GC can hitch -- see firmware/.

MODEL = '/sdcard/policy.bin'
G, SRC = 64, 320; SC = G / SRC; KSTACK = 3
UART_ID, BAUD = 4, 115200
PERIOD = 75                                # 13.33 Hz = the training temporal contract. Do NOT run other rates.
GC_EVERY = 2000 if _HAS_FS else 15         # custom fw churn ~0 -> collect rarely; 4M heap -> collect ~63ms.
VLEN = 4

model = ml.Model(MODEL); print('ppo_nav_main: model in', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(4)   # 4 buffers: a freshly
# integrated frame is always ready -> snapshot returns in ~15 ms and the 20 ms event integration
# OVERLAPS predict in the background.
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)              # 64x64 AREA target (ulab-fallback path)
_warm = image.Image(SRC, SRC, image.GRAYSCALE)       # 320x320 cached scratch for the copy-trick

_img = np.zeros((1, G, G, KSTACK)); _first = True    # persistent; ch0=oldest ... ch(K-1)=newest
def build_img():
    global _first
    d = _c.snapshot()
    if _HAS_FS:                                       # COPY-TRICK: the camera fb is UNCACHEABLE, so
        # per-element reads are ~10x slow. Burst-copy the fb to a cached image (~0.7 ms) then do a fused
        # box-downsample + stack on the copy (~1.7 ms) -> ~2.4 ms vs ~16 ms for draw_image(AREA).
        _warm.draw_image(d, 0, 0)
        (framestack.fill_ds if _first else framestack.push_ds)(_img, _warm)
        _first = False
    else:                                             # ulab fallback (stock fw): AREA downsample + shift
        _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
        newf = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((G, G))
        if _first:
            for i in range(KSTACK): _img[0, :, :, i] = newf
            _first = False
        else:
            for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]
            _img[0, :, :, KSTACK - 1] = newf
    return _img

goal_vec = [1.0, 0.0, 0.0, 0.0]            # CF sends [cos(brg), sin(brg), yaw_rate/1.0, |v_EKF|/3.5]
vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))

uart = UART(UART_ID, BAUD)
def send_action(a0, a1):                    # N6 -> CF : 0xAA55 | a0,a1 int16*1e4 | XOR(payload)
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
_rx = bytearray()
def poll_vector():                          # CF -> N6 : 0xCF55 | 4*int16*1e4 | XOR(payload)
    global goal_vec, vector, _rx
    if uart.any():
        _rx.extend(uart.read(uart.any()))
    i = 0; n = len(_rx)
    while i + 11 <= n:
        if _rx[i] != 0xCF or _rx[i + 1] != 0x55:
            i += 1; continue
        body = _rx[i + 2:i + 10]; ck = 0
        for b in body:
            ck ^= b
        if (ck & 0xFF) == _rx[i + 10]:
            vals = struct.unpack('<hhhh', body)
            goal_vec = [v / 10000.0 for v in vals]
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

print('ppo_nav_main [feedforward 64x64x3, vmax 3.5 m/s, 13.3 Hz] up.')
gc.collect()
k = 0; _hb = time.ticks_ms()
while True:
    tick = time.ticks_ms()
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    img = build_img()
    out = model.predict([img, vector])                 # graph output = raw action MEAN
    a0 = math.tanh(float(out[0][0][0])); a1 = math.tanh(float(out[0][0][1]))   # tanh squash on the M55
    send_action(a0, a1)                                 # NO output filtering
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz free%d' % (k, a0, a1, goal_vec[0], goal_vec[1], hz, gc.mem_free()))
