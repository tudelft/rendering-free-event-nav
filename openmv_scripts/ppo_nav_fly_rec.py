# ppo_nav_fly_rec.py -- fly the policy (same as ppo_nav_main.py) AND record a flight trace.
#
# Records a fixed MISSION_S window starting from the first CF goal message: per frame the newest 64x64
# obs, the goal vector, and the RAW action [a0,a1]. The 3-frame stacks are rebuilt offline. The log
# doubles as a task-real calibration source (feed it to deploy/convert/make_calib.py --from-events-like
# tooling, or recalibrate PTQ on it).
#
# Format: b'PPOFLY01' | u16 G(=64) | u16 VLEN | u16 pad
#   per frame: u32 t_ms | u16 dt_ms | G*G u8 obs(newest) | VLEN f32 goal | 2 f32 action  (=4126 B)
import csi, image, time, ml, struct, gc, math
from machine import UART
from ulab import numpy as np
try:
    import framestack
    _HAS_FS = True
except ImportError:
    _HAS_FS = False

MODEL = '/sdcard/policy.bin'
LOGPATH = '/sdcard/ppo_nav_fly.bin'
G, SRC = 64, 320; SC = G / SRC; KSTACK = 3
UART_ID, BAUD = 4, 115200
PERIOD = 75; GC_EVERY = 2000 if _HAS_FS else 15
VLEN = 4; MISSION_S = 10; KMAX = 4000

model = ml.Model(MODEL); print('ppo_nav_fly_rec: model in', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(4)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

_img = np.zeros((1, G, G, KSTACK)); _first = True
def build_img():
    global _first
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)   # logged obs (draw_image AREA)
    ob = _g.bytearray()
    if _HAS_FS:
        (framestack.fill if _first else framestack.push)(_img, _g)
        _first = False
    else:
        newf = np.array(np.frombuffer(ob, dtype=np.uint8), dtype=np.float).reshape((G, G))
        if _first:
            for i in range(KSTACK): _img[0, :, :, i] = newf
            _first = False
        else:
            for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]
            _img[0, :, :, KSTACK - 1] = newf
    return _img, ob

goal_vec = [1.0, 0.0, 0.0, 0.0]
vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))
uart = UART(UART_ID, BAUD)
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
_rx = bytearray(); _rx_seen = False
def poll_vector():
    global goal_vec, vector, _rx, _rx_seen
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
            goal_vec = [v / 1e4 for v in struct.unpack('<hhhh', body)]
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN)); _rx_seen = True
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

f = open(LOGPATH, 'wb'); f.write(b'PPOFLY01' + struct.pack('<HHH', G, VLEN, 0))
print('ppo_nav_fly_rec armed -> waiting for first CF goal, then flying + logging %ds' % MISSION_S)
gc.collect()
k = 0; nlog = 0; _hb = time.ticks_ms(); _started = False; t_mission = None; last_tick = time.ticks_ms()
while k < KMAX:
    tick = time.ticks_ms(); dt = time.ticks_diff(tick, last_tick); last_tick = tick
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    if (not _started) and _rx_seen:
        _started = True; t_mission = tick; print('>>> first CF goal -> fly + record START')
    img, ob = build_img()
    out = model.predict([img, vector])
    a0 = math.tanh(float(out[0][0][0])); a1 = math.tanh(float(out[0][0][1]))   # tanh on the M55
    send_action(a0, a1)
    if _started:
        f.write(struct.pack('<IH', time.ticks_diff(tick, t_mission), dt & 0xFFFF))
        f.write(bytes(ob)); f.write(struct.pack('<%df' % VLEN, *goal_vec)); f.write(struct.pack('<2f', a0, a1))
        nlog += 1
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d st%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz' % (k, _started, a0, a1, goal_vec[0], goal_vec[1], hz))
    if _started and time.ticks_diff(tick, t_mission) >= MISSION_S * 1000:
        break
f.close(); send_action(0.0, 0.0)
print('DONE: logged %d frames -> %s' % (nlog, LOGPATH) if nlog else 'DONE: no CF goal received, nothing logged.')
