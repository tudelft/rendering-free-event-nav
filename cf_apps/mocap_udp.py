"""Receive rigid-body position from UnifiedMocapRouter's `udp` agent.

Run the router with the udp agent, pointing at the machine running the flight
script, e.g.:

    ./build/mocap-router optitrack udp -f 100 -c enu -r far -s 1 -n far \
        -i <THIS_MACHINE_IP> -p 5005

Packet layout (from src/agents/udp_agent.hpp + include/pose_calculations.hpp):

    offset  bytes  field
    0       4      ac_id            (uint32)
    4       8      pose.timeUs      (uint64)
    12      4      pose.x           (float)   <- we read from here
    16      4      pose.y           (float)
    20      4      pose.z           (float)
    24..    ...    quaternion + twist (ignored here)

So x/y/z are three little-endian floats at offset 12. We only need position,
so we read those and ignore the rest - robust to the twist struct's padding.

This ONLY listens. It never sends anything to the drone: the mocap is logged as
ground truth, the Crazyflie keeps flying on its own onboard estimate.
"""

import socket
import struct
import threading


class MocapReceiver:
    def __init__(self, port=5005, bind_ip='0.0.0.0'):
        self.port = port
        self.bind_ip = bind_ip
        self._lock = threading.Lock()
        self._latest = None      # (x, y, z) or None
        self._count = 0
        self._running = False
        self._sock = None
        self._thread = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_ip, self.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < 24:          # need at least up to z
                continue
            ts, = struct.unpack_from('<Q', data, 4)     # pose.timeUs (Motive)
            x, y, z = struct.unpack_from('<fff', data, 12)
            with self._lock:
                self._latest = (x, y, z, ts)
                self._count += 1

    def latest(self):
        """Most recent (x, y, z, timeUs), or None if no packet received yet.

        timeUs is Motive's own timestamp (microseconds), for tight time sync
        against the Crazyflie's onboard clock. Position is the first 3 elements
        so callers that only want xyz can still do x, y, z = latest()[:3].
        """
        with self._lock:
            return self._latest

    @property
    def count(self):
        with self._lock:
            return self._count

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
