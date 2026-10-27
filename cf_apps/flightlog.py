"""Shared two-block flight logger -> single CSV.

Two blocks are needed because one LogConfig packet only holds ~26 bytes
(about 6 floats). The pose block drives row writing; the flow block just
refreshes the most recent values.
"""

import csv
import threading
import time

from cflib.crazyflie.log import LogConfig

LOG_PERIOD_MS = 50   # 20 Hz per block -> 40 packets/s total, well within budget

# time_s = host wall-clock (common reference for both streams).
# cf_ts  = Crazyflie onboard timestamp (ms since boot) from the CRTP log packet.
# (mocap adds mx/my/mz + m_ts = Motive timeUs when a mocap_source is given.)
CSV_COLUMNS = [
    'time_s', 'cf_ts', 'x', 'y', 'z', 'yaw',
    'deltaX', 'deltaY', 'zrange', 'varX', 'vbat',
]


class FlightLogger:
    N6_COLS = ('a0raw', 'a1raw', 'yawRateCmd', 'speedCmd', 'goalx', 'goaly')

    def __init__(self, path, mocap_source=None, n6=False):
        # mocap_source: optional callable returning (x, y, z) ground-truth from
        # an external system (e.g. mocap_udp.MocapReceiver.latest), or None. It
        # is READ ONLY here - logged next to the onboard estimate, never fed
        # back to the drone. When given, three extra columns mx/my/mz appear.
        # n6: when True, also log the n6 app's ACTION + goal-bearing channels
        #     (only present when the n6_uart app is flashed) so a run records
        #     what the N6 commanded vs what the drone did.
        self._mocap_source = mocap_source
        self._n6 = n6
        columns = list(CSV_COLUMNS)
        if n6:
            columns += list(self.N6_COLS)
        if mocap_source is not None:
            columns += ['mx', 'my', 'mz', 'm_ts']
        self._file = open(path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=columns)
        self._writer.writeheader()
        self._lock = threading.Lock()
        self._latest_flow = {k: '' for k in
                             ('deltaX', 'deltaY', 'zrange', 'varX', 'vbat')}
        self._latest_n6 = {k: '' for k in self.N6_COLS}
        self._configs = []
        self._t0 = time.time()
        self.rows = 0
        # Wall-clock time of the last packet received, used by the link watchdog.
        self.last_packet = time.time()
        # Most recent (x, y, z) estimate, or None until the first pose packet.
        self._latest_pose = None

    def attach(self, cf):
        # Types are deliberately omitted so cflib uses the firmware's own TOC
        # types (range.zrange in particular is not a float on all builds).
        pose = LogConfig(name='pose', period_in_ms=LOG_PERIOD_MS)
        for v in ('stateEstimate.x', 'stateEstimate.y',
                  'stateEstimate.z', 'stateEstimate.yaw'):
            pose.add_variable(v)

        flow = LogConfig(name='flow', period_in_ms=LOG_PERIOD_MS)
        for v in ('motion.deltaX', 'motion.deltaY', 'range.zrange',
                  'kalman.varX', 'pm.vbat'):
            flow.add_variable(v)

        blocks = [(pose, self._on_pose), (flow, self._on_flow)]

        if self._n6:
            n6b = LogConfig(name='n6f', period_in_ms=LOG_PERIOD_MS)
            for v in ('n6.a0raw', 'n6.a1raw', 'n6.yawRateCmd',
                      'n6.speedCmd', 'n6.goalx', 'n6.goaly'):
                n6b.add_variable(v)
            blocks.append((n6b, self._on_n6))

        for conf, cb in blocks:
            cf.log.add_config(conf)
            conf.data_received_cb.add_callback(cb)
            conf.start()
            self._configs.append(conf)

    def _on_flow(self, timestamp, data, logconf):
        with self._lock:
            self.last_packet = time.time()
            self._latest_flow = {
                'deltaX': data['motion.deltaX'],
                'deltaY': data['motion.deltaY'],
                'zrange': data['range.zrange'],
                'varX': data['kalman.varX'],
                'vbat': data['pm.vbat'],
            }

    def _on_n6(self, timestamp, data, logconf):
        with self._lock:
            self._latest_n6 = {k: data['n6.' + k] for k in self.N6_COLS}

    def latest_pose(self):
        """Most recent (x, y, z) estimate, or None if no packet has arrived."""
        with self._lock:
            return self._latest_pose

    def latest_vbat(self):
        """Most recent battery voltage (float), or None if not seen yet."""
        with self._lock:
            v = self._latest_flow.get('vbat', '')
            return float(v) if v != '' else None

    def _on_pose(self, timestamp, data, logconf):
        with self._lock:
            self.last_packet = time.time()
            self._latest_pose = (data['stateEstimate.x'],
                                 data['stateEstimate.y'],
                                 data['stateEstimate.z'])
            row = {
                'time_s': round(time.time() - self._t0, 3),
                'cf_ts': timestamp,          # onboard ms-since-boot, for sync
                'x': data['stateEstimate.x'],
                'y': data['stateEstimate.y'],
                'z': data['stateEstimate.z'],
                'yaw': data['stateEstimate.yaw'],
            }
            row.update(self._latest_flow)
            if self._n6:
                row.update(self._latest_n6)
            if self._mocap_source is not None:
                m = self._mocap_source()
                if m is not None:
                    row['mx'], row['my'], row['mz'] = m[0], m[1], m[2]
                    row['m_ts'] = m[3] if len(m) > 3 else ''
                else:
                    row['mx'] = row['my'] = row['mz'] = row['m_ts'] = ''
            self._writer.writerow(row)
            self.rows += 1

    def silent_for(self):
        """Seconds since the last log packet arrived (0 if data is flowing)."""
        with self._lock:
            return time.time() - self.last_packet

    def close(self):
        for conf in self._configs:
            try:
                conf.stop()
            except Exception:
                pass
        self._file.close()
