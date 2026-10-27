"""Takeoff / hand-over / land wrapper for the onboard N6 avoidance app.

The flying is fully ONBOARD (the n6_uart app cruises + steers from the N6). This
script only:
  1. takes off to cruise altitude (hover setpoints),
  2. hands control to the onboard policy  (set n6.enable = 1),
  3. monitors telemetry,
  4. on Ctrl+C / timeout / low battery: n6.enable = 0 and lands IN PLACE.

Coordination: the app injects setpoints ONLY when n6.enable=1; this script sends
setpoints ONLY when it is 0. So exactly one controller is active at a time.

FIRST RUNS: DRY_RUN=True with props OFF, then a real flight in a LARGE space.
"""

import sys
import time
from datetime import datetime

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from flightlog import FlightLogger
from mocap_udp import MocapReceiver

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')   # or set CFLIB_URI

DRY_RUN     = False     # props OFF for the first runs
CRUISE_ALT  = 0.80     # m   -> sets n6.cruiseAlt (fixed hold altitude)
# The N6 policy commands the SPEED (0-3.5 m/s, deploy/contracts.md). Flying too fast
# for the height saturates the flow deck, so the app CLAMPS v_cmd to n6.maxSpeed.
MAX_SPEED   = 3.5     # m/s -> sets n6.maxSpeed (flow-deck speed cap)
#  FLOW-DECK LIMIT: flow saturates ~ v/h = 7 rad/s. At 0.25 m keep MAX_SPEED
#  <= ~1.2 m/s (start 0.5).
# a1 (policy speed action) -> commanded speed, mapping owned here:
#     v_cmd = VMIN + 0.5 * (clip(a1,-1,1) + 1) * (VMAX - VMIN)
# a1=-1 -> VMIN, a1=+1 -> VMAX. VMAX is also the obs[3] speed normalisation the
# CF uplinks to the N6 (|v_EKF| / VMAX), so it must match the policy (3.5).
# n6.maxSpeed still applies as a HARD safety clamp on top, so keep VMAX <= MAX_SPEED.
# NOTE: only affects real speed in VELOCITY mode (POS_MODE=False); in posMode the
# speed is posCtlPid velMax and a1 is only a move/hover gate.
VMIN        = 0.0        # m/s at a1=-1  (policy 'brake/stop')
VMAX        = MAX_SPEED  # m/s at a1=+1  (policy 'full speed'); keep <= MAX_SPEED
YAW_GAIN    = 1.0      # -> n6.yawGain: extra scale on the N6 yaw command (1.0 = as trained)
YAW_SCALE   = 1.0      # -> n6.yawScale: rad/s that a0=1 maps to, AND the obs[2] norm.
                       #    MUST match the policy (deploy/contracts.md: 1.0 rad/s)
# --- GOAL: the fixed point the drone flies toward (drives the bearing sent to the N6).
# Coordinates are in the TAKEOFF frame (estimator is reset at takeoff -> +x = takeoff
# heading, +y = left). Set at hand-over (n6.enable=1), so the drone must be pointing
# where you want "forward" when it takes over.
USE_GOAL_XY = True     # True -> explicit (GOAL_X, GOAL_Y); False -> GOAL_DIST straight ahead
GOAL_X      = 5.0     # m  goal x (forward from takeoff)
GOAL_Y      = 0.0      # m  goal y (left from takeoff)
GOAL_DIST   = 10.0     # m  used only when USE_GOAL_XY=False (distance ahead of heading)
POS_MODE    = True    # True -> n6.posMode=1 (position "carrot"); False -> velocity (faithful)
LOOKAHEAD   = 0.6      # m   -> n6.lookahead (carrot distance ahead of nose, posMode only)
VEL_CAP     = MAX_SPEED # m/s -> caps position-mode cruise via posCtlPid.xVelMax/yVelMax
TAKEOFF_S   = 1.0
HOVER_S     = 1.5      # settle before handing over
RATE_HZ     = 25
MISSION_MAX_S = 30.0   # hard cap on autonomous flight
LOW_VBAT    = 3.15
USE_MOCAP   = False    # True -> log mocap ground truth (mx/my/mz) alongside the
                       # onboard estimate. This is what tells us whether the
                       # 'sideways drift' is REAL-WORLD motion or just the
                       # estimate. Needs a UnifiedMocapRouter `udp` agent (see
                       # mocap_udp.py and README.md).
MOCAP_PORT  = 5005


def hover(cf, z, yaw_rate=0.0):
    cf.commander.send_hover_setpoint(0.0, 0.0, yaw_rate, z)


def ramp(cf, z0, z1, seconds):
    n = max(1, int(seconds * RATE_HZ))
    for i in range(n + 1):
        hover(cf, z0 + (z1 - z0) * i / n)
        time.sleep(1.0 / RATE_HZ)


def reset_estimator(cf):
    # Zero persistent initial pos, reset, settle briefly. Call RIGHT before
    # takeoff: the flow deck is blind < ~8 cm, so the longer the drone sits on
    # the ground after a reset, the more x/y drifts before it ever leaves.
    for n in ('kalman.initialX', 'kalman.initialY', 'kalman.initialZ',
              'kalman.initialYaw'):
        try: cf.param.set_value(n, '0')
        except Exception: pass
    time.sleep(0.2)
    cf.param.set_value('kalman.resetEstimation', '1'); time.sleep(0.2)
    cf.param.set_value('kalman.resetEstimation', '0'); time.sleep(1.5)


def main():
    cflib.crtp.init_drivers()
    print('=' * 56)
    print(f'  Onboard N6 avoidance. DRY_RUN={DRY_RUN}  cruiseAlt={CRUISE_ALT} m')
    print('  Props OFF for DRY_RUN. LARGE clear space for real flight.')
    print('=' * 56)

    with SyncCrazyflie(URI) as scf:
        cf = scf.cf
        print('Connected.')

        if int(cf.param.get_value('deck.bcFlow2')) != 1:
            print('Flow deck NOT detected - aborting.'); return

        # make sure the onboard policy starts DISABLED, then push the cruise config
        cf.param.set_value('n6.enable', 0)
        cf.param.set_value('n6.cruiseAlt', CRUISE_ALT)
        cf.param.set_value('n6.maxSpeed', MAX_SPEED)   # flow-deck HARD safety cap
        cf.param.set_value('n6.vmin', VMIN)            # a1 -> speed mapping (velocity mode)
        cf.param.set_value('n6.vmax', VMAX)
        cf.param.set_value('n6.yawGain', YAW_GAIN)     # extra yaw scale (1.0 = as trained)
        cf.param.set_value('n6.yawScale', YAW_SCALE)   # yaw rad/s scale + obs[2] norm (MATCH policy!)
        cf.param.set_value('n6.useGoalXY', 1 if USE_GOAL_XY else 0)  # goal point vs ahead-of-heading
        cf.param.set_value('n6.goalX', GOAL_X)         # goal (takeoff frame), used at hand-over
        cf.param.set_value('n6.goalY', GOAL_Y)
        cf.param.set_value('n6.goalDist', GOAL_DIST)   # only if useGoalXY=0
        cf.param.set_value('n6.posMode', 1 if POS_MODE else 0)
        cf.param.set_value('n6.lookahead', LOOKAHEAD)
        if POS_MODE:
            # Cap the position-controller cruise speed. RAM writes (not persisted),
            # so they reset to firmware defaults on power-cycle.
            cf.param.set_value('posCtlPid.xVelMax', VEL_CAP)
            cf.param.set_value('posCtlPid.yVelMax', VEL_CAP)
        mode = f'POSITION carrot (lookahead={LOOKAHEAD} m)' if POS_MODE else 'VELOCITY'
        print(f'Cruise config: {mode}  v={VMIN}-{VMAX} m/s  yawScale={YAW_SCALE} rad/s  '
              f'yawGain={YAW_GAIN}  cruiseAlt={CRUISE_ALT} m')

        # telemetry (a0/a1 = raw N6 action; yawRateCmd deg/s; speedCmd m/s)
        state = {'rxFrames': 0, 'linkOk': 0, 'a0raw': 0.0, 'a1raw': 0.0,
                 'yawRateCmd': 0.0, 'speedCmd': 0.0, 'vbat': 4.0}
        lg = LogConfig(name='m', period_in_ms=100)
        for v in ('n6.rxFrames', 'n6.linkOk', 'n6.yawRateCmd', 'n6.speedCmd',
                  'pm.vbat'):
            lg.add_variable(v)
        cf.log.add_config(lg)
        lg.data_received_cb.add_callback(
            lambda t, d, c: state.update({k.split('.')[1]: d[k] for k in d}))
        lg.start()

        # CSV flight log: onboard estimate (x/y/z/yaw) + raw optical flow
        # (deltaX/deltaY) + optional mocap truth. This is the record we plot
        # afterward to see whether it flew along its nose or drifted sideways.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f'n6_mission_{stamp}.csv'
        mocap = None
        if USE_MOCAP:
            mocap = MocapReceiver(port=MOCAP_PORT)
            mocap.start()
            print(f'Mocap UDP listener on :{MOCAP_PORT} (log-only ground truth).')
        flog = FlightLogger(csv_path, mocap_source=mocap.latest if mocap else None,
                            n6=True)   # also log the N6 action + goal-bearing
        flog.attach(cf)
        print(f'Logging flight to {csv_path}')

        cf.supervisor.send_arming_request(True)
        time.sleep(1.0)

        t0 = time.time()
        try:
            # reset the estimator RIGHT before takeoff (minimises ground drift)
            print('Resetting estimator ...')
            reset_estimator(cf)
            print('Taking off ...')
            ramp(cf, 0.05, CRUISE_ALT, TAKEOFF_S)
            for _ in range(int(HOVER_S * RATE_HZ)):
                hover(cf, CRUISE_ALT); time.sleep(1.0 / RATE_HZ)

            print('Handing over to ONBOARD policy (n6.enable=1). '
                  'This script stops commanding now.')
            cf.param.set_value('n6.enable', 1)

            tmission = time.time()
            while True:
                time.sleep(0.2)
                el = time.time() - tmission
                print(f'  t={el:5.1f}s  rxFrames={state["rxFrames"]:6d}  '
                      f'linkOk={state["linkOk"]}  yawRate={state["yawRateCmd"]:+5.0f}  '
                      f'speed={state["speedCmd"]:+.2f}  vbat={state["vbat"]:.2f}')
                if el > MISSION_MAX_S:
                    print('Mission time cap reached.'); break
                if state['vbat'] and state['vbat'] < LOW_VBAT:
                    print('Low battery.'); break
                if not state['linkOk']:
                    print('  (N6 link not fresh - app is hovering)')
        except KeyboardInterrupt:
            print('\nEMERGENCY STOP')
        finally:
            print('Disabling policy and landing in place ...')
            try: cf.param.set_value('n6.enable', 0)
            except Exception: pass
            time.sleep(0.1)
            try:
                ramp(cf, CRUISE_ALT, 0.05, 2.0)
            except Exception:
                pass
            cf.commander.send_stop_setpoint()
            cf.supervisor.send_arming_request(False)
            try: lg.stop()
            except Exception: pass
            try:
                flog.close()
                print(f'Wrote {flog.rows} rows to {csv_path}')
            except Exception: pass
            if mocap is not None:
                print(f'Mocap: {mocap.count} packets total.')
                try: mocap.stop()
                except Exception: pass
            print(f'Done. {time.time()-t0:.1f}s total.')


if __name__ == '__main__':
    main()
