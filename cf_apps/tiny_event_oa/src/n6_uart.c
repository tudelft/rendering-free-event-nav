/*
 * n6_uart.c - Onboard event-avoidance link + cruise controller.
 *   OpenMV N6 (PPO event policy) <-> Crazyflie 2.1 Brushless, over UART2.
 *
 * This implements the UART link in deploy/contracts.md: the N6 emits an
 * ACTION (yaw-rate + speed) every tick, and the CF streams back the policy's
 * OBSERVATION VECTOR (goal bearing + its own yaw-rate + speed). The app READS
 * the N6 action and COMMANDS the drone with it.
 *
 * WIRING (UART2, 115200 8N1, 3.3V, crossed, common GND):
 *   CF PA2 (UART2 TX) -> N6 UART4 RX ;  CF PA3 (UART2 RX) <- N6 UART4 TX ;  GND
 *
 * DOWNLINK  N6 -> CF   (7 bytes) : the ACTION
 *   0xAA 0x55 | int16 a0 | int16 a1 | xor(4 payload bytes)
 *   int16 little-endian = value*10000  (a0,a1 in [-1,1]).
 *   a0 = yaw_rate_cmd / yawScale -> yaw_rate_setpoint = a0 * n6.yawScale rad/s
 *   a1 = speed_cmd in [-1,1]     -> v_cmd = vmin + (a1+1)/2 * (vmax - vmin)
 *   Contract values: yawScale = 1.0 rad/s, vmin..vmax = 0..3.5 m/s.
 *   Frame dropped if xor mismatches.
 *
 * UPLINK    CF -> N6   (11 bytes) : the OBSERVATION VECTOR[4]
 *   0xCF 0x55 | int16 cos*1e4 | int16 sin*1e4 | int16 (yawrate/yawScale)*1e4
 *             | int16 (v/vmax)*1e4 | xor(8 payload bytes)
 *   cos,sin  = egocentric goal bearing, BODY frame (x=FORWARD, y=LEFT).
 *   yawrate  = own gyro-z [rad/s] ;  v = own horizontal speed [m/s] (EKF).
 *
 * FLOW-DECK LIMIT: state estimation is the Flow deck v2, whose optical flow
 * saturates around v/h = 7 rad/s. v_cmd is therefore CLAMPED to n6.maxSpeed
 * (conservative default 0.6 m/s; n6_mission.py sets the flight value).
 *
 * CONTROL SHAPING (deploy/contracts.md):
 *   - yaw-rate setpoint passed through a first-order filter, tau = n6.yawTau
 *     (0.08 s) so the airframe matches the trained response.
 *   - speed setpoint slew-limited to n6.speedSlew (3 m/s^2).
 *
 * SAFETY:
 *   - Commands the drone ONLY when n6.enable=1 AND already flying (rising-edge
 *     arm). It never takes off by itself. n6.linkOn=1 runs the link only (bench
 *     test, no control). With both 0 the app loop is DORMANT (UART2 is opened
 *     at boot with RX interrupts only, but never read or written) so a
 *     flashed-but-idle drone flies normally under cfclient.
 *   - Link watchdog: no valid frame for n6.linkTimeoutMs -> yaw-rate and speed
 *     ramp to zero (hover in place) instead of flying blind.
 *
 * BUILD (brushless - DEFCONFIG, never `make PLATFORM=cf21bl`):
 *   make cf21bl_defconfig && make -j && make cload
 */

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

#include "stm32fxxx.h"
#include "deck.h"
#include "uart2.h"
#include "commander.h"
#include "log.h"
#include "param.h"
#include "num.h"
#include "supervisor.h"

// FLOW-DECK COEXISTENCE (found the hard way). UART2 activity corrupts
// the flow deck's PMW3901 SPI reads two ways, both avoided here:
//   1) uart2Init RACING the running flow task -> init it EARLY, in the forced
//      deck driver (.init below) at deckInit, before the flow task hammers SPI.
//      (App-level lazy init ran too late and killed the flow ~2/3 of boots.)
//   2) interrupt TX (uart2SendData / TXE ISR) -> use POLLED busy-wait TX below.
// Interrupt RX is safe (the whisker deck streams thousands/sec with flow fine).
// Needs CONFIG_DECK_FORCE="n6uart" in app-config to load the driver.

#define DEBUG_MODULE "N6AVOID"
#include "debug.h"

#define N6_BAUD          115200
#define CONTROL_HZ       50               // setpoint / loop rate
#define DT               (1.0f / CONTROL_HZ)   // 0.02 s
#define UPLINK_DIV       1                // uplink every loop -> 50 Hz (contract >=33)
#define RAD2DEG          57.29578f

// N6 action / obs scaling - MUST match the trained policy (deploy/contracts.md).
#define N6_YAWRATE_SCALE 1.0f             // rad/s that maps a0 (and vector[2]) to 1
#define N6_SPEED_MIN     0.0f             // a1=-1 -> 0.0 m/s
#define N6_SPEED_MAX     3.5f             // a1=+1 -> 3.5 m/s (also maps vector[3] to 1)

// ---- tunable params (live-editable from cfclient/cflib) ----
static uint8_t enable    = 0;     // 1 = policy controls the drone (must be flying)
static uint8_t linkOn    = 0;     // 1 = run UART link only (bench test, no control)
static float   cruiseAlt = 0.3f;  // m, fixed hold altitude
static uint8_t posMode   = 0;     // 0 = body-velocity cruise; 1 = position "carrot" ahead
static float   lookahead = 0.6f;  // m, carrot distance ahead of current pos (posMode=1)
static float   maxYaw    = 120.0f;// deg/s clamp on the commanded yaw rate
static float   yawScale  = N6_YAWRATE_SCALE; // rad/s that a0=1 maps to, AND the obs[2]
                                  // norm. Must match the policy (contract: 1.0 rad/s).
static float   yawGain   = 1.0f;  // scale on the N6 yaw command (0.5 = half the yaw)
static float   maxSpeed  = 0.6f;  // m/s FLOW-DECK cap on v_cmd (contract band is 0..3.5)
static float   vmin      = N6_SPEED_MIN; // a1=-1 -> vmin  (a1->speed map; set from n6_mission)
static float   vmax      = N6_SPEED_MAX; // a1=+1 -> vmax, AND the obs[3] norm
static float   yawTau    = 0.08f; // s, first-order yaw-rate filter time constant
static float   speedSlew = 3.0f;  // m/s^2 speed slew-rate limit
static uint16_t linkTimeoutMs = 700; // no valid frame this long -> hover
static float   goalDist  = 10.0f; // m - goal point is this far straight ahead of start
static uint8_t useGoalXY = 0;     // 0 = goalDist ahead of heading; 1 = use goalX/goalY below
static float   goalX     = 10.0f; // explicit goal point in the TAKEOFF frame (x=fwd) if useGoalXY
static float   goalY     = 0.0f;  // explicit goal point y (left)
static float   testGoal  = 0.0f;  // BENCH UPLINK TEST: >0 -> uplink a KNOWN bearing that
                                  // sweeps the full circle at this many Hz (ignores odometry,
                                  // no flight needed). 0 = normal. e.g. 0.1 = 1 turn / 10 s.

// ---- telemetry (log) ----
static uint32_t rawRx    = 0;     // TOTAL bytes seen on UART2 RX
static uint8_t  lastByte = 0;     // most recent raw byte received
static uint32_t rxFrames = 0;     // valid downlink frames (header+xor OK)
static float    a0raw    = 0.0f;  // N6 action[0] (yaw_rate/yawScale)
static float    a1raw    = 0.0f;  // N6 action[1] (speed_cmd)
static float    yawRateCmd = 0.0f;// filtered yaw-rate command [deg/s]
static float    speedCmd = 0.0f;  // slewed forward-speed command [m/s]
static float    hSpeed   = 0.0f;  // own horizontal speed [m/s] (uplinked)
static float    goalx    = 1.0f;  // uplink goal bearing cos (body)
static float    goaly    = 0.0f;  // uplink goal bearing sin (body)
static uint8_t  linkOk   = 0;     // downlink fresh within timeout
static float    goalRem  = 0.0f;  // metres remaining to the goal point

// ---- state read via the log API ----
static logVarId_t idYaw, idX, idY, idVx, idVy, idGyroZ;

// ---- downlink parser state machine ----
static uint8_t  dl[7];
static uint8_t  dlIdx = 0;
static float    a0 = 0.0f, a1 = -1.0f;        // latest action (a1=-1 -> min speed)
static uint32_t lastFrameMs = 0;

// ---- mission state ----
static bool    wasEnabled = false;
static float   yaw0deg = 0.0f;  // heading captured at hand-over
static float   startX = 0.0f, startY = 0.0f;  // position captured at hand-over
static float   goalWX = 0.0f, goalWY = 0.0f;  // FIXED goal point in the world
static uint8_t prevEnable = 0;  // for rising-edge arming
static bool    armed = false;   // took control via a 0->1 enable while flying

static int16_t clamp16(float scaled)
{
  if (scaled >  32767.0f) return  32767;
  if (scaled < -32768.0f) return -32768;
  return (int16_t)scaled;
}

// Feed one received byte into the 7-byte downlink (ACTION) parser.
static void dlByte(uint8_t b)
{
  if (dlIdx == 0) { if (b == 0xAA) dl[dlIdx++] = b; return; }
  if (dlIdx == 1) { if (b == 0x55) dl[dlIdx++] = b; else dlIdx = 0; return; }
  dl[dlIdx++] = b;
  if (dlIdx < 7) return;
  dlIdx = 0;

  uint8_t xr = dl[2] ^ dl[3] ^ dl[4] ^ dl[5];
  if (xr != dl[6]) return;                       // checksum fail -> drop
  int16_t r0 = (int16_t)(dl[2] | (dl[3] << 8));
  int16_t r1 = (int16_t)(dl[4] | (dl[5] << 8));
  a0 = r0 / 10000.0f;                            // yaw_rate_cmd / yawScale
  a1 = r1 / 10000.0f;                            // speed_cmd in [-1,1]
  a0raw = a0; a1raw = a1;
  rxFrames++;
  lastFrameMs = T2M(xTaskGetTickCount());
}

// UART2 is brought up EARLY (deckInit, before the flow task starts reading) via
// this forced deck driver - NOT lazily in the app - so uart2Init cannot race an
// active flow-deck SPI read.
static void n6UartDeckInit(DeckInfo *info) { (void)info; uart2Init(N6_BAUD); }
static bool n6UartDeckTest(void) { return true; }
static const DeckDriver n6UartDeck = {
  .name = "n6uart",
  .usedPeriph = DECK_USING_UART2,
  .init = n6UartDeckInit,
  .test = n6UartDeckTest,
};
DECK_DRIVER(n6UartDeck);

// Polled (interrupt-free) TX. The TXE interrupt corrupts the flow deck; this
// busy-waits instead (~1 ms of app-task spin per 11-byte frame at 115200), with
// a spin-count timeout so a wiring fault can't hang the loop forever.
static void usart2PolledTx(const uint8_t *d, uint32_t n)
{
  for (uint32_t i = 0; i < n; i++) {
    uint32_t guard = 100000;
    while (!(USART2->SR & USART_FLAG_TXE) && --guard) { }
    USART2->DR = d[i] & 0x00FF;
  }
}

// Uplink the policy OBSERVATION VECTOR[4] (header 0xCF 0x55, 8 payload bytes).
static void sendUplink(float c, float s, float yawNorm, float vNorm)
{
  uint8_t p[8];
  int16_t v0 = clamp16(c       * 10000.0f);
  int16_t v1 = clamp16(s       * 10000.0f);
  int16_t v2 = clamp16(yawNorm * 10000.0f);
  int16_t v3 = clamp16(vNorm   * 10000.0f);
  p[0] = v0 & 0xFF; p[1] = (v0 >> 8) & 0xFF;
  p[2] = v1 & 0xFF; p[3] = (v1 >> 8) & 0xFF;
  p[4] = v2 & 0xFF; p[5] = (v2 >> 8) & 0xFF;
  p[6] = v3 & 0xFF; p[7] = (v3 >> 8) & 0xFF;
  uint8_t xr = p[0]^p[1]^p[2]^p[3]^p[4]^p[5]^p[6]^p[7];
  uint8_t frame[11] = {0xCF, 0x55, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7], xr};
  usart2PolledTx(frame, 11);   // polled, not interrupt - keeps the flow deck alive
}

void appMain(void)
{
  DEBUG_PRINT("N6 app loaded (DORMANT). It touches nothing until asked:\n"
              "  n6.linkOn=1 -> run the UART link only (bench test, no control)\n"
              "  n6.enable=1 -> hand over flight control (while already flying)\n");

  // Pure TOC lookups, no hardware touched.
  idYaw   = logGetVarId("stateEstimate", "yaw");
  idX     = logGetVarId("stateEstimate", "x");
  idY     = logGetVarId("stateEstimate", "y");
  idVx    = logGetVarId("stateEstimate", "vx");
  idVy    = logGetVarId("stateEstimate", "vy");
  idGyroZ = logGetVarId("gyro", "z");

  // UART2 is inited at boot by the n6uart deck driver (see top of file) - the
  // app never calls uart2Init, so it can't race the flow-deck SPI reads.

  TickType_t next = xTaskGetTickCount();
  const TickType_t period = M2T(1000 / CONTROL_HZ);
  uint32_t loop = 0;

  while (1) {
    vTaskDelayUntil(&next, period);
    loop++;

    // 0) DORMANT unless the link or flight control is requested.
    if (!enable && !linkOn) {
      wasEnabled = false;
      armed      = false;
      prevEnable = 0;
      linkOk     = 0;
      yawRateCmd = 0.0f;
      speedCmd   = 0.0f;
      continue;
    }

    // UART2 was already brought up at boot by the n6uart deck driver (early, so
    // it can't corrupt the flow deck). Nothing to init here.

    // 1) Drain UART and parse downlink ACTION frames (interrupt RX - flow-safe).
    uint8_t b;
    int guard = 0;
    while (guard++ < 128 && uart2GetDataWithTimeout(1, &b, M2T(2)) > 0) {
      rawRx++;
      lastByte = b;
      dlByte(b);
    }

    // 2) Read state estimate.
    float yaw = logGetFloat(idYaw);        // deg, world
    float gz  = logGetFloat(idGyroZ);      // deg/s, body yaw rate
    float x   = logGetFloat(idX);
    float y   = logGetFloat(idY);
    float vx  = logGetFloat(idVx);         // m/s, world
    float vy  = logGetFloat(idVy);
    hSpeed = sqrtf(vx * vx + vy * vy);     // horizontal speed magnitude

    uint32_t nowMs = T2M(xTaskGetTickCount());
    linkOk = (nowMs - lastFrameMs) < linkTimeoutMs;

    // 3) Egocentric bearing (cos,sin) to the FIXED goal POINT, body frame.
    if (testGoal > 0.0001f) {
      // BENCH UPLINK TEST: ignore odometry; uplink a KNOWN bearing sweeping the
      // full circle at `testGoal` Hz. The N6 should see (goalx,goaly) trace
      // cos/sin - unambiguous proof the uplink arrives and frames parse. No
      // flight needed (works under linkOn=1, motors off).
      float ang = 2.0f * 3.14159265f * testGoal * (nowMs * 0.001f);
      goalx = cosf(ang); goaly = sinf(ang); goalRem = goalDist;
    } else if (wasEnabled) {
      float dx = goalWX - x, dy = goalWY - y;
      goalRem = sqrtf(dx * dx + dy * dy);
      float cy = cosf(yaw / RAD2DEG), sy = sinf(yaw / RAD2DEG);
      float bx =  cy * dx + sy * dy;       // forward
      float by = -sy * dx + cy * dy;       // left
      if (goalRem > 0.01f) { goalx = bx / goalRem; goaly = by / goalRem; }
      else                 { goalx = 1.0f;         goaly = 0.0f; }
    } else {
      goalx = 1.0f; goaly = 0.0f; goalRem = goalDist;
    }

    // 4) Uplink the observation vector[4] (>=33 Hz). yaw-rate & speed normalised.
    if ((loop % UPLINK_DIV) == 0) {
      float gzNorm = (gz / RAD2DEG) / yawScale;           // obs[2] = yaw_rate / yawScale
      float vNorm  = (vmax > 0.01f) ? (hSpeed / vmax) : 0.0f;  // EKF speed / vmax (obs slot 3)
      sendUplink(goalx, goaly, gzNorm, vNorm);
    }

    // 5) Turn the N6 ACTION into yaw-rate + speed targets (or zero on link loss).
    float yawTarget, speedTarget;
    if (linkOk) {
      yawTarget = (a0 * yawScale) * RAD2DEG * yawGain;         // a0 * yawScale rad/s -> deg/s
      if (yawTarget >  maxYaw) yawTarget =  maxYaw;
      if (yawTarget < -maxYaw) yawTarget = -maxYaw;
      float a1c = a1;                                     // clip(a1,-1,1)
      if (a1c >  1.0f) a1c =  1.0f;
      if (a1c < -1.0f) a1c = -1.0f;
      speedTarget = vmin + (a1c + 1.0f) * 0.5f * (vmax - vmin);
      if (speedTarget > maxSpeed) speedTarget = maxSpeed; // FLOW-DECK safety cap
      if (speedTarget < 0.0f)     speedTarget = 0.0f;
    } else {
      yawTarget = 0.0f;
      speedTarget = 0.0f;
    }

    // First-order yaw-rate filter (tau) + speed slew-rate limit (contract shaping).
    float alpha = (yawTau > DT) ? (DT / yawTau) : 1.0f;
    yawRateCmd += (yawTarget - yawRateCmd) * alpha;
    float dv = speedTarget - speedCmd, dvMax = speedSlew * DT;
    if (dv >  dvMax) dv =  dvMax;
    if (dv < -dvMax) dv = -dvMax;
    speedCmd += dv;

    // 6) Flight control - only after enable goes 0->1 WHILE flying.
    bool flying = supervisorIsFlying();
    if (!flying) armed = false;
    if (flying && enable && !prevEnable) armed = true;
    prevEnable = enable;

    if (enable && flying && armed) {
      if (!wasEnabled) {
        // Hand-over: fix the goal POINT (world/takeoff frame).
        yaw0deg = yaw;
        startX  = x;  startY = y;
        if (useGoalXY) {
          goalWX = goalX;  goalWY = goalY;                  // explicit point set by n6_mission
        } else {
          goalWX = x + goalDist * cosf(yaw / RAD2DEG);      // goalDist straight ahead of heading
          goalWY = y + goalDist * sinf(yaw / RAD2DEG);
        }
        wasEnabled = true;
        yawRateCmd = 0.0f; speedCmd = 0.0f;   // start from rest
        DEBUG_PRINT("N6 ENABLED: goal=(%.2f,%.2f) maxSpeed=%.2f\n",
                    (double)goalWX, (double)goalWY, (double)maxSpeed);
      }

      setpoint_t sp;
      memset(&sp, 0, sizeof(sp));
      sp.mode.z   = modeAbs;       sp.position.z = cruiseAlt;   // altitude hold
      sp.mode.yaw = modeVelocity;  sp.attitudeRate.yaw = yawRateCmd;  // direct yaw-rate

      if (posMode) {
        // Position "carrot" ahead along the nose; cruise speed is capped by
        // posCtlPid.xVelMax/yVelMax (set from the mission script). Note: posMode
        // does NOT track the policy's speed command precisely - velocity mode
        // (posMode=0) is the contract-faithful path.
        float c = cosf(yaw / RAD2DEG), s = sinf(yaw / RAD2DEG);
        float d = (speedCmd > 0.01f) ? lookahead : 0.0f;
        sp.mode.x = modeAbs;         sp.mode.y = modeAbs;
        sp.position.x = x + d * c;
        sp.position.y = y + d * s;
      } else {
        // Velocity: body-forward at the (filtered/slewed) commanded speed.
        sp.mode.x = modeVelocity;    sp.mode.y = modeVelocity;
        sp.velocity_body = true;
        sp.velocity.x = speedCmd;
        sp.velocity.y = 0.0f;
      }
      commanderSetSetpoint(&sp, COMMANDER_PRIORITY_CRTP);
    } else {
      wasEnabled = false;
    }
  }
}

PARAM_GROUP_START(n6)
PARAM_ADD(PARAM_UINT8,  enable,   &enable)
PARAM_ADD(PARAM_UINT8,  linkOn,   &linkOn)
PARAM_ADD(PARAM_FLOAT,  cruiseAlt,&cruiseAlt)
PARAM_ADD(PARAM_UINT8,  posMode,  &posMode)
PARAM_ADD(PARAM_FLOAT,  lookahead,&lookahead)
PARAM_ADD(PARAM_FLOAT,  maxYaw,   &maxYaw)
PARAM_ADD(PARAM_FLOAT,  yawScale, &yawScale)
PARAM_ADD(PARAM_FLOAT,  yawGain,  &yawGain)
PARAM_ADD(PARAM_FLOAT,  maxSpeed, &maxSpeed)
PARAM_ADD(PARAM_FLOAT,  vmin,     &vmin)
PARAM_ADD(PARAM_FLOAT,  vmax,     &vmax)
PARAM_ADD(PARAM_FLOAT,  yawTau,   &yawTau)
PARAM_ADD(PARAM_FLOAT,  testGoal, &testGoal)
PARAM_ADD(PARAM_FLOAT,  speedSlew,&speedSlew)
PARAM_ADD(PARAM_UINT16, linkTimeoutMs, &linkTimeoutMs)
PARAM_ADD(PARAM_FLOAT,  goalDist, &goalDist)
PARAM_ADD(PARAM_UINT8,  useGoalXY,&useGoalXY)
PARAM_ADD(PARAM_FLOAT,  goalX,    &goalX)
PARAM_ADD(PARAM_FLOAT,  goalY,    &goalY)
PARAM_GROUP_STOP(n6)

LOG_GROUP_START(n6)
LOG_ADD(LOG_UINT32, rawRx,     &rawRx)
LOG_ADD(LOG_UINT8,  lastByte,  &lastByte)
LOG_ADD(LOG_UINT32, rxFrames,  &rxFrames)
LOG_ADD(LOG_UINT8,  linkOk,    &linkOk)
LOG_ADD(LOG_FLOAT,  a0raw,     &a0raw)
LOG_ADD(LOG_FLOAT,  a1raw,     &a1raw)
LOG_ADD(LOG_FLOAT,  yawRateCmd,&yawRateCmd)
LOG_ADD(LOG_FLOAT,  speedCmd,  &speedCmd)
LOG_ADD(LOG_FLOAT,  hSpeed,    &hSpeed)
LOG_ADD(LOG_FLOAT,  goalx,     &goalx)
LOG_ADD(LOG_FLOAT,  goaly,     &goaly)
LOG_ADD(LOG_FLOAT,  goalRem,   &goalRem)
LOG_GROUP_STOP(n6)
