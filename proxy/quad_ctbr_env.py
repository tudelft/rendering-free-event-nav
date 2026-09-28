"""Crazyflie Brushless environment (CTBR action space) — JAX, jittable, vmap-friendly.

SkyDreamer-for-Crazyflie-Brushless, milestone M1.

Dynamics are a pure-JAX port of the first-principle model from
    Graefe, Scherer, Hoenig, Trimpe, "How to Model Your Crazyflie Brushless",
    ICRA 2026  (arXiv 2603.05944), eqs (1)-(10), Table II.
Their reference sim is MJX-based (github.com/Data-Science-in-Mechanical-Engineering/
CrazyflieBrushJAX); we port the ODE directly because Mission 1 is contact-free
(gate/bar collisions are geometric terminations, not physical contacts), so we
don't need MuJoCo's contact solver. Cross-validate against their repo.

Action (CTBR, Swift-style):  a = [c, wx_sp, wy_sp, wz_sp] in [-1, 1]
    c            -> collective thrust in [0, F_max_total]
    wx,wy,wz_sp  -> body-rate setpoints in [-rate_max, rate_max]
A rate controller (P on body-rate error) + mixer converts CTBR -> per-motor
commands u in [0,1]^4, matching how the CF firmware's fast loop will execute
our setpoints at deploy time.  >>> RATE GAINS ARE PLACEHOLDERS: tune Kp_rate to
the Crazyflie firmware rate PID so sim ~ real. <<<

State (17-dim, per the paper): p(3, world), v(3, world), q(4, body->world,
    [w,x,y,z]), w(3, body angular rate), Om(4, motor speeds [rad/s]).

Frame: world Z-up, gravity = -g z_hat, thrust along body +z.

Reward mirrors SkyDreamer sec. II-C:  r = 5*r_prog - r_rate + 30*r_gate
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

# --------------------------------------------------- identified CFB params (Table II)
# Values below are the CFB *without* propeller guards (lighter). NOTE: adding the
# N6 + GenX320 payload will raise mass and inertia -> re-identify or widen DR.
SIGMA = 2.9e3            # motor-speed normalizer for the polynomials
A_THRUST = jnp.array([-0.043, 0.562, -0.23])   # eq 9: a1*x + a2*x^2 + a3*x^3  (x=Om/SIGMA)
A_TORQUE = jnp.array([2.9, 8.7, -3.4]) * 1e-4  # eq 10: reactive drag torque(x)


class Params(NamedTuple):
    dt: float = 1.0 / 90.0        # control period (SkyDreamer f_c = 90 Hz)
    substeps: int = 4             # inner RK4 substeps (motor dyn are fast; T=50ms though)
    g: float = 9.81
    mass: float = 0.040           # kg  (44g with guards)
    ell: float = 0.0707           # m, quad width
    Jx: float = 1.8e-5            # kgm^2 (with guards: 3.3e-5)
    Jy: float = 2.4e-5            # (3.6e-5)
    Jz: float = 3.0e-5            # (5.9e-5)
    T_motor: float = 0.05         # motor time constant [s]
    K_motor: float = 2900.0       # motor amplification [rad/s]; steady-state Om = K*u
    Jm: float = 0.5e-7            # motor moment of inertia
    rate_max: float = 6.0         # CTBR body-rate bound [rad/s]
    Kp_rate: float = 30.0         # ⚠️ rate-loop P gain [1/s] — tune to CF firmware
    # gate: center, inward normal (travel dir), half-extent (lateral, vertical)
    gate_center: tuple = (2.0, 0.0, 1.5)
    gate_normal: tuple = (1.0, 0.0, 0.0)
    gate_half: tuple = (0.25, 0.25)
    # progress goal sits THIS far beyond the gate along the normal, so "make
    # progress" pulls the drone THROUGH the gate (not hover at its center — a
    # target AT the center makes crossing = moving away = negative progress).
    # Mirrors SkyDreamer's progress-toward-the-next-gate signal.
    goal_beyond: float = 0.6
    tilt_limit: float = jnp.pi / 3


class State(NamedTuple):
    p: jax.Array      # (3,)
    v: jax.Array      # (3,)
    q: jax.Array      # (4,) [w,x,y,z]
    w: jax.Array      # (3,) body rates
    Om: jax.Array     # (4,) motor speeds [rad/s]
    prev_dist: jax.Array  # scalar, for r_prog


# --------------------------------------------------- quaternion utils
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_R(q):
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------- motor polynomials (eq 9, 10)
def _poly(coeffs, x):
    # coeffs = [c1, c2, c3] for c1*x + c2*x^2 + c3*x^3
    return coeffs[0] * x + coeffs[1] * x ** 2 + coeffs[2] * x ** 3


def _thrust(Om):
    return _poly(A_THRUST, Om / SIGMA)          # per-motor thrust [N], Om in rad/s


def _drag_torque(Om):
    return _poly(A_TORQUE, Om / SIGMA)          # per-motor reactive torque [N.m]


def _thrust_to_Om(f, pm: Params):
    """Invert the thrust polynomial (eq 9) for per-motor thrust f -> motor speed Om.
    3 Newton steps on x = Om/SIGMA over the hover-relevant increasing branch."""
    x = jnp.clip(jnp.sqrt(jnp.clip(f, 0.0, None) / 0.289), 0.0, 1.0)  # rough init
    for _ in range(3):
        F = _poly(A_THRUST, x) - f
        dF = A_THRUST[0] + 2 * A_THRUST[1] * x + 3 * A_THRUST[2] * x ** 2
        x = jnp.clip(x - F / (dF + 1e-6), 0.0, 1.0)
    return x * SIGMA


# --------------------------------------------------- CTBR -> motor commands u
def _mixer_inv(pm: Params):
    """Inverse of [F_total; tau_x; tau_y; tau_z] <- per-motor thrusts (X config)."""
    h = pm.ell / 2.0
    L0 = jnp.array([-h, -h, h, h])       # roll  (eq 8)
    L1 = jnp.array([-h, h, h, -h])       # pitch (eq 8)
    kappa = 2.84e-3                      # reactive-torque / thrust ratio (from eq9/10 @ x=1)
    s = jnp.array([-1.0, 1.0, -1.0, 1.0])
    M = jnp.stack([jnp.ones(4), L0, L1, kappa * s])   # [F_total; tx; ty; tz] <- f
    return jnp.linalg.inv(M)


def _ctbr_to_u(a, s: State, pm: Params, M_inv):
    a = jnp.clip(a, -1.0, 1.0)
    F_max_total = 4.0 * 0.289
    F_c = 0.5 * (a[0] + 1.0) * F_max_total          # collective thrust [N]
    w_sp = a[1:4] * pm.rate_max
    J = jnp.array([pm.Jx, pm.Jy, pm.Jz])
    tau_des = J * pm.Kp_rate * (w_sp - s.w)         # desired body torques (P law)
    f_des = M_inv @ jnp.concatenate([jnp.array([F_c]), tau_des])   # per-motor thrust
    Om_des = _thrust_to_Om(jnp.clip(f_des, 0.0, None), pm)
    return jnp.clip(Om_des / pm.K_motor, 0.0, 1.0)


# --------------------------------------------------- dynamics (eqs 1-5)
def _deriv(s: State, u, pm: Params):
    J = jnp.array([pm.Jx, pm.Jy, pm.Jz])
    R = _quat_to_R(s.q)
    f_i = _thrust(s.Om)                              # per-motor thrust
    F_body = jnp.array([0.0, 0.0, jnp.sum(f_i)])
    dv = R @ F_body / pm.mass - jnp.array([0.0, 0.0, pm.g])     # eq 2
    dp = s.v                                                    # eq 1
    dq = 0.5 * _quat_mul(s.q, jnp.concatenate([jnp.zeros(1), s.w]))  # eq 3
    # motor dynamics (eq 5): first-order toward K*u
    dOm = (-s.Om + pm.K_motor * u) / pm.T_motor
    # torques (eqs 4,7,8): roll/pitch from thrust arms, yaw from reactive drag + Jm*dOm
    h = pm.ell / 2.0
    L0 = jnp.array([-h, -h, h, h]); L1 = jnp.array([-h, h, h, -h])
    sgn = jnp.array([-1.0, 1.0, -1.0, 1.0])
    tau = jnp.array([
        jnp.dot(L0, f_i),
        jnp.dot(L1, f_i),
        jnp.dot(sgn, _drag_torque(s.Om) + pm.Jm * dOm),
    ])
    dw = (tau - jnp.cross(s.w, J * s.w)) / J                    # eq 4
    return dp, dv, dq, dw, dOm


def _rk4(s: State, u, pm: Params):
    h = pm.dt / pm.substeps

    def one(y):
        p, v, q, w, Om = y
        st = State(p, v, q, w, Om, s.prev_dist)
        return _deriv(st, u, pm)

    y = (s.p, s.v, s.q, s.w, s.Om)
    for _ in range(pm.substeps):
        k1 = one(y)
        y1 = tuple(a + 0.5 * h * b for a, b in zip(y, k1))
        k2 = one(y1)
        y2 = tuple(a + 0.5 * h * b for a, b in zip(y, k2))
        k3 = one(y2)
        y3 = tuple(a + h * b for a, b in zip(y, k3))
        k4 = one(y3)
        y = tuple(a + (h / 6.0) * (b1 + 2 * b2 + 2 * b3 + b4)
                  for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4))
    p, v, q, w, Om = y
    q = q / jnp.linalg.norm(q)
    return State(p=p, v=v, q=q, w=w, Om=Om, prev_dist=s.prev_dist)


# --------------------------------------------------- reward / done / reset / obs
def _goal(pm: Params):
    """Progress target: a point `goal_beyond` metres past the gate along its normal."""
    return jnp.array(pm.gate_center) + pm.goal_beyond * jnp.array(pm.gate_normal)


def _reward(s_prev: State, s: State, pm: Params):
    gate = jnp.array(pm.gate_center)
    goal = _goal(pm)
    dist = jnp.linalg.norm(s.p - goal)          # progress measured to the far goal
    r_prog = s_prev.prev_dist - dist
    r_rate = (jnp.exp(jnp.minimum(jnp.sum(jnp.abs(s.w)), 17.0)) - 1.0) \
        / (2.0 * (1.0 / pm.dt) * 1e5)                           # SkyDreamer rate penalty
    n = jnp.array(pm.gate_normal)
    sd_prev = jnp.dot(s_prev.p - gate, n)
    sd = jnp.dot(s.p - gate, n)
    off = jnp.abs(s.p - gate)
    lateral_ok = (off[1] < pm.gate_half[0]) & (off[2] < pm.gate_half[1])
    crossed = (sd_prev < 0) & (sd >= 0) & lateral_ok
    r_gate = jnp.where(crossed,
                       1.0 - jnp.maximum(off[1] / pm.gate_half[0], off[2] / pm.gate_half[1]),
                       0.0)
    return 5.0 * r_prog - r_rate + 30.0 * r_gate, dist, crossed


def _done(s: State, pm: Params):
    R = _quat_to_R(s.q)
    tilt = jnp.arccos(jnp.clip(R[2, 2], -1.0, 1.0))
    return (tilt > pm.tilt_limit) | (s.p[2] < -0.05)


def reset(key, pm: Params) -> State:
    p = jax.random.uniform(key, (3,), minval=-0.3, maxval=0.3) + jnp.array([0.0, 0.0, 1.0])
    # start at TRUE hover: invert the thrust poly for per-motor thrust = mg/4
    # (the old u=0.34 gave only ~165 mN vs mg=392 mN, so episodes began by dropping).
    Om_hover = _thrust_to_Om(pm.mass * pm.g / 4.0, pm) * jnp.ones(4)
    dist = jnp.linalg.norm(p - _goal(pm))       # prev_dist to the far goal
    return State(p=p, v=jnp.zeros(3), q=jnp.array([1.0, 0.0, 0.0, 0.0]),
                 w=jnp.zeros(3), Om=Om_hover, prev_dist=dist)


def observe(s: State, pm: Params) -> jax.Array:
    """M1 privileged/full-state obs (M2 swaps in the segmentation mask).
    Includes motor speeds, matching SkyDreamer's use of motor RPM."""
    gate = jnp.array(pm.gate_center)
    return jnp.concatenate([s.p, s.v, s.q, s.w, s.Om / pm.K_motor, gate - s.p])  # 20-dim


def step(key, s: State, a: jax.Array, pm: Params, M_inv):
    u = _ctbr_to_u(a, s, pm, M_inv)
    s_next = _rk4(s, u, pm)
    reward, dist, crossed = _reward(s, s_next, pm)
    s_next = s_next._replace(prev_dist=dist)
    return s_next, observe(s_next, pm), reward, _done(s_next, pm), {"crossed": crossed}


if __name__ == "__main__":
    # smoke test: vmapped random rollout. UNVALIDATED — run once JAX is on the cluster.
    pm = Params()
    M_inv = _mixer_inv(pm)
    B, T = 8, 200
    key = jax.random.PRNGKey(0)
    s = jax.vmap(reset, in_axes=(0, None))(jax.random.split(key, B), pm)

    @jax.jit
    def rollout(s, key):
        def body(carry, _):
            s, key = carry
            key, ka = jax.random.split(key)
            a = jax.random.uniform(ka, (B, 4), minval=-1.0, maxval=1.0)
            s, obs, r, d, info = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(key, B), s, a, pm, M_inv)
            return (s, key), r
        (s, _), rs = jax.lax.scan(body, (s, key), None, length=T)
        return rs

    rs = rollout(s, key)
    print("rollout ok:", rs.shape, "mean reward:", float(rs.mean()))
