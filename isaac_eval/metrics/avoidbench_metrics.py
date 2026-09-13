"""AvoidBench metrics (Yu et al., arXiv 2301.07430) computed from our Isaac
sqroom eval episode logs (ep*.npz with traj/status/obs_xy/obs_r).
SR (eq.3), PO (eq.4), AGV (eq.6), MP (eq.7), traversability (Sec II env metric:
mean free-flight distance from random poses / drone diameter; normalized P_tau
= TRAV/TRAV_max with TRAV_max from the empty map)."""
import numpy as np, glob, heapq, sys, math

import os as _os
_ROOM = float(_os.environ.get('ROOM', 22.0))
GOAL = np.array([_ROOM, 0.0]); GOAL_R = 1.0; HALF_W = _ROOM / 2.0; X0 = 0.0
TICK = 0.01              # traj appended every STEER_DT kinematic tick
CLEAR = 0.2              # collision clearance = effective drone radius
DIA = 2 * CLEAR          # drone diameter for traversability normalization
RES = 0.10               # A* grid resolution

def astar_dmin(start, xy, r):
    """shortest free path length start->GOAL on inflated occupancy grid"""
    nx = int(round((_ROOM + 2.0) / RES)); ny = int(round(2 * HALF_W / RES))
    gx = lambda x: int(round((x + 1.0) / RES)); gy = lambda y: int(round((y + HALF_W) / RES))
    occ = np.zeros((nx, ny), bool)
    xs = (np.arange(nx) * RES) - 1.0; ys = (np.arange(ny) * RES) - HALF_W
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for (ox, oy), orad in zip(xy, r):
        occ |= (X - ox) ** 2 + (Y - oy) ** 2 < (orad + CLEAR) ** 2
    occ[:, :2] = occ[:, -2:] = True                     # walls
    s = (gx(start[0]), gy(start[1])); g = (gx(GOAL[0]), gy(GOAL[1]))
    if occ[s]: occ[s] = False
    if occ[g]: occ[g] = False
    h = lambda n: math.hypot(n[0] - g[0], n[1] - g[1])
    pq = [(h(s), 0.0, s)]; seen = {s: 0.0}
    NB = [(1,0,1),(  -1,0,1),(0,1,1),(0,-1,1),(1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
    while pq:
        f, c, n = heapq.heappop(pq)
        if math.hypot(n[0]-g[0], n[1]-g[1]) * RES < GOAL_R: return c
        if c > seen.get(n, 1e18) + 1e-9: continue
        for dx, dy, w in NB:
            m = (n[0]+dx, n[1]+dy)
            if not (0 <= m[0] < nx and 0 <= m[1] < ny) or occ[m]: continue
            nc = c + w * RES if not occ[m] else None
            if nc < seen.get(m, 1e18): seen[m] = nc; heapq.heappush(pq, (nc + h(m)*RES*0.999, nc, m))
    return None

def free_flight(p, th, xy, r, cap=30.0):
    """distance until collision along heading th from p (analytic ray vs circles+walls)"""
    d = np.array([math.cos(th), math.sin(th)]); best = cap
    for (ox, oy), orad in zip(xy, r):                    # ray-circle
        oc = np.array([ox, oy]) - p; proj = oc @ d
        if proj <= 0: continue
        perp2 = oc @ oc - proj ** 2; R = orad + CLEAR
        if perp2 < R * R: best = min(best, proj - math.sqrt(R * R - perp2))
    for sign in (1, -1):                                 # side walls
        if d[1] * sign > 1e-9:
            best = min(best, (sign * (HALF_W - CLEAR) - p[1]) / d[1] * sign if sign > 0 else (-(HALF_W - CLEAR) - p[1]) / d[1])
    if d[0] > 1e-9:  best = min(best, (_ROOM - p[0]) / d[0])
    if d[0] < -1e-9: best = min(best, (X0 - p[0]) / d[0])
    return max(best, 0.0)

def traversability(xy, r, n=1000, seed=0):
    rng = np.random.RandomState(seed); tot = 0.0; k = 0
    while k < n:
        p = rng.uniform([X0, -HALF_W + 1], [_ROOM, HALF_W - 1])
        if len(xy) and (np.linalg.norm(p - xy, axis=-1) - r < CLEAR).any(): continue
        tot += free_flight(p, rng.uniform(-math.pi, math.pi), xy, r); k += 1
    return tot / n / DIA

rows = []
for f in sorted(glob.glob(sys.argv[1] + '/ep*.npz')):
    d = np.load(f, allow_pickle=True)
    traj = d['traj'][:, :2]; status = str(d['status'])
    xy, r = d['obs_xy'], d['obs_r']
    dmin = astar_dmin(traj[0], xy, r)
    t_trial = (len(traj) - 1) * TICK
    d_trav = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    a = GOAL - traj[0]; b = traj[-1] - traj[0]; c = a - b
    mp = (1 - abs(a @ c) / (a @ a)) * 100
    trav = traversability(xy, r, seed=hash(f) % 2**31)
    rows.append(dict(ep=f[-7:-4], status=status, dmin=dmin, t=t_trial,
                     dtrav=d_trav, mp=min(mp, 100.0), trav=trav,
                     po=(d_trav - dmin) / dmin * 100 if status == 'reached' and dmin else None,
                     agv=dmin / t_trial if status == 'reached' and dmin else None))

trav_max = 22.0 * 0  # placeholder; empty-map TRAV computed below
trav_empty = traversability(np.zeros((0, 2)), np.zeros(0), seed=1)
print(f"{'ep':>4} {'status':>9} {'dmin':>6} {'d_trav':>7} {'PO%':>7} {'AGV':>6} {'MP%':>6} {'trav':>6} {'P_tau':>6}")
for w in rows:
    print(f"{w['ep']:>4} {w['status']:>9} {w['dmin']:6.2f} {w['dtrav']:7.2f} "
          f"{w['po']:7.1f}" if w['po'] is not None else
          f"{w['ep']:>4} {w['status']:>9} {w['dmin']:6.2f} {w['dtrav']:7.2f} {'--':>7}", end='')
    print(f" {w['agv']:6.2f}" if w['agv'] else f" {'--':>6}", end='')
    print(f" {w['mp']:6.1f} {w['trav']:6.2f} {w['trav']/trav_empty:6.3f}")
n = len(rows); nr = sum(1 for w in rows if w['status'] == 'reached')
agvs = [w['agv'] for w in rows if w['agv']]; pos = [w['po'] for w in rows if w['po'] is not None]
mps = [w['mp'] for w in rows]
print(f"\nSR {nr}/{n} = {nr/n*100:.0f}%  |  AGV mean {np.mean(agvs):.2f} m/s  |  "
      f"PO mean {np.mean(pos):.1f}%  |  MP mean {np.mean(mps):.1f}%  |  "
      f"trav mean {np.mean([w['trav'] for w in rows]):.2f} (P_tau {np.mean([w['trav'] for w in rows])/trav_empty:.3f}, empty {trav_empty:.2f})")
