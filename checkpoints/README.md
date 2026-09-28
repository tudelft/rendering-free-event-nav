# checkpoints/

`ours_champion.pkl` is the policy reported in the paper: the analytic-proxy arm, trained for
40 M environment steps (34 min on one RTX 4090) and evaluated zero-shot everywhere else.

```python
import pickle
ck = pickle.load(open('checkpoints/ours_champion.pkl', 'rb'))
ck['task']    # 'yawzoneh64r7w20gxv3sq'  - rebuild the env with this
ck['meta']    # {'H': 64, 'C': 1, 'frame_stack': 3, 'adim': 2, 'vec_dim': 4, ...}
ck['actor']   # Flax params for training/nets.py:Actor
ck['critic']  # Flax params for training/nets.py:Critic
```

Reported scores for this exact file:

| evaluation | result |
|---|---|
| in-proxy, deterministic, training distribution | 91 % goal-reach |
| Isaac Sim + EVIS, shadow scene, n=30 | 83 % (25/30), 2.38 m/s AGV |
| Isaac Sim + EVIS, texture scene, n=30 | 70 % (21/30), 1.62 m/s AGV |
| real flights, three arenas | 82 % (5/5, 5/6, 4/6) |

Reproduce the first row with `python training/eval_proxy.py checkpoints/ours_champion.pkl 30`,
and the Isaac rows with `training/configs/isaac_eval.sh` (needs Isaac Sim 4.2 and the EVIS
submodule).
