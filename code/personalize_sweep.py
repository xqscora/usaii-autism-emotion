"""How many calibration samples does a child actually need? Sweep K with a FIXED eval set.

Core selling point: a handful of samples per emotion already lifts accuracy a lot.
Fixed eval = each speaker's samples after the first 3 per emotion; K (0..3) are taken from
the front as calibration, so every K is measured on the same held-out clips (fair).

Uses cached embeddings. Run: python personalize_sweep.py
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(__file__)
d = np.load(os.path.join(HERE, "..", "ravdess_w2v_embeddings.npz"))
X, y, actors = d["X"], d["y"], d["actors"]
TRAIN = actors <= 20
TEST = sorted(set(actors[actors > 20].tolist()))
HOLD = 3  # per emotion, reserved as fixed eval

sc0 = StandardScaler().fit(X[TRAIN])
clf0 = LogisticRegression(max_iter=3000).fit(sc0.transform(X[TRAIN]), y[TRAIN])

# precompute each test speaker's fixed eval indices and ordered calibration pool
pools = {}
for a in TEST:
    m = np.where(actors == a)[0]
    ya = y[m]
    rng = np.random.RandomState(0)
    cal_pool, ev = [], []
    for e in sorted(set(ya.tolist())):
        idx = m[ya == e]
        rng.shuffle(idx)
        ev += idx[HOLD:].tolist()       # fixed eval
        cal_pool.append(idx[:HOLD])     # up to 3 available to use as calibration
    pools[a] = (np.array(ev, dtype=int), cal_pool)

print("K (calib samples/emotion) | mean acc over", len(TEST), "unseen speakers")
print("-" * 52)
for K in [0, 1, 2, 3]:
    accs = []
    for a in TEST:
        ev, cal_pool = pools[a]
        if K == 0:
            acc = accuracy_score(y[ev], clf0.predict(sc0.transform(X[ev])))
        else:
            cal = np.concatenate([c[:K] for c in cal_pool])
            Xp = np.vstack([X[TRAIN], X[cal]])
            yp = np.concatenate([y[TRAIN], y[cal]])
            scp = StandardScaler().fit(Xp)
            clfp = LogisticRegression(max_iter=3000).fit(scp.transform(Xp), yp)
            acc = accuracy_score(y[ev], clfp.predict(scp.transform(X[ev])))
        accs.append(acc)
    bar = "#" * int(np.mean(accs) * 40)
    print(f"   K={K}  | {np.mean(accs):.3f}  {bar}")
