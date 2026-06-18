"""Per-person calibration experiment — the minimal proof of the project's core idea:
'give the model a few samples of THIS person and it understands them better.'

This is the smallest version of the Cerome-per-child idea: instead of one generic model,
each person gets a model nudged toward how *they* express emotion. Run on RAVDESS (where we
have labels and multiple speakers); the same mechanism is what carries over to an autistic
child whose expression is atypical/blurry.

Uses cached wav2vec2 embeddings (no re-extraction). Run: python personalize_demo.py
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

HERE = os.path.dirname(__file__)
d = np.load(os.path.join(HERE, "..", "ravdess_w2v_embeddings.npz"))
X, y, actors = d["X"], d["y"], d["actors"]

TRAIN = actors <= 20                      # generic-model training speakers
TEST_ACTORS = sorted(set(actors[actors > 20].tolist()))
K = 3                                     # calibration samples per emotion per person
rng = np.random.RandomState(0)

# generic model (same for every test person): trained only on speakers 1-20
sc0 = StandardScaler().fit(X[TRAIN])
clf0 = LogisticRegression(max_iter=3000, C=1.0).fit(sc0.transform(X[TRAIN]), y[TRAIN])

base_accs, pers_accs = [], []
print(f"per-person calibration (K={K} samples/emotion):\n")
for a in TEST_ACTORS:
    m = actors == a
    Xa, ya = X[m], y[m]
    cal, ev = [], []
    for e in sorted(set(ya.tolist())):
        idx = np.where(ya == e)[0]
        rng.shuffle(idx)
        cal += idx[:K].tolist()
        ev += idx[K:].tolist()
    cal, ev = np.array(cal), np.array(ev)

    base = accuracy_score(ya[ev], clf0.predict(sc0.transform(Xa[ev])))

    Xp = np.vstack([X[TRAIN], Xa[cal]])
    yp = np.concatenate([y[TRAIN], ya[cal]])
    scp = StandardScaler().fit(Xp)
    clfp = LogisticRegression(max_iter=3000, C=1.0).fit(scp.transform(Xp), yp)
    pers = accuracy_score(ya[ev], clfp.predict(scp.transform(Xa[ev])))

    base_accs.append(base)
    pers_accs.append(pers)
    print(f"  speaker {a}: generic {base:.3f}  ->  personalized {pers:.3f}   ({pers-base:+.3f})")

b, p = float(np.mean(base_accs)), float(np.mean(pers_accs))
print(f"\n=== MEAN over {len(TEST_ACTORS)} unseen speakers ===")
print(f"  generic model        : {b:.3f}")
print(f"  + per-person calib   : {p:.3f}   ({p-b:+.3f})")
print(f"  (just {K} samples per emotion per person)")
