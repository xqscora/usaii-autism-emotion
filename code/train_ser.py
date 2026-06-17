"""Train a speech-emotion classifier on RAVDESS (from scratch, CPU-friendly).

Features: librosa acoustic features (MFCC + chroma + mel + spectral contrast + ZCR),
mean/std-pooled over time -> one vector per clip.
Classifier: RandomForest (sklearn).
Split: SPEAKER-INDEPENDENT (actors 1-20 train, 21-24 test) so we measure real
generalization to unseen voices — the honest number, not an inflated one.

This is the baseline emotion model. The autism-specific work (adapting to autistic
voices) builds on top; see README.
"""
import os
import glob
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "datasets", "RAVDESS")


def extract_features(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, duration=3.0, offset=0.5)
    if y.size < sr // 2:
        y = np.pad(y, (0, sr // 2))
    parts = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    parts += [mfcc.mean(axis=1), mfcc.std(axis=1)]
    parts.append(librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1))
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    parts.append(librosa.power_to_db(mel).mean(axis=1))
    parts.append(librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1))
    parts.append(np.array([librosa.feature.zero_crossing_rate(y).mean()]))
    return np.concatenate(parts).astype(np.float32)


def main():
    files = sorted(glob.glob(os.path.join(DATA, "Actor_*", "*.wav")))
    print(f"found {len(files)} wav files")
    X, y, actors = [], [], []
    for i, f in enumerate(files):
        p = os.path.basename(f).split("-")
        try:
            emo = int(p[2]); actor = int(p[6].split(".")[0])
            X.append(extract_features(f)); y.append(emo); actors.append(actor)
        except Exception as e:
            print("skip", os.path.basename(f), e)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} features extracted")
    X, y, actors = np.array(X), np.array(y), np.array(actors)
    print("feature matrix:", X.shape)

    tr, te = actors <= 20, actors > 20      # speaker-independent
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)

    acc = accuracy_score(y[te], pred)
    print(f"\n=== SPEAKER-INDEPENDENT test accuracy: {acc:.3f}  (chance = {1/8:.3f}) ===\n")
    labels = sorted(EMOTIONS)
    print(classification_report(y[te], pred, labels=labels,
                                target_names=[EMOTIONS[i] for i in labels], zero_division=0))

    out = os.path.join(HERE, "..", "ravdess_ser_model.joblib")
    joblib.dump({"model": clf, "scaler": scaler, "emotions": EMOTIONS,
                 "feature_fn": "librosa mfcc40(mean,std)+chroma+mel+contrast+zcr"}, out)
    print("saved model ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
