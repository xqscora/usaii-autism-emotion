"""Stronger SER: train on wav2vec2 embeddings instead of hand-crafted librosa features.

Extract mean-pooled wav2vec2 hidden states per clip (768-d), train a linear classifier
on top. Same speaker-independent split (actors 1-20 train, 21-24 test) for an honest number.

CPU works but is slow (~20-30 min for 1440 clips); on Kaggle GPU it's a couple minutes.
"""
import os
import glob
import numpy as np
import torch
import librosa
import joblib
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "datasets", "RAVDESS")
BACKBONE = "superb/wav2vec2-base-superb-er"


def main():
    fe = AutoFeatureExtractor.from_pretrained(BACKBONE)
    model = AutoModel.from_pretrained(BACKBONE).eval()
    torch.set_num_threads(os.cpu_count() or 4)

    files = sorted(glob.glob(os.path.join(DATA, "Actor_*", "*.wav")))
    print(f"{len(files)} files; extracting wav2vec2 embeddings on CPU (slow)...")
    X, y, actors = [], [], []
    for i, f in enumerate(files):
        p = os.path.basename(f).split("-")
        emo, actor = int(p[2]), int(p[6].split(".")[0])
        speech, _ = librosa.load(f, sr=16000, duration=3.0, offset=0.5)
        inputs = fe(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state.mean(dim=1)[0].numpy()
        X.append(emb); y.append(emo); actors.append(actor)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    X, y, actors = np.array(X), np.array(y), np.array(actors)
    print("embeddings:", X.shape)

    tr, te = actors <= 20, actors > 20
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y[tr])
    pred = clf.predict(Xte)

    acc = accuracy_score(y[te], pred)
    print(f"\n=== wav2vec2 + LogReg  SPEAKER-INDEPENDENT accuracy: {acc:.3f}  (chance {1/8:.3f}) ===\n")
    labels = sorted(EMOTIONS)
    print(classification_report(y[te], pred, labels=labels,
                                target_names=[EMOTIONS[i] for i in labels], zero_division=0))

    out = os.path.join(HERE, "..", "ravdess_w2v_ser_model.joblib")
    joblib.dump({"model": clf, "scaler": scaler, "emotions": EMOTIONS, "backbone": BACKBONE}, out)
    # cache embeddings too, so we don't have to re-extract on CPU
    np.savez(os.path.join(HERE, "..", "ravdess_w2v_embeddings.npz"), X=X, y=y, actors=actors)
    print("saved model + cached embeddings")


if __name__ == "__main__":
    main()
