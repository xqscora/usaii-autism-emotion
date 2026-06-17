"""Inference with OUR trained 8-class emotion model.

Pipeline: speech -> wav2vec2 (superb) embedding (mean-pooled) -> StandardScaler -> LogisticRegression -> emotion.
Trained on RAVDESS, SPEAKER-INDEPENDENT test accuracy = 64% (vs 37% for the librosa+RF
baseline; chance = 12.5%). Abstains when not confident.

Run:
    python demo_emotion.py                # built-in sample clip
    python demo_emotion.py path/to.wav    # your own audio
"""
import os
import sys
import numpy as np
import torch
import librosa
import joblib

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "ravdess_w2v_ser_model.joblib")
BACKBONE = "superb/wav2vec2-base-superb-er"
_C = {}


def _load():
    if "clf" not in _C:
        from transformers import AutoFeatureExtractor, AutoModel
        b = joblib.load(MODEL_PATH)
        _C.update(clf=b["model"], scaler=b["scaler"], emo=b["emotions"])
        _C["fe"] = AutoFeatureExtractor.from_pretrained(BACKBONE)
        _C["bb"] = AutoModel.from_pretrained(BACKBONE).eval()
    return _C


def predict_emotion(wav, sr=16000, conf_threshold=0.40):
    """wav: file path or 1-D float array @16kHz. Returns (label, confidence, all_probs)."""
    c = _load()
    if isinstance(wav, str):
        speech, _ = librosa.load(wav, sr=sr, duration=3.0, offset=0.5)
    else:
        speech = np.asarray(wav, dtype="float32")
    inputs = c["fe"](speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        emb = c["bb"](**inputs).last_hidden_state.mean(dim=1).numpy()
    emb = c["scaler"].transform(emb)
    probs = c["clf"].predict_proba(emb)[0]
    classes = c["clf"].classes_
    all_probs = {c["emo"][int(cl)]: round(float(p), 3) for cl, p in zip(classes, probs)}
    conf = float(probs.max())
    top = c["emo"][int(classes[int(probs.argmax())])]
    label = top if conf >= conf_threshold else "uncertain (abstain)"
    return label, conf, all_probs


if __name__ == "__main__":
    _load()
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = librosa.example("libri1")
        print("no audio given -> using built-in sample clip")
    label, conf, all_probs = predict_emotion(src)
    print(f"\n=== RESULT (our trained 8-class model) ===")
    print(f"emotion   : {label}")
    print(f"confidence: {conf:.3f}")
    print(f"all probs : {all_probs}")
