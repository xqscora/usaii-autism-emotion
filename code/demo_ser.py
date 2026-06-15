"""Working SER demo: speech in -> emotion out.

Uses a pretrained wav2vec2 SER model (SUPERB / IEMOCAP, 4 classes) as the BASELINE.
This is the honest starting point: it's trained on neurotypical speech, so on autistic
kids it will sometimes be confidently wrong -- which is exactly why we wrap it with an
abstention threshold (don't show a definitive emotion when unsure) and why the project
plan adds ASDSpeech continued-pretraining + per-wearer calibration on top.

Run:
    python demo_ser.py                 # runs on a built-in sample speech clip
    python demo_ser.py path/to.wav     # runs on your own audio
"""
import sys
import torch

MODEL_ID = "superb/wav2vec2-base-superb-er"   # 4 emotions: neu / hap / sad / ang
_CACHE = {}


def load_model():
    if "model" not in _CACHE:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        _CACHE["fe"] = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        _CACHE["model"] = AutoModelForAudioClassification.from_pretrained(MODEL_ID).eval()
    return _CACHE["fe"], _CACHE["model"]


def predict_emotion(wav, sr=16000, conf_threshold=0.5):
    """wav: file path or 1-D numpy array @16kHz. Returns (label, confidence, all_probs)."""
    import librosa
    import numpy as np
    fe, model = load_model()
    if isinstance(wav, str):
        speech, _ = librosa.load(wav, sr=sr)
    else:
        speech = np.asarray(wav, dtype="float32")
    inputs = fe(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    id2label = model.config.id2label
    all_probs = {id2label[i]: round(float(p), 3) for i, p in enumerate(probs)}
    conf, idx = float(probs.max()), int(probs.argmax())
    label = id2label[idx] if conf >= conf_threshold else "uncertain (abstain)"
    return label, conf, all_probs


if __name__ == "__main__":
    fe, model = load_model()
    print("Loaded:", MODEL_ID)
    print("Emotions:", model.config.id2label)
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        import librosa
        src = librosa.example("libri1")   # a real speech clip, auto-downloaded (small)
        print("No audio given -> using built-in sample:", src)
    label, conf, all_probs = predict_emotion(src)
    print("\n=== RESULT ===")
    print(f"emotion : {label}")
    print(f"confidence: {conf:.3f}")
    print(f"all probs : {all_probs}")
