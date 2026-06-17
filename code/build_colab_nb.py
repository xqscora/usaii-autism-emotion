# -*- coding: utf-8 -*-
"""Generate the shareable Colab demo notebook (demo_colab.ipynb).
Uses OUR trained model (wav2vec2 superb embeddings -> LogReg, 64% speaker-independent on RAVDESS).
Run: python build_colab_nb.py
"""
import json, os

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})

md("""# 🎙️ Autism Emotion Recognition — Live Demo
**USAII Global AI Hackathon 2026 · Cora & Mujahid**

Speak into your mic (or upload a clip) → **our trained model** reads the emotion in your voice.

This runs the model *we trained*: wav2vec2 (superb) embeddings → LogisticRegression, trained on
RAVDESS — **64% speaker-independent accuracy** (8 emotions, chance = 12.5%).

> ⚠️ **Honest note.** It's trained on *neurotypical adult* speech (RAVDESS). On autistic kids it will
> sometimes be **confidently wrong** — that's exactly the problem we're solving. So it **abstains when
> unsure**, and the full project adds autism-specific adaptation (ASDSpeech continued-pretraining +
> per-wearer calibration). Voice is the live channel; face only calibrates.

Run the cells top to bottom.""")

code("!pip -q install transformers librosa soundfile scikit-learn joblib")

code("""import torch, joblib, urllib.request
from transformers import AutoFeatureExtractor, AutoModel

# our trained classifier (69 KB) from the repo
urllib.request.urlretrieve(
    "https://github.com/xqscora/usaii-autism-emotion/raw/master/ravdess_w2v_ser_model.joblib",
    "model.joblib")
B = joblib.load("model.joblib")

BACKBONE = "superb/wav2vec2-base-superb-er"
fe = AutoFeatureExtractor.from_pretrained(BACKBONE)
bb = AutoModel.from_pretrained(BACKBONE).eval()
print("loaded. emotions:", B["emotions"])""")

code("""def predict_emotion(speech, sr=16000, conf_threshold=0.40):
    inp = fe(speech, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        emb = bb(**inp).last_hidden_state.mean(dim=1).numpy()
    emb = B["scaler"].transform(emb)
    probs = B["model"].predict_proba(emb)[0]
    classes = B["model"].classes_
    all_probs = {B["emotions"][int(c)]: round(float(p), 3) for c, p in zip(classes, probs)}
    conf = float(probs.max())
    top = B["emotions"][int(classes[int(probs.argmax())])]
    label = top if conf >= conf_threshold else "uncertain (abstain)"
    return label, conf, all_probs""")

md("## 🎤 Option A — speak into your mic\nRun this, allow mic access, talk for 4 seconds.")

code("""from IPython.display import Javascript, display
from google.colab import output
import base64, io, librosa

_REC = '''
async function record(ms){
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  const stopped = new Promise(r => rec.onstop = r);
  rec.start(); await new Promise(r => setTimeout(r, ms)); rec.stop(); await stopped;
  stream.getTracks().forEach(t => t.stop());
  const buf = await (new Blob(chunks)).arrayBuffer();
  const bytes = new Uint8Array(buf); let bin = '';
  for (let i=0;i<bytes.length;i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
'''
def record(sec=4):
    display(Javascript(_REC)); print(f"Recording {sec}s -- speak now!")
    b64 = output.eval_js(f'record({sec*1000})')
    speech, _ = librosa.load(io.BytesIO(base64.b64decode(b64)), sr=16000)
    return speech

speech = record(4)
label, conf, all_probs = predict_emotion(speech)
print(f"\\n🎭 emotion: {label}   (confidence {conf:.2f})")
print("all probs:", all_probs)""")

md("## 📁 Option B — upload an audio file\nIf the mic cell is fussy, upload a .wav/.mp3 of someone talking.")

code("""from google.colab import files
import librosa
up = files.upload()
speech, _ = librosa.load(list(up.keys())[0], sr=16000)
label, conf, all_probs = predict_emotion(speech)
print(f"🎭 emotion: {label}   (confidence {conf:.2f})")
print("all probs:", all_probs)""")

md("""## 🔭 What's next (the actual project)
This baseline (64% on neurotypical RAVDESS) proves the pipeline. The real work is making it right for autistic kids:
1. **Continued pretraining on ASDSpeech** (197 autistic children's voices) so the backbone isn't neurotypical-only.
2. **Per-wearer calibration** — a few samples per child, not one generic template.
3. **Uncertainty-aware abstention** so a confident wrong read never gets shown.
4. **Face as calibration** (FER-Autism), not a live camera.

Repo: https://github.com/xqscora/usaii-autism-emotion""")

nb = {"cells": cells,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}

out = os.path.join(os.path.dirname(__file__), "..", "demo_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.abspath(out), "with", len(cells), "cells")
