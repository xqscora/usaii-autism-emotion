# -*- coding: utf-8 -*-
"""Generate the shareable Colab demo notebook (demo_colab.ipynb).
Run: python build_colab_nb.py
"""
import json, os

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)})

md("""# 🎙️ Autism Emotion Recognition — Live Demo
**USAII Global AI Hackathon 2026 · Cora & Mujahid**

Speak into your mic (or upload a clip) → the model reads the emotion in your voice.

> ⚠️ **Honest note.** This baseline uses a model trained on *neurotypical adult* speech
> (SUPERB / IEMOCAP, 4 emotions: neutral / happy / angry / sad). On autistic kids it will
> sometimes be **confidently wrong** — that's exactly the problem we're solving. So it
> (1) shows confidence and **abstains when unsure**; and the full project adds
> (2) continued self-supervised pretraining on **ASDSpeech** (so the backbone learns how
> autistic children actually sound), (3) **per-wearer calibration**, and (4) **face** as an
> occasional ground-truth check. Voice is the live channel; face only calibrates.

Run the cells top to bottom. First run downloads the model (~360 MB).""")

code("!pip -q install transformers librosa soundfile")

code("""import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL = "superb/wav2vec2-base-superb-er"   # 4 emotions: neutral / happy / angry / sad
fe = AutoFeatureExtractor.from_pretrained(MODEL)
model = AutoModelForAudioClassification.from_pretrained(MODEL).eval()
print("Loaded:", MODEL)
print("Emotions:", model.config.id2label)""")

code("""import torch

def predict_emotion(speech, sr=16000, conf_threshold=0.5):
    \"\"\"speech: 1-D float array @16kHz. Returns (label, confidence, all_probs).
    Abstains ('uncertain') when the top prob is below conf_threshold -- the guard
    against confidently-wrong reads on atypical voices.\"\"\"
    inputs = fe(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    id2label = model.config.id2label
    all_probs = {id2label[i]: round(float(p), 3) for i, p in enumerate(probs)}
    conf, idx = float(probs.max()), int(probs.argmax())
    label = id2label[idx] if conf >= conf_threshold else "uncertain (abstain)"
    return label, conf, all_probs""")

md("## 🎤 Option A — speak into your mic\nRun this, allow mic access, and talk for 4 seconds.")

code("""from IPython.display import Javascript, display
from google.colab import output
import base64, io, librosa

_RECORD_JS = '''
async function record(ms){
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  const stopped = new Promise(r => rec.onstop = r);
  rec.start();
  await new Promise(r => setTimeout(r, ms));
  rec.stop();
  await stopped;
  stream.getTracks().forEach(t => t.stop());
  const buf = await (new Blob(chunks)).arrayBuffer();
  const bytes = new Uint8Array(buf);
  let bin = '';
  for (let i=0; i<bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
'''

def record(sec=4):
    display(Javascript(_RECORD_JS))
    print(f"Recording {sec}s -- speak now!")
    b64 = output.eval_js(f'record({sec*1000})')
    audio_bytes = base64.b64decode(b64)
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    return speech

speech = record(4)
label, conf, all_probs = predict_emotion(speech)
print(f"\\n🎭 emotion: {label}   (confidence {conf:.2f})")
print("all probs:", all_probs)""")

md("## 📁 Option B — upload an audio file\nIf the mic cell is fussy, just upload a .wav/.mp3 of someone talking.")

code("""from google.colab import files
import librosa

up = files.upload()
path = list(up.keys())[0]
speech, _ = librosa.load(path, sr=16000)
label, conf, all_probs = predict_emotion(speech)
print(f"🎭 emotion: {label}   (confidence {conf:.2f})")
print("all probs:", all_probs)""")

md("""## 🔭 What's next (the actual project)
This baseline proves the pipeline runs. The real work is making it right for autistic kids:
1. **Continued pretraining on ASDSpeech** (197 autistic children's voices) so the backbone isn't a neurotypical-only model.
2. **Per-wearer calibration** — a few samples per child instead of one generic template.
3. **Uncertainty-aware abstention** (conformal prediction) so a confident wrong read never gets shown.
4. **Face as calibration** (FER-Autism), not a live camera.

Repo & plan: see the project README.""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "..", "demo_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.abspath(out), "with", len(cells), "cells")
