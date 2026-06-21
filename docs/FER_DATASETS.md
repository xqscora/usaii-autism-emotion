# Face datasets — direct download guide

Priority for **USAII autism emotion** face track.

## Tier A — script auto-download (no Kaggle / no form)

| ID | What | Labels | Access |
|----|------|--------|--------|
| `fer2013_hf` | FER2013 enhanced (~35k faces, 48×48) | 7 emotions | Hugging Face `abhilash88/fer2013-enhanced` |
| `fer2013_raw_hf` | FER2013 classic parquet | 7 emotions | HF `TrainingDataPro/faces-emotions-dataset` (fallback) |

```powershell
python code/download_fer_datasets.py --dataset fer2013_hf
```

## Tier B — autism face emotion (best for our pitch, semi-manual)

| ID | What | Labels | Access |
|----|------|--------|--------|
| `fer_autism` | FER-Autism children faces | 6: natural, anger, fear, joy, sadness, surprise | [Mendeley b33pf78h62](https://data.mendeley.com/datasets/b33pf78h62) — click **Download All**, unzip to `datasets/FER-Autism/` |

```powershell
python code/download_fer_datasets.py --dataset fer_autism --zip path\to\download.zip
```

CC BY 4.0. This is the one README already flagged. **No public HF mirror with images found.**

## Tier C — useful but not emotion-labeled / not images

| ID | Notes |
|----|--------|
| `1saccc/ASD` on HF | OpenFace CSV + face mp4s, ASD vs NT, emotion **from video stimulus** — mixed format |
| Hugging Rain Man | AU labels only, **no public pixels** (privacy) |
| FADC (GitHub) | ASD vs TD detection, **not** 6-class emotion — contact authors for files |
| AffectNet / RAF-DB | Strong FER but **license form**, not instant |

## Tier D — skip for hackathon deadline

- JAFFE (Zenodo login, restricted)
- Kaggle FER2013 / FERAC (account + ToS)

## Our capture pipeline (per-person, long run)

```powershell
python code/face_capture.py --person-id child_01
# Ctrl+C to stop. Clips -> datasets/captured/child_01/
```

Colab: mount Drive, same script with `--root /content/drive/MyDrive/usaii-capture`

## Train

```powershell
python code/download_fer_datasets.py --dataset fer2013_hf
python code/train_fer.py --data datasets/fer2013_hf --epochs 8
python code/train_fer.py --data datasets/FER-Autism --epochs 12   # after manual zip
```
