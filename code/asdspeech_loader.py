"""ASDSpeech feature loader.

Each recording .mat has key 'features': a (10, 1) object array where each cell is
a feature matrix of shape (n_vocalization_subset, 49) — 49 acoustic + conversational
features (pitch, jitter, formants, energy, voicing, spectral slope, vocalization
stats, ...). z-normalized per the original paper.

NOTE: ASDSpeech labels are ADOS severity scores (in data_*.xlsx), NOT emotion.
So this loader feeds the *self-supervised / domain-adaptation* stage of our pipeline
(teaching the backbone how autistic children sound), not the emotion-classification head.
"""
import os
import glob
import numpy as np
import scipy.io

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "ASDSpeech", "data")


def load_recording(mat_path):
    """Return a list of feature matrices (one per vocalization-subset) for one recording."""
    m = scipy.io.loadmat(mat_path)
    feats = m["features"]            # (10, 1) object array
    mats = []
    for cell in feats.ravel():
        arr = np.asarray(cell, dtype=np.float32)
        if arr.ndim == 2 and arr.size:
            mats.append(arr)
    return mats


def iter_recordings(data_dir=DATA_DIR):
    for p in sorted(glob.glob(os.path.join(data_dir, "*.mat"))):
        yield os.path.splitext(os.path.basename(p))[0], load_recording(p)


def summarize(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    print(f"recordings: {len(files)}")
    if not files:
        print("  (no .mat found — check datasets/ASDSpeech/data)")
        return
    mats = load_recording(files[0])
    print(f"recording[0] = {os.path.basename(files[0])} -> {len(mats)} feature matrices")
    shapes = [a.shape for a in mats]
    print(f"  matrix shapes: {shapes[:5]}{' ...' if len(shapes) > 5 else ''}")
    if mats:
        a = mats[0]
        print(f"  matrix[0]: shape={a.shape} dtype={a.dtype} "
              f"mean={a.mean():.3f} std={a.std():.3f} min={a.min():.2f} max={a.max():.2f}")
    # total vocalization-subsets across the corpus
    total = sum(len(load_recording(f)) for f in files)
    print(f"total feature matrices across corpus: {total}")


if __name__ == "__main__":
    summarize()
