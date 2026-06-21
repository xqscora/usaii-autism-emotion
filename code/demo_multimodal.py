"""Audio + face fusion on video or mic+frame.

Voice (RAVDESS w2v, 8-class) + face (FER-Autism ResNet18, 6-class) -> fused label.
Video with sound track = true simultaneous. Webcam uses short mic clip + one frame.

Usage:
  python code/demo_multimodal.py clip.mp4
  python code/demo_multimodal.py clip.mp4 --out labeled.mp4
  python code/demo_multimodal.py --camera --seconds 4
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from demo_emotion import predict_emotion as predict_audio  # noqa: E402
from demo_face_emotion import (  # noqa: E402
    annotate_frame,
    detect_faces,
    load_model as load_face_model,
    predict_crop,
    read_bgr,
)

# RAVDESS label -> FER-Autism label
AUDIO_TO_FACE = {
    "angry": "anger",
    "fearful": "fear",
    "happy": "joy",
    "sad": "sadness",
    "surprised": "surprise",
    "neutral": "natural",
    "calm": "natural",
    "disgust": "anger",
}


def align_audio_probs(audio_probs: dict[str, float], face_classes: list[str]) -> np.ndarray:
    vec = np.zeros(len(face_classes), dtype=np.float64)
    for name, p in audio_probs.items():
        mapped = AUDIO_TO_FACE.get(name)
        if mapped in face_classes:
            vec[face_classes.index(mapped)] += float(p)
    s = vec.sum()
    return vec / s if s > 0 else vec


def fuse_probs(
    face_probs: dict[str, float],
    audio_probs: dict[str, float],
    face_classes: list[str],
    *,
    w_audio: float = 0.45,
    w_face: float = 0.55,
) -> tuple[str, float, dict[str, float]]:
    f = np.array([face_probs.get(c, 0.0) for c in face_classes], dtype=np.float64)
    fs = f.sum()
    if fs > 0:
        f /= fs
    a = align_audio_probs(audio_probs, face_classes)
    fused = w_audio * a + w_face * f
    s = fused.sum()
    if s > 0:
        fused /= s
    idx = int(fused.argmax())
    out = {c: round(float(fused[i]), 3) for i, c in enumerate(face_classes)}
    return face_classes[idx], float(fused[idx]), out


def face_probs_from_frame(frame, detector, face_model, face_classes, img_size) -> dict[str, float]:
    probs_acc = np.zeros(len(face_classes), dtype=np.float64)
    n = 0
    for x, y, w, h in detect_faces(detector, frame):
        crop = frame[y : y + h, x : x + w]
        _, _, probs = predict_crop(face_model, face_classes, crop, img_size)
        for i, c in enumerate(face_classes):
            probs_acc[i] += probs.get(c, 0.0)
        n += 1
    if n == 0:
        _, _, probs = predict_crop(face_model, face_classes, frame, img_size)
        return probs
    vec = probs_acc / n
    s = vec.sum()
    if s > 0:
        vec /= s
    return {c: round(float(vec[i]), 3) for i, c in enumerate(face_classes)}


def process_video(
    path: Path,
    *,
    w_audio: float,
    w_face: float,
    out_path: Path | None,
    show: bool,
    face_weights: Path,
    sample_every: int = 15,
) -> None:
    face_model, face_classes, face_acc, _, img_size = load_face_model(face_weights)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("extracting audio track...")
    speech, _ = librosa.load(str(path), sr=16000, mono=True)
    a_label, a_conf, a_probs = predict_audio(speech)
    print(f"audio: {a_label} ({a_conf:.2f})")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    face_sum = np.zeros(len(face_classes), dtype=np.float64)
    face_n = 0
    n = 0
    last_fused = ("?", 0.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % sample_every == 0:
            fp = face_probs_from_frame(frame, detector, face_model, face_classes, img_size)
            for i, c in enumerate(face_classes):
                face_sum[i] += fp.get(c, 0.0)
            face_n += 1
        if face_n > 0:
            vec = face_sum / face_n
            s = vec.sum()
            if s > 0:
                vec /= s
            f_probs = {c: float(vec[i]) for i, c in enumerate(face_classes)}
            fused_label, fused_conf, _ = fuse_probs(f_probs, a_probs, face_classes, w_audio=w_audio, w_face=w_face)
            last_fused = (fused_label, fused_conf)

        annotated, _ = annotate_frame(frame, detector, face_model, face_classes, img_size)
        cv2.putText(
            annotated,
            f"audio:{a_label[:8]} fused:{last_fused[0]} {last_fused[1]:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )
        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("multimodal (q=quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    vec = face_sum / max(face_n, 1)
    s = vec.sum()
    if s > 0:
        vec /= s
    f_probs = {c: float(vec[i]) for i, c in enumerate(face_classes)}
    f_top = face_classes[int(vec.argmax())]
    fused_label, fused_conf, fused_all = fuse_probs(f_probs, a_probs, face_classes, w_audio=w_audio, w_face=w_face)

    print(f"face (avg {face_n} samples): {f_top}")
    print(f"face probs: {f_probs}")
    print(f"fused: {fused_label} ({fused_conf:.3f})")
    print(f"fused all: {fused_all}")
    print(f"models face_test_acc={face_acc:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="Video file with audio")
    ap.add_argument("--camera", action="store_true", help="One frame + record audio via ffmpeg mic (Windows may vary)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--w-audio", type=float, default=0.45)
    ap.add_argument("--w-face", type=float, default=0.55)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--face-weights", type=str, default=str(HERE.parent / "fer_autism_model.pt"))
    args = ap.parse_args()

    if args.camera:
        print("camera multimodal: capture frame + use demo_emotion on separate audio clip")
        print("for reliable A+V use a video file: python code/demo_multimodal.py clip.mp4")
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("camera frame failed")
        face_model, face_classes, _, _, img_size = load_face_model(Path(args.face_weights))
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        f_probs = face_probs_from_frame(frame, detector, face_model, face_classes, img_size)
        print("face:", f_probs)
        print("record audio with demo_emotion.py then fuse manually for now")
        return

    if not args.video:
        ap.print_help()
        sys.exit(1)

    process_video(
        Path(args.video),
        w_audio=args.w_audio,
        w_face=args.w_face,
        out_path=Path(args.out) if args.out else None,
        show=not args.no_show,
        face_weights=Path(args.face_weights),
    )


if __name__ == "__main__":
    main()
