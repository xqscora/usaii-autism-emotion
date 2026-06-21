"""Face emotion inference — FER-Autism trained model.

Usage:
  python code/demo_face_emotion.py path/to.jpg
  python code/demo_face_emotion.py path/to.mp4 --out labeled.mp4
  python code/demo_face_emotion.py --camera
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "fer_autism_model.pt"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_bgr(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_model(path: Path = WEIGHTS):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from train_fer import TinyCNN, build_model

    classes = ckpt["classes"]
    backbone = ckpt.get("backbone", "tinycnn")
    img_size = int(ckpt.get("img_size", 48))
    if backbone == "tinycnn" and "net.0.weight" in ckpt.get("state_dict", {}):
        model = TinyCNN(len(classes))
    else:
        model = build_model(backbone, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    acc = float(ckpt.get("test_acc", ckpt.get("val_acc", 0)))
    return model, classes, acc, backbone, img_size


def preprocess_bgr(bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def predict_crop(model, classes, bgr_crop, img_size: int):
    x = preprocess_bgr(bgr_crop, img_size)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), {c: round(float(p), 3) for c, p in zip(classes, probs)}


def _looks_like_face_crop(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    if h < 8 or w < 8:
        return False
    return max(h, w) <= 256 or (0.7 <= w / h <= 1.4 and min(h, w) <= 200)


def detect_faces(detector, frame: np.ndarray, min_size: int = 60):
    if _looks_like_face_crop(frame):
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(min_size, min_size))
    if len(faces) == 0:
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]
    return [tuple(int(v) for v in f) for f in faces]


def annotate_frame(frame, detector, model, classes, img_size):
    out = frame.copy()
    results = []
    for x, y, w, h in detect_faces(detector, frame):
        crop = frame[y : y + h, x : x + w]
        label, conf, probs = predict_crop(model, classes, crop, img_size)
        results.append((label, conf, probs))
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out, results


def run_image(path, detector, model, classes, img_size):
    frame = read_bgr(path)
    if frame is None:
        raise SystemExit(f"cannot read image: {path}")
    _, results = annotate_frame(frame, detector, model, classes, img_size)
    for i, (label, conf, probs) in enumerate(results):
        print(f"face[{i}] emotion: {label}  confidence: {conf:.3f}")
        print(f"all: {probs}")
    return 0 if results else 1


def run_video(path, detector, model, classes, img_size, out_path, show):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, results = annotate_frame(frame, detector, model, classes, img_size)
        if results and n % int(max(fps, 1)) == 0:
            label, conf, _ = results[0]
            print(f"frame {n}: {label} {conf:.3f}")
        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("demo_face_emotion video (q=quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    if writer:
        writer.release()
        print(f"saved {out_path}")
    if show:
        cv2.destroyAllWindows()
    print(f"processed {n} frames")


def run_camera(detector, model, classes, img_size, camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open camera {camera_id}. Check Windows Privacy > Camera."
        )
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        raise SystemExit(f"camera {camera_id} opened but no frames")
    print(f"camera ok {probe.shape}. q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, _ = annotate_frame(frame, detector, model, classes, img_size)
        cv2.imshow("demo_face_emotion (q=quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="Image or video")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--camera-id", type=int, default=0)
    ap.add_argument("--video", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--weights", type=str, default=str(WEIGHTS))
    args = ap.parse_args()

    model, classes, acc, backbone, img_size = load_model(Path(args.weights))
    print(f"loaded {args.weights} backbone={backbone} size={img_size} test_acc={acc:.3f}")
    print(f"classes={classes}")

    detector = cv2.CascadeClassifier(CASCADE)
    media = args.video or args.path

    if args.camera:
        run_camera(detector, model, classes, img_size, args.camera_id)
        return
    if not media:
        print("usage: demo_face_emotion.py image.jpg | video.mp4 | --camera")
        sys.exit(1)
    p = Path(media)
    if p.suffix.lower() in VIDEO_EXTS:
        run_video(p, detector, model, classes, img_size, Path(args.out) if args.out else None, not args.no_show)
        return
    sys.exit(run_image(p, detector, model, classes, img_size))


if __name__ == "__main__":
    main()
