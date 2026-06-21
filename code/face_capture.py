"""Long-run webcam face capture — one folder per person (local persistence).

Like antigravity-style per-subject storage for later personalize / fine-tune.

Usage:
  python code/face_capture.py --person-id child_01
  python code/face_capture.py --person-id demo --root datasets/captured --interval 0.4
  python code/face_capture.py --person-id p1 --root /content/drive/MyDrive/usaii-capture   # Colab + Drive

Ctrl+C to stop. Writes manifest.jsonl (append-only).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "datasets" / "captured"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def append_manifest(manifest: Path, row: dict) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-id", required=True, help="Folder name for this wearer/subject")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--interval", type=float, default=0.35, help="Seconds between saved frames")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--min-size", type=int, default=80, help="Min face box px")
    args = ap.parse_args()

    out_dir = Path(args.root) / args.person_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    detector = cv2.CascadeClassifier(CASCADE)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    print(f"capturing -> {out_dir}")
    print("Ctrl+C to stop")

    last_save = 0.0
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(args.min_size, args.min_size))
            now = time.time()
            if faces and (now - last_save) >= args.interval:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                crop = frame[y : y + h, x : x + w]
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                fname = f"{ts}.jpg"
                path = out_dir / fname
                cv2.imwrite(str(path), crop)
                append_manifest(
                    manifest,
                    {
                        "ts": ts,
                        "file": fname,
                        "person_id": args.person_id,
                        "box": [int(x), int(y), int(w), int(h)],
                    },
                )
                count += 1
                last_save = now
                print(f"saved {fname} (#{count})")

            cv2.imshow("face_capture (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"stopped. {count} faces in {out_dir}")


if __name__ == "__main__":
    main()
