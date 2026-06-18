"""Build the USAII demo video: text cards (PIL) + result figures + edge-tts narration -> ffmpeg mux.
Output: heard_demo.mp4 (1920x1080). Run: python make_video.py
Reuses the proven STEMINATE pipeline approach.
"""
import asyncio
import os
import subprocess
import sys

import edge_tts
import imageio_ffmpeg
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
CARDS = os.path.join(HERE, "cards")
AUDIO = os.path.join(HERE, "audio")
FIGS = os.path.join(PROJ, "figures")
os.makedirs(CARDS, exist_ok=True)
os.makedirs(AUDIO, exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VOICE = "en-US-AriaNeural"
OUT = os.path.join(HERE, "heard_demo.mp4")
W, H = 1920, 1080
BG = (11, 15, 26)
PURPLE = (124, 92, 255)
WHITE = (235, 238, 245)
GRAY = (150, 158, 175)
LEAD, TAIL = 0.35, 0.9


def _font(size, bold=True):
    for p in [r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
              r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def card(name, title, body=None, accent=None, foot=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 300
    for ln in _wrap(d, title, _font(76), W - 360):
        d.text((180, y), ln, font=_font(76), fill=WHITE); y += 96
    if accent:
        y += 20
        for ln in _wrap(d, accent, _font(60), W - 360):
            d.text((180, y), ln, font=_font(60), fill=PURPLE); y += 78
    if body:
        y += 30
        for ln in _wrap(d, body, _font(44, False), W - 360):
            d.text((180, y), ln, font=_font(44, False), fill=GRAY); y += 62
    if foot:
        d.text((180, H - 130), foot, font=_font(36, False), fill=PURPLE)
    p = os.path.join(CARDS, name)
    img.save(p)
    return p


# build the cards
c_problem = card("c1_problem.png",
                 "An autistic kid in distress often can't be read by the people around them.",
                 accent="Being understood is the precondition for being supported.",
                 body="And generic emotion AI is confidently wrong on autistic voices (Apple, 2025).")
c_demo = card("c2_demo.png",
              "Voice-first emotion reader",
              accent="speech  →  🎭 emotion  +  confidence  (abstains when unsure)",
              body="Baseline: 64% speaker-independent on RAVDESS  ·  chance = 12.5%",
              foot="live demo: speech in -> 'calm' 0.97")
c_vision = card("c3_vision.png",
                "Every child gets a model that learns THEM.",
                accent="A personalized model that grows with the child (framework: Cerome)",
                body="Honest: trained on neurotypical speech · autism adaptation next · prototype, not clinical · a human always confirms.")
c_close = card("c4_close.png",
               "Being understood is the first step to being supported.",
               accent="That's what we're building.",
               foot="github.com/xqscora/usaii-autism-emotion  ·  live Colab in description")

SEGMENTS = [
    (c_problem,
     "When an autistic kid is overwhelmed, it often comes out in a way the people around them can't "
     "read. The support is right there, but it never lands, because the kid isn't understood. Being "
     "understood is the precondition for being supported. And generic emotion A-I makes it worse: "
     "trained on neurotypical speech, it is confidently wrong on autistic voices."),
    (c_demo,
     "So we built a voice-first emotion reader. Speech goes in, it reads the emotion, and when it isn't "
     "sure, it says so instead of guessing. Our baseline is sixty-four percent speaker-independent on "
     "RAVDESS. An honest number — chance is twelve and a half percent."),
    (os.path.join(FIGS, "personalization_curve.png"),
     "But the real idea isn't one generic model. It's that every child gets a model that learns them. "
     "Give it just a few samples of one kid, and it understands that kid better. We measured it: "
     "sixty-seven percent rising to seventy-seven percent with only three samples per emotion."),
    (os.path.join(FIGS, "per_speaker.png"),
     "And here is the part that matters most. The speakers the generic model read worst improved the "
     "most — plus nineteen points for the hardest one. That is exactly the autistic case: atypical "
     "expression, generic model fails, personalization rescues it."),
    (c_vision,
     "Long term, each child builds up a personalized model over time, a framework we call Cerome, so "
     "that even when they express things in a blurry, non-standard way, the model that has been learning "
     "them still gets it. To be honest about the limits: today's model is trained on neurotypical "
     "speech, the autism-specific adaptation is next, it is a prototype, not a clinical tool, and a "
     "human always confirms."),
    (c_close,
     "Being understood is the first step to being supported. That is what we are building. The repo and "
     "a live demo are in the description."),
]


async def gen_tts():
    paths = []
    for i, (_img, text) in enumerate(SEGMENTS):
        out = os.path.join(AUDIO, f"seg{i:02d}.mp3")
        for attempt in range(4):
            try:
                await edge_tts.Communicate(text, VOICE).save(out)
                if os.path.getsize(out) > 1000:
                    break
            except Exception as e:
                print(f"[tts] seg{i} attempt {attempt} failed: {e!r}", flush=True)
                await asyncio.sleep(2)
        paths.append(out)
        print(f"[tts] seg{i:02d} ({os.path.getsize(out)} B)", flush=True)
    return paths


def build(paths):
    durs = [MP3(p).info.length for p in paths]
    seg = [d + LEAD + TAIL for d in durs]
    total = sum(seg)
    print(f"[time] TOTAL = {total:.1f}s ({int(total//60)}m{int(total%60):02d}s)", flush=True)
    args = [FFMPEG, "-y"]
    for i, (img, _t) in enumerate(SEGMENTS):
        args += ["-loop", "1", "-t", f"{seg[i]:.3f}", "-i", img, "-i", paths[i]]
    fc = []
    n = len(SEGMENTS)
    for i in range(n):
        lm = int(LEAD * 1000)
        fc.append(f"[{2*i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                  f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b0f1a,setsar=1,fps=30,format=yuv420p[v{i}]")
        fc.append(f"[{2*i+1}:a]adelay={lm}|{lm},apad,atrim=0:{seg[i]:.3f},asetpts=N/SR/TB[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    fo = max(0.0, total - 0.9)
    fc.append(f"[vc]fade=t=in:st=0:d=0.6,fade=t=out:st={fo:.3f}:d=0.9[vout]")
    fc.append(f"[ac]afade=t=in:st=0:d=0.4,afade=t=out:st={fo:.3f}:d=0.9[aout]")
    args += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", OUT]
    print("[ffmpeg] encoding...", flush=True)
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("[ffmpeg] FAILED\n", r.stderr[-2500:], flush=True)
        sys.exit(1)
    print(f"[ffmpeg] OK -> {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    build(asyncio.run(gen_tts()))
