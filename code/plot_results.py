"""Generate result figures for the brief / demo video / Devpost submission.
Run: python plot_results.py  ->  figures/*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)
PURPLE = "#7c5cff"

# --- Figure 1: calibration curve (K-sweep) ---
Ks = [0, 1, 2, 3]
accs = [66.7, 70.8, 73.6, 77.1]
plt.figure(figsize=(6.2, 4))
plt.plot(Ks, accs, "o-", color=PURPLE, linewidth=2.5, markersize=9)
for k, a in zip(Ks, accs):
    plt.annotate(f"{a:.0f}%", (k, a + 1.2), ha="center", fontweight="bold")
plt.axhline(12.5, ls="--", color="gray", alpha=0.6)
plt.text(0, 15, "chance 12.5%", color="gray", fontsize=9)
plt.xlabel("calibration samples per emotion (per child)")
plt.ylabel("accuracy  (%)")
plt.title("A few samples, and the model understands THEM", fontweight="bold")
plt.xticks(Ks)
plt.ylim(0, 90)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "personalization_curve.png"), dpi=130)
plt.close()

# --- Figure 2: per-speaker generic vs personalized ---
spk = ["21", "22", "23", "24"]
base = [55.6, 86.1, 50.0, 69.4]
pers = [75.0, 91.7, 58.3, 75.0]
x = np.arange(len(spk))
w = 0.36
plt.figure(figsize=(6.2, 4))
plt.bar(x - w / 2, base, w, label="generic model", color="#c9c9c9")
plt.bar(x + w / 2, pers, w, label="+ personalized (3 samples)", color=PURPLE)
for i in range(len(spk)):
    plt.annotate(f"+{pers[i]-base[i]:.0f}", (x[i] + w / 2, pers[i] + 1.5),
                 ha="center", color=PURPLE, fontweight="bold", fontsize=9)
plt.xlabel("unseen speaker")
plt.ylabel("accuracy  (%)")
plt.title("The speakers read WORST improve the MOST", fontweight="bold")
plt.xticks(x, spk)
plt.legend()
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "per_speaker.png"), dpi=130)
plt.close()

print("saved figures/personalization_curve.png and figures/per_speaker.png")
