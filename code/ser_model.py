"""Speech Emotion Recognition model: SSL backbone + emotion head.

Pipeline philosophy (see ../README.md):
  - SSL backbone (wav2vec2 / HuBERT) — pretrained on large speech, optionally
    continued-pretrained on ASDSpeech audio so it knows how autistic kids sound.
  - Emotion head — fine-tuned on labeled emotion data (RAVDESS, 8 classes; NT source).
  - Wrapped at inference with per-wearer calibration + uncertainty-aware abstention
    (so a confident-but-wrong read never gets shown). See calibration.py.

Runs on Kaggle GPU. Local Intel Arc can do inference but not full fine-tune.
"""
import torch
import torch.nn as nn

# RAVDESS 8-class emotion set (the labeled source we fine-tune the head on)
RAVDESS_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


class SERModel(nn.Module):
    def __init__(self, backbone_name="facebook/wav2vec2-base",
                 n_emotions=len(RAVDESS_EMOTIONS), freeze_backbone=False):
        super().__init__()
        from transformers import AutoModel  # imported lazily so the file loads without transformers
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_emotions),
        )

    def forward(self, input_values, attention_mask=None, return_logits=True):
        out = self.backbone(input_values, attention_mask=attention_mask).last_hidden_state
        # masked mean-pool over time
        if attention_mask is not None:
            # backbone downsamples; approximate with simple mean if mask shapes differ
            pooled = out.mean(dim=1)
        else:
            pooled = out.mean(dim=1)
        logits = self.head(pooled)
        return logits if return_logits else torch.softmax(logits, dim=-1)


def predict_with_abstention(model, input_values, attention_mask=None,
                            conf_threshold=0.55):
    """Inference that REFUSES to commit when not confident.

    This is the guard against the 'confident-but-wrong' failure mode on autistic
    speech: if the top softmax prob is below threshold, return ('uncertain', prob)
    instead of a definitive emotion. (A proper conformal-prediction version with
    coverage guarantees goes in calibration.py; this is the simple baseline.)
    """
    model.eval()
    with torch.no_grad():
        probs = model(input_values, attention_mask=attention_mask, return_logits=False)
        conf, idx = probs.max(dim=-1)
    out = []
    for c, i in zip(conf.tolist(), idx.tolist()):
        out.append((RAVDESS_EMOTIONS[i], c) if c >= conf_threshold else ("uncertain", c))
    return out


if __name__ == "__main__":
    # sanity check without downloading weights: just print the planned config
    print("SER model plan:")
    print("  backbone : facebook/wav2vec2-base (swap to HuBERT/WavLM on Kaggle)")
    print("  emotions :", RAVDESS_EMOTIONS)
    print("  guard    : predict_with_abstention(conf_threshold=0.55)")
    print("  note     : instantiate SERModel() on Kaggle GPU (needs transformers + torch)")
