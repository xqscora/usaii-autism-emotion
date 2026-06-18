# Project Brief — "Heard" (working name)

*USAII Global AI Hackathon 2026 · Cora Zeng & Mujahid · aligned with the community-support brief.*

## What we're building

A tool that helps autistic kids be **understood** by the people around them. It's voice-first — it listens to a child and reads the emotion in their voice. But the real point isn't "another emotion AI." The point is that **every child gets a model that learns *them*.**

## The problem (and why it's a community-support problem)

When an autistic kid — especially one whose expression is atypical — is upset or overwhelmed, it often comes out in a way the people around them (parents, teachers, peers) can't read. The support is right there, but it doesn't land, because the kid isn't understood. **Being understood is the precondition for being supported.**

And generic emotion AI makes this worse, not better: it's trained on neurotypical speech, so on autistic kids it's *confidently wrong* (documented — Apple, Interspeech 2025). A wrong, confident label is more harmful than no label.

## How we approach it

- **A speech-emotion baseline** — RAVDESS, 64% speaker-independent (an honest number, not cherry-picked; chance is 12.5%).
- **Per-child personalization — this is the core.** Give the model a few samples of one child and it understands *them* better. We measured it: **65% → 75% with just 3 samples per emotion**, and the speakers the generic model read *worst* improved the *most* (**+19 points** for the hardest one). That's exactly the autism case: atypical expression → generic model fails → personalization rescues it.
- **Abstention.** When the model isn't sure, it says "uncertain" instead of broadcasting a confident wrong guess.
- **The long-term frame.** Each child builds up a personalized model over time (the framework is called *Cerome*) — so even when they express things in a blurry, non-standard way, the model that's been learning *them* still gets it. The demo is the seed; the vision is a model that grows with the child.
- **The human stays in charge.** It surfaces a read, it doesn't decide. No clinical or diagnostic claims.

## Why us

Cora works on cognitive / personality modeling (the *Cerome* framework) and has published work on inner-voice self-regulation, with more on adolescent crisis in progress — personalizing a model to one mind is literally her research area. Mujahid drives the engineering.

## Where it is right now

- Working demo (**speech in → emotion out**, with abstention) + the personalization result above.
- Runnable in the browser, no install: **Colab** → https://colab.research.google.com/github/xqscora/usaii-autism-emotion/blob/master/demo_colab.ipynb
- Code & README: https://github.com/xqscora/usaii-autism-emotion

## Honest boundaries

The current model is trained on neurotypical adult speech; the autism-specific adaptation is the next step (we have the autistic-speech corpus, ASDSpeech, lined up, plus requests out for emotion-labeled autism datasets). This is a **prototype, not a clinical tool**, and a human always confirms before anything is acted on.
