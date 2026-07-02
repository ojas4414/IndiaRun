---
title: AI Candidate Ranking Sandbox
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: sandbox_app.py
pinned: false
---

# AI Candidate-Ranking Sandbox

Upload a small `candidates.jsonl` sample (a subset of the challenge file works
fine) to run the deterministic ranking pipeline end-to-end: honeypot
filtering, skill-graph + trajectory scoring, contrastive sentence attention,
and behavioral-twin disambiguation. Fully offline, CPU-only, no precomputed
index required.

Full source, the 100k-candidate pipeline, evaluation harness, and writeup:
**https://github.com/ojas4414/IndiaRun**
