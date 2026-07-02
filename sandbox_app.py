"""HuggingFace Spaces sandbox demo.

Deliberately independent of artifacts/: it embeds the JD on the fly and scores
whatever small file is uploaded directly via HybridScorer.score_sample(),
instead of retrieving from the prebuilt 100k-candidate FAISS index. That
avoids two problems a FAISS-based sandbox would have -- (1) needing to bundle
a 150MB+ index into the Space, and (2) silently returning an empty table
whenever an uploaded sample's candidate_ids don't happen to land in that
index's top-K funnel (guaranteed for the full pool, not guaranteed for an
arbitrary small subset).
"""
import json
import time

import gradio as gr

from embeddings.embedder import CandidateEmbedder, build_jd_text
from honeypot.detector import HoneypotDetector
from scoring.hybrid_scorer import HybridScorer
from reasoning.template_generator import ReasoningGenerator

# The sandbox is for a small sample, per the challenge spec. Capping keeps a
# single request fast on a free CPU Space even if someone uploads the full pool.
MAX_SANDBOX_CANDIDATES = 1000

_state = {}


def _lazy_init():
    """Build the JD embedding and scorer once, on first request."""
    if _state:
        return
    with open("jd_requirements.json", "r", encoding="utf-8") as f:
        jd_reqs = json.load(f)
    embedder = CandidateEmbedder()
    jd_embedding = embedder.embed_jd(build_jd_text(jd_reqs))
    _state["embedder"] = embedder
    _state["scorer"] = HybridScorer(jd_requirements=jd_reqs, jd_embedding=jd_embedding,
                                    shared_model=embedder.model)
    _state["reasoner"] = ReasoningGenerator(jd_requirements=jd_reqs)
    _state["detector"] = HoneypotDetector()


def process_sandbox(candidates_file):
    if not candidates_file:
        return "No file uploaded."

    _lazy_init()
    start_time = time.time()

    candidates_dict = {}
    honeypots_filtered = 0
    truncated = False
    with open(candidates_file.name, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if len(candidates_dict) >= MAX_SANDBOX_CANDIDATES:
                truncated = True
                break
            cand = json.loads(line)
            is_honeypot, _ = _state["detector"].check_candidate(cand)
            if is_honeypot:
                honeypots_filtered += 1
                continue
            candidates_dict[cand["candidate_id"]] = cand

    if not candidates_dict:
        return "No valid (non-honeypot) candidates found in the uploaded file."

    results = _state["scorer"].score_sample(candidates_dict, _state["embedder"])
    elapsed = time.time() - start_time

    header = (f"### Sandbox Results\n"
             f"{len(candidates_dict)} candidate(s) scored in {elapsed:.2f}s "
             f"(fully offline, CPU-only, deterministic).\n\n")
    if honeypots_filtered:
        header += f"_Filtered {honeypots_filtered} honeypot profile(s)._\n\n"
    if truncated:
        header += (f"_File exceeds the {MAX_SANDBOX_CANDIDATES}-candidate sandbox "
                   f"cap; only the first {MAX_SANDBOX_CANDIDATES} were scored. "
                   f"Use `rank.py` (or `docker compose up`) locally for the full "
                   f"100k-candidate pool._\n\n")

    lines = [header, "| Rank | Candidate ID | Score | Reasoning |", "|---|---|---|---|"]
    for rank, (cid, score) in enumerate(results):
        cand = candidates_dict[cid]
        evidence = _state["scorer"].attention_info.get(cid, {}).get("evidence")
        reasoning = _state["reasoner"].generate(cand, score, rank + 1, evidence=evidence)
        reasoning = reasoning.replace("|", "\\|")  # keep the markdown table intact
        lines.append(f"| {rank + 1} | {cid} | {score:.4f} | {reasoning} |")

    return "\n".join(lines)


demo = gr.Interface(
    fn=process_sandbox,
    inputs=gr.File(label="Upload a small candidates.jsonl sample"),
    outputs=gr.Markdown(),
    title="AI Candidate-Ranking Sandbox",
    description=(
        "Upload a JSONL sample (a subset of candidates.jsonl works fine -- up "
        f"to {MAX_SANDBOX_CANDIDATES} candidates) to run the full deterministic "
        "pipeline end-to-end: honeypot filtering, skill-graph + trajectory "
        "scoring, contrastive sentence attention, and behavioral-twin "
        "disambiguation. Runs fully offline, CPU-only, no precomputed index "
        "required. See github.com/ojas4414/IndiaRun for the full 100k-candidate "
        "pipeline (rank.py / docker compose up)."
    ),
)

if __name__ == "__main__":
    demo.launch()
