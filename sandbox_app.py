import gradio as gr
import json
import time
from data.loader import stream_raw_candidates
from scoring.hybrid_scorer import HybridScorer
from reasoning.template_generator import ReasoningGenerator

def process_sandbox(candidates_file):
    # This is meant for HuggingFace Spaces sandbox demo
    # It takes a JSONL file upload, runs the pipeline, and returns results
    if not candidates_file:
        return "No file uploaded."
        
    start_time = time.time()
    
    with open("jd_requirements.json", "r", encoding="utf-8") as f:
        jd_reqs = json.load(f)
        
    scorer = HybridScorer(
        jd_requirements=jd_reqs, 
        index_path="artifacts/faiss.index", 
        ids_path="artifacts/candidate_ids.txt",
        jd_embedding_path="artifacts/jd_embedding.npy"
    )
    reasoner = ReasoningGenerator(jd_requirements=jd_reqs)
    
    candidates_dict = {}
    with open(candidates_file.name, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cand = json.loads(line)
                candidates_dict[cand["candidate_id"]] = cand
                
    top_100_results = scorer.score_all(candidates_dict)
    
    output_markdown = f"### Sandbox Results (took {time.time() - start_time:.2f}s)\n\n"
    output_markdown += "| Rank | Candidate ID | Score | Reasoning |\n"
    output_markdown += "|---|---|---|---|\n"
    
    for rank, (cid, score) in enumerate(top_100_results):
        cand = candidates_dict[cid]
        reasoning = reasoner.generate(cand, score, rank + 1)
        output_markdown += f"| {rank+1} | {cid} | {score:.4f} | {reasoning} |\n"
        
    return output_markdown

# Define Gradio interface
demo = gr.Interface(
    fn=process_sandbox,
    inputs=gr.File(label="Upload candidates JSONL sample"),
    outputs=gr.Markdown(),
    title="Redrob Ranker Sandbox",
    description="Upload a small candidates.jsonl file to test the deterministic ranking pipeline. Runs offline, on CPU."
)

if __name__ == "__main__":
    demo.launch()
