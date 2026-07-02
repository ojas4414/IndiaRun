import argparse
import json
import time
from data.loader import stream_raw_candidates
from scoring.hybrid_scorer import HybridScorer
from reasoning.template_generator import ReasoningGenerator
from output.format_output import write_submission

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob Hackathon")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Loading JD requirements...")
    with open("jd_requirements.json", "r", encoding="utf-8") as f:
        jd_reqs = json.load(f)
        
    print("Initializing components...")
    scorer = HybridScorer(
        jd_requirements=jd_reqs, 
        index_path="artifacts/faiss.index", 
        ids_path="artifacts/candidate_ids.txt",
        jd_embedding_path="artifacts/jd_embedding.npy"
    )
    reasoner = ReasoningGenerator(jd_requirements=jd_reqs)
    
    # We need to map candidate_id to candidate data for the scoring step.
    # To keep within 16GB, we could either:
    # 1. Load all into memory (100k dicts might take ~1GB, which is fine)
    # 2. Or, we can just load them. Let's load all for now.
    
    print("Loading candidate data into memory...")
    candidates_dict = {}
    for cand in stream_raw_candidates(args.candidates):
        candidates_dict[cand["candidate_id"]] = cand
        
    print("Scoring candidates...")
    top_100_results = scorer.score_all(candidates_dict)
    
    print("Generating reasoning and formatting output...")
    final_output = []
    for rank, (cid, score) in enumerate(top_100_results):
        cand = candidates_dict[cid]
        evidence = scorer.attention_info.get(cid, {}).get("evidence")
        reasoning = reasoner.generate(cand, score, rank + 1, evidence=evidence)
        final_output.append((cid, score, reasoning))
        
    write_submission(args.out, final_output)
    
    end_time = time.time()
    print(f"Ranking complete in {end_time - start_time:.2f} seconds.")
    print(f"Output saved to {args.out}")

if __name__ == "__main__":
    main()
