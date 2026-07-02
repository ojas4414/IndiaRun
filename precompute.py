import os
import json
from tqdm import tqdm
from data.loader import stream_raw_candidates
from honeypot.detector import HoneypotDetector
from embeddings.embedder import CandidateEmbedder
from embeddings.faiss_index import CandidateIndex

def precompute(candidates_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing components...")
    detector = HoneypotDetector()
    embedder = CandidateEmbedder()
    index = CandidateIndex(dim=384) # all-MiniLM-L6-v2 dimension
    
    # Load JD requirements to embed the JD too
    with open("jd_requirements.json", "r", encoding="utf-8") as f:
        jd_reqs = json.load(f)
        
    jd_text = (
        "Ideal roles: " + ", ".join(jd_reqs.get("ideal_roles", [])) + " | " +
        "Required: " + ", ".join(jd_reqs["required_skills"]) + " | " +
        "Preferred: " + ", ".join(jd_reqs["preferred_skills"]) + " | " +
        f"Domain: {jd_reqs['domain']} | Seniority: {jd_reqs['seniority']} | " +
        "Culture: " + ", ".join(jd_reqs.get("culture_signals", []))
    )
    jd_embedding = embedder.embed_jd(jd_text)
    np_path = os.path.join(output_dir, "jd_embedding.npy")
    import numpy as np
    np.save(np_path, jd_embedding)
    print(f"Saved JD embedding to {np_path}")
    
    print("Starting multi-process pool for embedding...")
    embedder.start_pool()

    batch_texts = []
    batch_ids = []
    batch_size = 2048 # Larger batch for multiprocessing
    
    honeypot_count = 0
    clean_count = 0
    
    print("Processing candidates...")
    
    # Need to read total lines for tqdm if possible, but let's just use an infinite tqdm
    with tqdm() as pbar:
        for candidate in stream_raw_candidates(candidates_path):
            is_honeypot, reason = detector.check_candidate(candidate)
            if is_honeypot:
                honeypot_count += 1
                pbar.update(1)
                continue
                
            clean_count += 1
            text = embedder.extract_text(candidate)
            batch_texts.append(text)
            batch_ids.append(candidate["candidate_id"])
            
            if len(batch_texts) >= batch_size:
                embeddings = embedder.embed_batch(batch_texts)
                index.add_batch(embeddings, batch_ids)
                batch_texts = []
                batch_ids = []
            
            pbar.update(1)
            
    # Process remaining
    if batch_texts:
        embeddings = embedder.embed_batch(batch_texts)
        index.add_batch(embeddings, batch_ids)
        
    embedder.stop_pool()
        
    print(f"\nProcessed {clean_count + honeypot_count} candidates.")
    print(f"Honeypots detected: {honeypot_count}")
    print(f"Clean candidates: {clean_count}")
    
    index_path = os.path.join(output_dir, "faiss.index")
    ids_path = os.path.join(output_dir, "candidate_ids.txt")
    index.save(index_path, ids_path)
    print(f"Saved FAISS index to {index_path} and IDs to {ids_path}")

if __name__ == "__main__":
    import argparse

    # Default to the mounted data path used in the Docker image, then fall
    # back to the local challenge download for host runs.
    default_local = (
        r"c:\Users\Ojas\Downloads\[PUB] India_runs_data_and_ai_challenge"
        r"\[PUB] India_runs_data_and_ai_challenge"
        r"\India_runs_data_and_ai_challenge\candidates.jsonl"
    )
    default_candidates = os.environ.get("CANDIDATES_PATH")
    if not default_candidates:
        default_candidates = "/data/candidates.jsonl" if os.path.exists("/data/candidates.jsonl") else default_local

    parser = argparse.ArgumentParser(description="Precompute embeddings + FAISS index")
    parser.add_argument("--candidates", default=default_candidates,
                        help="Path to candidates.jsonl")
    parser.add_argument("--out-dir", default=os.environ.get("ARTIFACTS_DIR", "artifacts"),
                        help="Directory to write faiss.index / candidate_ids.txt")
    args = parser.parse_args()

    precompute(args.candidates, args.out_dir)
