import csv
from typing import List, Tuple

def write_submission(filepath: str, ranked_results: List[Tuple[str, float, str]]):
    """
    Writes the submission CSV matching exactly:
    candidate_id,rank,score,reasoning
    Rows 2-101
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for i, (cid, score, reasoning) in enumerate(ranked_results):
            rank = i + 1
            # Format score to 4 decimal places for consistency
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])
