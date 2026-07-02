import faiss
import numpy as np
import os

class CandidateIndex:
    def __init__(self, dim: int = 384):
        self.dim = dim
        # IndexFlatIP with normalized embeddings = Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.candidate_ids = []

    def add_batch(self, embeddings: np.ndarray, ids: list[str]):
        """Adds a batch of normalized embeddings to the index."""
        if len(embeddings) != len(ids):
            raise ValueError("Embeddings and IDs length mismatch")
            
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.candidate_ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 500):
        """
        Searches the index for the query embedding.
        Returns a list of (candidate_id, similarity_score).
        """
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.candidate_ids):
                results.append((self.candidate_ids[idx], float(score)))
                
        return results

    def save(self, index_path: str, ids_path: str):
        """Saves the FAISS index and the ordered candidate IDs to disk."""
        faiss.write_index(self.index, index_path)
        with open(ids_path, 'w', encoding='utf-8') as f:
            for cid in self.candidate_ids:
                f.write(f"{cid}\n")

    def load(self, index_path: str, ids_path: str):
        """Loads the FAISS index and candidate IDs from disk."""
        if not os.path.exists(index_path) or not os.path.exists(ids_path):
            raise FileNotFoundError("Index or IDs file not found")
            
        self.index = faiss.read_index(index_path)
        with open(ids_path, 'r', encoding='utf-8') as f:
            self.candidate_ids = [line.strip() for line in f if line.strip()]
