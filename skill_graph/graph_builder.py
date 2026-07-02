import re
from collections import defaultdict
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Skill normalization
# --------------------------------------------------------------------------
# The candidate data uses human-facing skill names ("Sentence Transformers",
# "Hugging Face Transformers", "LLMs") while the JD uses free-text tokens
# ("sentence-transformers", "LLM fine-tuning", "vector database").
# We map both vocabularies onto a shared set of *canonical* nodes so that
# matching is exact-on-canonical instead of fragile substring matching
# (which previously let "MAP" match "roadmap" and "E5" match any "e5").

_WS = re.compile(r"[\s/_]+")
_PUNCT = re.compile(r"[^a-z0-9\- ]")


def _clean(raw: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace to single spaces."""
    if not raw:
        return ""
    s = raw.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


# Maps a cleaned variant -> canonical node id.
# Anything not present falls through to its own cleaned form (so exact
# string matches like "python" still work without an explicit alias).
ALIASES: Dict[str, str] = {
    # --- embeddings / retrieval ---
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "text embeddings": "embeddings",
    "openai embeddings": "openai-embeddings",
    "sentence transformers": "sentence-transformers",
    "sentence-transformers": "sentence-transformers",
    "sbert": "sentence-transformers",
    "hugging face transformers": "transformers",
    "huggingface transformers": "transformers",
    "transformers": "transformers",
    "bge": "bge",
    "e5": "e5",
    "retrieval": "information-retrieval",
    "information retrieval": "information-retrieval",
    "ir": "information-retrieval",
    "semantic search": "semantic-search",
    "vector search": "vector-search",
    "similarity search": "vector-search",
    "hybrid search": "hybrid-search",
    "bm25": "hybrid-search",
    "rag": "rag",
    "retrieval augmented generation": "rag",
    # --- vector databases ---
    "vector database": "vector-database",
    "vector databases": "vector-database",
    "vector db": "vector-database",
    "faiss": "faiss",
    "pinecone": "pinecone",
    "weaviate": "weaviate",
    "qdrant": "qdrant",
    "milvus": "milvus",
    "pgvector": "pgvector",
    "chroma": "chroma",
    "chromadb": "chroma",
    "elasticsearch": "elasticsearch",
    "opensearch": "opensearch",
    # --- LLMs / fine-tuning ---
    "llm": "llm",
    "llms": "llm",
    "large language models": "llm",
    "langchain": "langchain",
    "llamaindex": "llamaindex",
    "llama index": "llamaindex",
    "fine tuning": "fine-tuning",
    "fine tuning llms": "fine-tuning",
    "llm fine tuning": "fine-tuning",
    "lora": "lora",
    "qlora": "qlora",
    "peft": "peft",
    "prompt engineering": "prompt-engineering",
    # --- ML frameworks / ops ---
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "jax": "jax",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "mlops": "mlops",
    "mlflow": "mlflow",
    "kubeflow": "kubeflow",
    "bentoml": "bentoml",
    "weights biases": "weights-and-biases",
    "weights and biases": "weights-and-biases",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "machine learning": "machine-learning",
    "deep learning": "deep-learning",
    "ai ml": "ai-ml",
    "ai-ml": "ai-ml",
    # --- ranking / evaluation ---
    "ranking evaluation": "ranking-evaluation",
    "learning to rank": "learning-to-rank",
    "learning-to-rank": "learning-to-rank",
    "ltr": "learning-to-rank",
    "neural ranking": "neural-ranking",
    "ndcg": "ndcg",
    "mrr": "mrr",
    "map": "map",
    "mean average precision": "map",
    "a b test interpretation": "a-b-testing",
    "a b testing": "a-b-testing",
    "ab testing": "a-b-testing",
    "offline-to-online correlation": "offline-online-correlation",
    "offline to online correlation": "offline-online-correlation",
    # --- infra / languages ---
    "python": "python",
    "distributed systems": "distributed-systems",
    "large-scale inference optimization": "inference-optimization",
    "large scale inference optimization": "inference-optimization",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "spark": "spark",
    "airflow": "airflow",
}


def normalize_skill(raw: str) -> Optional[str]:
    """Return the canonical node id for a skill string, or None if empty."""
    cleaned = _clean(raw)
    if not cleaned:
        return None
    return ALIASES.get(cleaned, cleaned)


class SkillGraph:
    """Undirected adjacency graph over *canonical* skill nodes.

    Edges connect transferable/adjacent skills so a candidate who knows a
    neighbouring skill gets partial credit for a required skill via BFS.
    """

    def __init__(self):
        self.adj: Dict[str, List[str]] = defaultdict(list)

        edges = [
            # embeddings hub
            ("embeddings", "sentence-transformers"),
            ("embeddings", "openai-embeddings"),
            ("embeddings", "bge"),
            ("embeddings", "e5"),
            ("embeddings", "transformers"),
            ("embeddings", "vector-search"),
            # retrieval cluster
            ("information-retrieval", "semantic-search"),
            ("information-retrieval", "vector-search"),
            ("information-retrieval", "hybrid-search"),
            ("information-retrieval", "ranking-evaluation"),
            ("semantic-search", "vector-search"),
            ("vector-search", "vector-database"),
            ("hybrid-search", "vector-database"),
            ("rag", "vector-database"),
            ("rag", "llm"),
            ("rag", "embeddings"),
            # vector database members
            ("vector-database", "faiss"),
            ("vector-database", "pinecone"),
            ("vector-database", "weaviate"),
            ("vector-database", "qdrant"),
            ("vector-database", "milvus"),
            ("vector-database", "pgvector"),
            ("vector-database", "chroma"),
            ("vector-database", "elasticsearch"),
            ("elasticsearch", "opensearch"),
            ("hybrid-search", "elasticsearch"),
            # LLM / fine-tuning
            ("llm", "transformers"),
            ("llm", "fine-tuning"),
            ("llm", "langchain"),
            ("llm", "llamaindex"),
            ("llm", "prompt-engineering"),
            ("fine-tuning", "lora"),
            ("fine-tuning", "peft"),
            ("lora", "qlora"),
            ("peft", "lora"),
            # ranking / eval
            ("ranking-evaluation", "ndcg"),
            ("ranking-evaluation", "mrr"),
            ("ranking-evaluation", "map"),
            ("ranking-evaluation", "learning-to-rank"),
            ("learning-to-rank", "neural-ranking"),
            ("learning-to-rank", "xgboost"),
            ("neural-ranking", "transformers"),
            ("a-b-testing", "offline-online-correlation"),
            ("a-b-testing", "ranking-evaluation"),
            # ML frameworks / ops
            ("transformers", "pytorch"),
            ("pytorch", "tensorflow"),
            ("tensorflow", "keras"),
            ("pytorch", "jax"),
            ("scikit-learn", "xgboost"),
            ("xgboost", "lightgbm"),
            ("machine-learning", "deep-learning"),
            ("deep-learning", "transformers"),
            ("machine-learning", "scikit-learn"),
            ("ai-ml", "machine-learning"),
            ("nlp", "transformers"),
            ("mlops", "mlflow"),
            ("mlops", "kubeflow"),
            ("mlops", "bentoml"),
            ("mlops", "weights-and-biases"),
            ("mlops", "inference-optimization"),
            ("inference-optimization", "distributed-systems"),
            ("distributed-systems", "kubernetes"),
            ("kubernetes", "docker"),
        ]

        for u, v in edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def get_neighbors(self, node: str) -> List[str]:
        return self.adj.get(node, [])
