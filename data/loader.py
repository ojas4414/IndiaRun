import json
from typing import Iterator
from .schema import Candidate

def stream_candidates(filepath: str) -> Iterator[Candidate]:
    """
    Streams candidates from a JSONL file, validating each against the schema.
    This avoids loading the entire 487MB file into memory at once.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            yield Candidate(**data)

def stream_raw_candidates(filepath: str) -> Iterator[dict]:
    """
    Streams raw dictionaries from a JSONL file for faster processing when
    full validation isn't needed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
