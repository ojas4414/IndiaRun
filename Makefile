.PHONY: precompute rank test validate baseline docker docker-clean

docker:
	docker compose up --build

# Remove the cached index so the next run rebuilds it from scratch.
docker-clean:
	rm -f artifacts/faiss.index artifacts/candidate_ids.txt

precompute:
	python precompute.py

rank:
	python rank.py --candidates ./candidates.jsonl --out ./submission.csv

validate:
	python validate_submission.py submission.csv

test:
	pytest tests/ -v

evaluate:
	python -m evaluation.evaluate

adversarial:
	python -m evaluation.adversarial

# Proof of determinism: rank twice and confirm the outputs are identical.
determinism:
	python rank.py --candidates ./candidates.jsonl --out /tmp/run_a.csv
	python rank.py --candidates ./candidates.jsonl --out /tmp/run_b.csv
	diff /tmp/run_a.csv /tmp/run_b.csv && echo "DETERMINISM OK: identical output"

baseline:
	python baseline/naive_llm_ranker.py
