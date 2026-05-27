# Arastirma Ussu - Makefile
# Windows + Git Bash only. If moved to WSL, change the VENV path.
.PHONY: sanity smoke layer0 layer1 layer2 layer3 layer4 layer5 layer55 lint clean qdrant-up qdrant-down

# Environment
VENV := .venv/Scripts/python
PYTEST := $(VENV) -m pytest
export PYTHONUTF8 := 1

# --- Sanity (Layer 0) -----------------------------------
sanity:
	$(VENV) scripts/toolcall_sanity.py || echo "TOOLCALL: FAIL - manual ReAct will be used"
	$(PYTEST) tests/test_environment.py -v

# --- Per-layer smoke tests ------------------------------
layer0: sanity

layer1:
	$(PYTEST) tests/test_agent.py -m smoke -v --tb=short

layer2:
	$(PYTEST) tests/test_ingest.py -m smoke -v --tb=short

layer3:
	$(PYTEST) tests/test_memory.py -m smoke -v --tb=short

layer4:
	$(PYTEST) tests/test_crew.py -m smoke -v --tb=short

layer5:
	$(PYTEST) tests/test_guards.py -m smoke -v --tb=short

layer55:
	$(PYTEST) tests/test_eval.py -m "not experimental and not integration" -v --tb=short

# --- General --------------------------------------------
smoke:
	$(PYTEST) tests/ -m smoke -v --tb=short

lint:
	$(VENV) -m ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

# --- Qdrant Docker (Layer 3+) ---------------------------
qdrant-up:
	docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
		-v "$(CURDIR)/data/qdrant_storage:/qdrant/storage" \
		qdrant/qdrant:latest

qdrant-down:
	docker stop qdrant && docker rm qdrant
