PY ?= python
.PHONY: setup sample api ui test eval smoke
setup:            ## create venv + install deps
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
sample:           ## generate sample HLD PDFs + ground truth into ./data
	$(PY) -m scripts.make_sample_hld
api:              ## start the backend on :8000
	uvicorn app.api:app --reload --port 8000
ui:               ## start the Streamlit UI on :8501
	streamlit run ui/streamlit_app.py
test:             ## unit/API tests (no Ollama or GPU needed)
	MOCK_LLM=1 EMBEDDING_BACKEND=hash $(PY) -m pytest -q
eval:             ## extraction precision/recall (uses your real LLM unless MOCK_LLM=1)
	$(PY) -m scripts.evaluate_extraction
smoke:            ## real server + headless UI end-to-end check
	$(PY) -m scripts.ui_smoke
