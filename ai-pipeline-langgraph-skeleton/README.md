# AI Engineer Assignment — LangGraph + LangChain + LangSmith (Skeleton)

This repository is a **clean, modular skeleton** to implement:
- Routing agent with **LangGraph**
- Tools: **OpenWeatherMap** (real-time weather) and **RAG over PDF**
- **Qdrant** for embeddings storage
- **LangSmith** evaluation hooks
- **Streamlit** chat UI
- **Pytest** unit tests

> You will fill the TODOs and extend this skeleton with your implementation details.

## Quickstart

```bash
# 1) Create and activate venv (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Copy env and set your keys
cp .env.example .env
# Set: OPENAI_API_KEY or GROQ_API_KEY, OPENWEATHERMAP_API_KEY, LANGSMITH_API_KEY (optional)
# For Qdrant: leave localhost docker or set QDRANT_URL/QDRANT_API_KEY for cloud

# 4) Run Streamlit UI
streamlit run app/streamlit_app.py

# 5) Run tests
pytest -q
```

## Project Structure

```
ai-pipeline-langgraph-skeleton/
├─ app/
│  └─ streamlit_app.py            # Minimal chat UI that calls the graph
├─ src/
│  ├─ graph/
│  │  └─ agent_graph.py           # LangGraph routing and nodes
│  ├─ rag/
│  │  ├─ pdf_loader.py            # Load & chunk PDFs
│  │  └─ index.py                 # Embed and index into Qdrant
│  ├─ vectorstore/
│  │  └─ qdrant_store.py          # Qdrant client helpers
│  ├─ weather/
│  │  └─ api.py                   # OpenWeatherMap client
│  ├─ eval/
│  │  └─ langsmith_eval.py        # Hooks/placeholders for evaluation
│  ├─ config.py                   # Pydantic settings
│  └─ llm.py                      # LLM factory (OpenAI or Groq)
├─ tests/
│  ├─ test_graph_routing.py
│  ├─ test_rag.py
│  └─ test_weather_api.py
├─ data/
│  └─ sample.pdf                  # Put your PDF here (placeholder)
├─ .streamlit/
│  └─ config.toml
├─ scripts/
│  ├─ run_streamlit.sh
│  └─ run_streamlit.bat
├─ .env.example
├─ requirements.txt
├─ pyproject.toml
├─ Makefile
└─ README.md
```

## Notes

- **LangSmith**: Set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, optionally `LANGCHAIN_PROJECT`. Screenshots of evals can go in a `docs/` folder.
- **Qdrant**: You can run a local container quickly:
  ```bash
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  ```
- This repo keeps code **modular** and testable: each tool/node is isolated with clean interfaces.


## Working Without an OpenAI Key

You can complete the assignment **without** an OpenAI key:

- **LLM**: Use **Groq** (free tier often available). Set `GROQ_API_KEY` in `.env`. The project prefers Groq automatically.
- **Embeddings**: Local **sentence-transformers** (no key needed).
- **Vector DB**: Local **Qdrant** (no key needed).
- **Weather**: If `OPENWEATHERMAP_API_KEY` is missing, we **auto-fallback** to **Open-Meteo** (no API key). This still delivers real-time weather for demo. You can later switch to OpenWeatherMap by adding the key.

### Minimum `.env` (no keys mode)
```ini
# Only these are needed for a working demo
GROQ_API_KEY=your-groq-key   # or run a local LLM via Ollama and adapt src/llm.py
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=assignment_docs
MODEL_NAME=llama-3.1-70b-versatile
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

> If you **must** strictly use OpenWeatherMap for grading, add `OPENWEATHERMAP_API_KEY`. Otherwise the app uses Open-Meteo transparently.
