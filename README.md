# 🌦️ LangGraph RAG + Weather Agent

A Streamlit-based AI application using **LangGraph**, **LangChain**, and **LangSmith** — demonstrating **RAG (Retrieval-Augmented Generation)** over PDFs and **real-time weather** retrieval, with vector storage in **Qdrant**.

---

## 🖼️ Demo Screenshots

<img width="1849" height="1079" alt="Streamlit UI" src="https://github.com/user-attachments/assets/8365b6a7-2eb1-4a79-9440-b491e7e3afd8" />

<img width="1849" height="1079" alt="LangSmith Tracing Dashboard" src="https://github.com/user-attachments/assets/d6680e80-957c-41ac-9f0b-a55e99b7dc18" />

---

## 🚀 Features
- **RAG over PDFs** (chunking + embeddings stored in Qdrant)
- **Live Weather** via OpenWeatherMap (auto-fallback to Open-Meteo if no key)
- **Agentic pipeline (LangGraph)**: combines PDF answer + weather every turn
- **LLM via LangChain** (Groq/OpenAI, configurable)
- **LangSmith** tracing + optional dataset logging (auto-creates dataset)
- **Streamlit** chat UI with PDF upload & neat answer layout
- **Unit tests** for router/RAG/weather logic (pytest)

---

## ⚙️ Setup Instructions

```bash
# Clone repo
git clone https://github.com/<your-username>/ai-pipeline-langgraph-assignment.git
cd ai-pipeline-langgraph-assignment

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate (Mac/Linux)

# Install dependencies
pip install -r requirements.txt
