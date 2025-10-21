A Streamlit-based AI application using **LangGraph**, **LangChain**, and **LangSmith** —  
demonstrating RAG (Retrieval-Augmented Generation) over PDFs and real-time weather retrieval.

---

## 🖼️ Demo Screenshot 
# 🌦️ LangGraph RAG + Weather Agent

<img width="1849" height="1079" alt="image" src="https://github.com/user-attachments/assets/8365b6a7-2eb1-4a79-9440-b491e7e3afd8" />

<img width="1849" height="1079" alt="image" src="https://github.com/user-attachments/assets/d6680e80-957c-41ac-9f0b-a55e99b7dc18" />

---
 
## 🚀 Features
- **RAG over PDFs** (indexed in Qdrant)
- **Real-time Weather Info** via OpenWeatherMap
- **Combined Responses** (both weather + document context)
- **LangSmith Integration** for tracing and evaluation
- Clean modular code following best practices

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
