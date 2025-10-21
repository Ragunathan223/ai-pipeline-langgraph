# 🧠 LangGraph RAG + Weather Agent

![Uploading image.png…]()


A simple AI pipeline using **LangGraph**, **LangChain**, and **LangSmith**, demonstrating:
- RAG over uploaded PDF documents (stored in Qdrant)
- Live weather data from OpenWeatherMap / Open-Meteo
- Combined output in a Streamlit chat UI
- End-to-end observability via LangSmith

## 🚀 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/<your_username>/ai-pipeline-langgraph-assignment.git
cd ai-pipeline-langgraph-assignment

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# or source .venv/bin/activate (Mac/Linux)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add environment variables
Create a `.env` file:
GROQ_API_KEY=...
OPENWEATHERMAP_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=ai-pipeline-assignment
LANGSMITH_LOG_EXAMPLES=1
QDRANT_URL=http://localhost:6333

# 5. Run the app
streamlit run app/streamlit_app.py
