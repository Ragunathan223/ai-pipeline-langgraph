# app/streamlit_app.py
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure project root (contains `src/`) is on PYTHONPATH
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()
st.set_page_config(page_title="LangGraph RAG + Weather", page_icon="⛅", layout="wide")

from src.graph.agent_graph import build_graph, AppState  # noqa: E402
from src.config import Settings  # noqa: E402
from src.weather.api import fetch_weather  # noqa: E402

settings = Settings()  # reads from .env

if "graph" not in st.session_state:
    st.session_state.graph = build_graph(settings)
if "history" not in st.session_state:
    st.session_state.history = []

st.title("LangGraph Agent: Weather + PDF RAG")
st.caption("Each answer includes: (1) PDF-based answer, (2) current weather for your city (if provided).")

if not os.getenv("OPENWEATHERMAP_API_KEY"):
    st.info("Weather uses **Open-Meteo fallback** when no OpenWeatherMap key is set.")

with st.sidebar:
    st.header("Controls")
    upload = st.file_uploader("Upload a PDF to index", type=["pdf"])
    if upload:
        from src.rag.index import index_pdf_into_qdrant  # noqa: E402
        with st.spinner("Indexing PDF into Qdrant..."):
            index_pdf_into_qdrant(upload, settings)
        st.success("Indexed!")

    city = st.text_input("City for weather", value="", placeholder="Type a city (e.g., Chennai)", key="city")
    fetch_disabled = not (city and city.strip())

    # Result placeholder (appears right under the button)
    weather_placeholder = st.empty()
    if st.button("Fetch weather now", disabled=fetch_disabled):
        try:
            w = fetch_weather(city, settings.OPENWEATHERMAP_API_KEY)
            feels = getattr(w, "feels_like_c", None)
            hum = getattr(w, "humidity_pct", None)
            wind = getattr(w, "wind_kph", None)
            details = []
            if isinstance(feels, (int, float)):
                details.append(f"feels like {feels:.1f}°C")
            if isinstance(hum, (int, float)):
                details.append(f"humidity {hum:.0f}%")
            if isinstance(wind, (int, float)):
                details.append(f"wind {wind:.0f} km/h")
            extra = (" | " + ", ".join(details)) if details else ""
            weather_placeholder.success(f"{w.city}: {w.description}, {w.temperature_c:.1f}°C (via {w.provider}){extra}")
        except Exception as e:
            weather_placeholder.error(f"Weather lookup failed: {e}")

    # ---------- LangSmith status (safe; informational only) ----------
    st.divider()
    st.caption("LangSmith")
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("1", "true", "yes")
    ds_log_on = os.getenv("LANGSMITH_LOG_EXAMPLES", "0") == "1"
    project = os.getenv("LANGCHAIN_PROJECT", "ai-pipeline-assignment")
    st.write(f"Tracing: {'ON' if tracing_on else 'OFF'}")
    st.write(f"Dataset logs: {'ON' if ds_log_on else 'OFF'}")
    st.write(f"Project: {project}")

st.subheader("Chat")
user_msg = st.chat_input("Ask something about your PDF (weather will be included if you provided a city)...")
if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})

    state = AppState(
        history=st.session_state.history,
        query=user_msg,
        params={"city": st.session_state.get("city")},  # may be blank; graph still returns both sections
        context=[],
        answer=""
    )

    # Always append a response, even if something fails.
    try:
        result = st.session_state.graph.invoke(state)
        assistant_content = (result.get("answer") or "").strip() or "_No answer generated._"
        assistant_context = result.get("context", [])
    except Exception as e:
        # LangSmith telemetry is fully no-throw now, but keep this guard anyway
        assistant_content = f"Sorry — something went wrong while answering: {e}"
        assistant_context = []

    st.session_state.history.append({
        "role": "assistant",
        "content": assistant_content,
        "context": assistant_context
    })

# Render chat; assistant replies get a neat two-column layout
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and "## PDF Answer" in m["content"]:
            parts = m["content"].split("## Current Weather")
            pdf_answer = parts[0].replace("## PDF Answer", "").strip()
            weather_answer = parts[1].strip() if len(parts) > 1 else ""

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### PDF Answer")
                st.write(pdf_answer)
                if m.get("context"):
                    with st.expander("Retrieved context"):
                        for i, chunk in enumerate(m["context"], 1):
                            st.markdown(f"**Snippet {i}:**")
                            st.write(chunk)
            with col2:
                st.markdown("### Current Weather")
                if weather_answer:
                    st.write(weather_answer)
                else:
                    st.write("No city provided. Type a city to include live weather.")
        else:
            st.write(m["content"])
