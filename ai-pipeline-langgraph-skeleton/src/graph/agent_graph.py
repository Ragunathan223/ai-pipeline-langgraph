# src/graph/agent_graph.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_chat_model
from ..config import Settings
from ..weather.api import fetch_weather
from ..rag.index import rag_retrieve
from ..eval.langsmith_eval import record_eval


class AppState(TypedDict):
    history: List[Dict[str, Any]]
    query: str
    params: Dict[str, Any]   # {"city": "..."}; may be blank
    context: List[str]
    answer: str


def _safe_eval(inputs: Dict[str, Any], outputs: Dict[str, Any], run_name: str) -> None:
    """Never let telemetry affect UX."""
    try:
        record_eval(inputs, outputs, run_name=run_name)
    except Exception:
        pass


def _extractive_fallback(docs: List[str], max_chars: int = 700) -> str:
    """LLM-free fallback for reliability."""
    if not docs:
        return "I couldn't find anything relevant in the uploaded PDF."
    out, total = [], 0
    for i, d in enumerate(docs[:3], 1):
        snippet = d.strip().replace("\n\n", "\n")
        if total + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - total)]
        out.append(f"Snippet {i}:\n{snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return (
        "I couldn't run the LLM. Here are the most relevant excerpts:\n\n"
        + "\n\n---\n\n".join(out)
    )


def both_node(state: AppState, settings: Settings) -> AppState:
    """Main logic: always return PDF RAG + Weather."""
    city = (state.get("params") or {}).get("city", "").strip()

    # ---------- RAG ----------
    rag_ctx: List[str] = []
    rag_answer = ""
    try:
        docs = rag_retrieve(state["query"], settings, k=5)
        if docs:
            rag_ctx = docs
            try:
                llm = get_chat_model(settings.MODEL_NAME)
                context = "\n\n".join(docs)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful RAG assistant. Answer using ONLY the provided context. "
                               "If the answer isn't in the context, say you don't know."),
                    ("human", "Question: {q}\n\nContext:\n{ctx}\n\nAnswer (cite which snippet you used if helpful):")
                ])
                chain = prompt | llm | StrOutputParser()
                # Add tags/metadata so traces look nice in LangSmith
                rag_answer = (chain.invoke(
                    {"q": state["query"], "ctx": context},
                    config={
                        "tags": ["assignment", "rag", "both-node"],
                        "metadata": {"route": "both", "city": city or None},
                    }
                ) or "").strip()
            except Exception:
                rag_answer = ""
            if not rag_answer:
                rag_answer = _extractive_fallback(docs)
        else:
            rag_answer = "I couldn't find anything relevant in the uploaded PDF."
    except Exception as e:
        rag_answer = f"RAG failed: {e}"

    # ---------- WEATHER ----------
    weather_block = ""
    try:
        if city:
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
            weather_block = f"{w.city}: {w.description}, {w.temperature_c:.1f}°C (via {w.provider}){extra}"
        else:
            weather_block = "No city provided. Include a city in your question to show live weather."
    except Exception as e:
        weather_block = f"Weather lookup failed: {e}"

    # ---------- COMBINE ----------
    combined = (
        "## 📘 PDF Answer\n"
        f"{rag_answer}\n\n"
        "## 🌤️ Current Weather\n"
        f"{weather_block}"
    )

    state["context"] = rag_ctx if rag_ctx else []
    state["answer"] = combined

    # Best-effort dataset example (uses LANGSMITH_LOG_EXAMPLES)
    _safe_eval(
        {"query": state["query"], "city": city, "found_docs": bool(rag_ctx)},
        {"answer": combined},
        run_name="both-node",
    )
    return state


def build_graph(settings: Settings):
    """Single-node graph: always do both."""
    graph = StateGraph(AppState)
    graph.add_node("both", lambda s: both_node(s, settings))
    graph.set_entry_point("both")
    graph.add_edge("both", END)
    return graph.compile()
