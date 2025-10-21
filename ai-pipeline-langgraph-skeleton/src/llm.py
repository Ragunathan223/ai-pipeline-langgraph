# src/llm.py
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

GROQ_DEFAULT_PRIMARY = "llama-3.3-70b-versatile"
GROQ_DEFAULT_FALLBACK = "llama-3.1-8b-instant"
DECOMMISSIONED = {
    "llama-3.1-70b-versatile": GROQ_DEFAULT_PRIMARY,
    "llama3-70b-8192": GROQ_DEFAULT_PRIMARY,
}

def _normalize_model(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return GROQ_DEFAULT_PRIMARY
    return DECOMMISSIONED.get(name, name)

def get_chat_model(model_name: Optional[str] = None) -> BaseChatModel:
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    requested = _normalize_model(model_name or os.getenv("MODEL_NAME", GROQ_DEFAULT_PRIMARY))

    if groq_key:
        try:
            return ChatGroq(model=requested, temperature=0.2)
        except Exception:
            return ChatGroq(model=GROQ_DEFAULT_PRIMARY, temperature=0.2)

    if openai_key:
        return ChatOpenAI(model=requested, temperature=0.2)

    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY.\n"
        "Alternatively, adapt src/llm.py for a local LLM (e.g., Ollama)."
    )
