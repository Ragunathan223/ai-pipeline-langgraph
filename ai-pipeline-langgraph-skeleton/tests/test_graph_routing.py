from src.graph.agent_graph import router, AppState

def make_state(q: str) -> AppState:
    return {"history": [], "query": q, "params": {}, "context": [], "answer": ""}

def test_router_weather_keywords():
    assert router(make_state("What's the weather in Chennai?")) == "weather"
    assert router(make_state("Temperature today?")) == "weather"

def test_router_default_rag():
    assert router(make_state("Explain section 2 of my PDF.")) == "rag"
