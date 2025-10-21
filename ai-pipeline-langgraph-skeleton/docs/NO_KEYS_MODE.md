# Running Without OpenAI or OpenWeatherMap Keys

- Use **Groq** for LLM: set `GROQ_API_KEY` in `.env` (free tier often available).
- Weather: if `OPENWEATHERMAP_API_KEY` is absent, app automatically uses **Open-Meteo** (no key).
- Embeddings + Qdrant: fully local, no keys required.
