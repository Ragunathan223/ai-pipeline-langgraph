from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # LLM
    MODEL_NAME: str = Field(default="gpt-4o-mini")
    EMBEDDINGS_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # LangSmith
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str | None = None

    # Weather
    OPENWEATHERMAP_API_KEY: str | None = None

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = Field(default="assignment_docs")

    # App
    PORT: int = Field(default=8501)

    class Config:
        env_file = ".env"
        extra = "ignore"
