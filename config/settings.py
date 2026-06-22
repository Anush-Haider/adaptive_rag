import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Self-Correcting RAG Agent"
    DEBUG: bool = True

    # Provider configuration: "ollama" or "groq"
    LLM_PROVIDER: str = "ollama"

    # Model Configurations
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama3-8b-8192"

    # Vector DB Config
    CHROMA_PERSIST_DIR: str = "./.chroma_db"
    EMBEDDING_MODEL: str = "nallama"  # local embedding or huggingface open-source

    # Fallback Tool Configurations
    MAX_SEARCH_RESULTS: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()