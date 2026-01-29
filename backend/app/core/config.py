"""Configuration management for the application."""
from pydantic_settings import BaseSettings
from pydantic import model_validator
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Neo4j Configuration
    NEO4J_URI: str
    NEO4J_USER: Optional[str] = None
    NEO4J_USERNAME: Optional[str] = None  # Alternative name for compatibility (Neo4j Aura uses this)
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"
    
    @model_validator(mode='after')
    def set_neo4j_user(self):
        """Set NEO4J_USER from NEO4J_USERNAME if needed."""
        if not self.NEO4J_USER and self.NEO4J_USERNAME:
            object.__setattr__(self, 'NEO4J_USER', self.NEO4J_USERNAME)
        if not self.NEO4J_USER:
            object.__setattr__(self, 'NEO4J_USER', "neo4j")
        return self
    
    # Together AI Configuration (OpenAI-compatible)
    TOGETHER_API_KEY: str
    TOGETHER_BASE_URL: str = "https://api.together.xyz/v1"
    EMBED_MODEL: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    CHAT_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    
    # OpenAI Configuration (optional, for embeddings if not using Together)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Application Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # RAG Configuration
    TOP_K_RESULTS: int = 5
    EMBEDDING_DIMENSION: int = 768
    BATCH_SIZE: int = 128
    
    class Config:
        # Load .env from backend directory
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields like AURA_INSTANCEID


# Global settings instance
settings = Settings()

