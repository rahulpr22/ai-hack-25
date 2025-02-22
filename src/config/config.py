from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API Keys and Environment
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "imagine-dragons"
    PINECONE_ENVIRONMENT: str = "aws-starter"
    
    # Web Scraping Configuration
    ALLOWED_DOMAINS: List[str] = [
        "cardekho.com"
    ]
    
    SCRAPING_DELAY: int = 2  # Delay between requests in seconds
    MAX_RETRIES: int = 3
    
    # Data Processing
    CHUNK_SIZE: int = 500  # Size of text chunks for embedding
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    return Settings() 