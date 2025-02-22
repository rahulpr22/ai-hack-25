from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API Keys and Environment
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "imagine-dragons"
    PINECONE_ENVIRONMENT: str = "aws-starter"
    PERPLEXITY_API_KEY: str
    
    # Web Scraping Configuration
    ALLOWED_DOMAINS: List[str] = [
        "cardekho.com",
        "carwale.com",
        "autocarindia.com",
        "carandbike.com",
        "zigwheels.com"
    ]
    
    SCRAPING_DELAY: int = 2  # Delay between requests in seconds
    MAX_RETRIES: int = 3
    USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]
    
    # Data Processing
    CHUNK_SIZE: int = 500  # Size of text chunks for embedding
    CHUNK_OVERLAP: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    return Settings() 