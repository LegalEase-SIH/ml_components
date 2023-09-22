import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings
import pathlib


BASE_DIR = pathlib.Path(__file__).parent.parent
ENV_FILE = str(BASE_DIR / ".env")

print(BASE_DIR)

class Settings(BaseSettings):
    PINECONE_API_KEY: str  = Field(..., env='PINECONE_API_KEY')
    PINECONE_ENVIRONMENT: str = Field(..., env='PINECONE_ENVIRONMENT')
    PINECONE_INDEX: str = Field(..., env='PINECONE_INDEX')
    OPENAI_API_KEY: str = Field(..., env='OPENAI_API_KEY')

    class Config:
        env_file = ENV_FILE


@lru_cache
def get_settings():
    return Settings()
