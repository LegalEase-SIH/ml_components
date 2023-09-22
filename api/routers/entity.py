import copy
from typing import List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pinecone
from sentence_transformers import SentenceTransformer

from api.config import get_settings
from api.schema import EntitySchema


settings = get_settings()


model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT)
index = pinecone.Index(settings.PINECONE_INDEX)