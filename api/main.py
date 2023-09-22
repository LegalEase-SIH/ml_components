from fastapi import FastAPI
from routers.chat import router as chat_router
from routers.entity import router as ner_router

app = FastAPI()

app.include_router(router=chat_router, prefix='/ml/chat', tags=['chat-api'])
app.include_router(router=ner_router, prefix='/ml/ner', tags=['ner-api'])
