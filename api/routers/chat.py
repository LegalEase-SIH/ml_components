import os
from typing import List, Dict

import pinecone
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
import openai

load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)
index = pinecone.Index(os.environ['PINECONE_INDEX'])
model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_URL = "https://indiankanoon.org/search/?formInput="
router = APIRouter()


class cfg:
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: int = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0
    stop_sequence = None
    model: str = "gpt-3.5-turbo"


class SearchRequest(BaseModel):
    context: str
    top_k: int = 10


class SearchResponse(BaseModel):
    title: str
    text: str
    url: str


class ChatRequest(BaseModel):
    current_query: str
    previous_query: List[Dict[str, str]]


class ChatResponse(BaseModel):
    assistant_res: str


def get_search_results(context: str, top_k: int = 10):
    embeddings = model.encode(context).tolist()
    response = index.query(
        vector=embeddings,
        include_metadata=True,
        top_k=top_k,
    )
    res = response['matches']
    return res


@router.post('/openai', response_model=ChatResponse)
def get_chat_response(request: ChatRequest):
    print(request)
    print(request.current_query, request.previous_query)
    res = get_search_results(context=request.current_query, top_k=2)
    context = ''
    for _item in res:
        item = _item['metadata']
        # print(item)
        if 'Act ID' in item.keys():
            context += f'{item["Act ID"]}, {item["Act Title"]}, {item["text"]}\n\n'
        else:
            context += f'{item["section"]}, {item["text"]}\n\n'

    PROMPT = f'As an unbiased Legal Assistant for the Indian Judiciary, answer the following question, honestly without any bias.'

    try:
        MESSAGES = [{"role": "system", "content": PROMPT}]
        for chats in request.previous_query:
            print(chats)
            MESSAGES.append({"role": "user", "content": chats['userQuestion']})
            MESSAGES.append({"role": "assistant", "content": chats['reply']})

        response = openai.ChatCompletion.create(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            model=cfg.model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Using the following context:\n{context}\nAnswer the following question: {request.current_query}"}
            ]
        )
        print(response['choices'][0]['message']['content'])
        return ChatResponse(assistant_res=response['choices'][0]['message']['content'])
    except Exception as e:
        print(e)
        return JSONResponse(content={"msg": "something wrong with openai api..."}, status_code=500)


@router.post('/search', response_model=List[SearchResponse])
def get_relevent_documents(request: SearchRequest):
    res = get_search_results(context=request.context, top_k=request.top_k)
    response = []
    for _item in res:
        item = _item['metadata']
        # print(item)
        if 'Act ID' in item.keys():
            "THE+INSURANCE+ACT%2C+1938+-+Section+64V."
            response.append(SearchResponse(
                title=item['Act ID'], text=item['Act Title'] + ' ' + item['text'], url=BASE_URL+item['Act ID'].replace(' ', '+')))
        else:
            response.append(SearchResponse(
                title=item['section'], text=item['text'], url=BASE_URL+item['section'].replace(' ', '+')))
    return response
