from fastapi import APIRouter
from collections import defaultdict
from typing import Dict, List
from pydantic import BaseModel
import spacy
import os

# os.system(command="!pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl")

nlp = spacy.load("en_legal_ner_trf")

router = APIRouter()


class EntityRequest(BaseModel):
    raw_text: str


class EntityResponse(BaseModel):
    entitities: Dict[str, List[str]]


@router.post("/", response_model=EntityResponse)
def get_entities(request: EntityRequest):
    raw_text = request.raw_text
    doc = nlp(raw_text)

    res = defaultdict(list)

    for ent in doc.ents:
        res[ent.label_].append(ent)

    return EntityResponse(entitities=res)
