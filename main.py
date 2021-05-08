import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_sci_sm", disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "parser"])

app = FastAPI()


class UserRequestIn(BaseModel):
    text: str


class EntityOut(BaseModel):
    start: int
    end: int
    type: str
    text: str


class EntitiesOut(BaseModel):
    entities: List[EntityOut]


@app.post("/entities", response_model=EntitiesOut)
def read_entities(user_request_in: UserRequestIn):
    doc = nlp(user_request_in.text)

    return {
        "entities": [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "type": ent.label_,
                "text": ent.text,
            } for ent in doc.ents
        ]
    }
