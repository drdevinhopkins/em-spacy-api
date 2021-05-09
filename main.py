import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import requests
from bs4 import BeautifulSoup
import os
import json
import pandas as pd

from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

from negspacy.negation import Negex
from negspacy.termsets import termset


nlp = spacy.load("en_core_sci_sm", disable=[
                 "tok2vec", "tagger", "attribute_ruler", "lemmatizer"])


ts = termset("en_clinical_sensitive")
nlp.add_pipe(
    "negex",
    config={
        "chunk_prefix": ["no", "denies"],
    },
    last=True,
)


app = FastAPI()


def get_tgt():
    url = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
    myobj = {'apikey': os.environ.get('umls_api_key')}
    response = requests.post(url, data=myobj)
    soup = BeautifulSoup(response.text, 'html.parser')
    tgt = soup.find('form').get('action')
    return tgt


def get_st(tgt):
    url = tgt
    service = 'http://umlsks.nlm.nih.gov'
    myobj = {'service': service}

    response = requests.post(url, data=myobj)
    st = response.text
    return st


def atom_to_cui(search_term, tgt):
    st = get_st(tgt)
    x = requests.get(
        f'https://uts-ws.nlm.nih.gov/rest/search/current?string={search_term}&ticket={st}')
    results = json.loads(x.text)['result']['results']
    return results[0]['ui']


tgt = get_tgt()


class UserRequestIn(BaseModel):
    text: str


class EntityOut(BaseModel):
    start: int
    end: int
    type: str
    text: str
    cui: str
    negation: bool


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
                "cui": atom_to_cui(ent.text, tgt),
                "negation": ent._.negex
            } for ent in doc.ents

        ]
    }
