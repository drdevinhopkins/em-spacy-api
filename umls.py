import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os


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


def search_by_atom(search_term, tgt):
    st = get_st(tgt)
    x = requests.get(
        f'https://uts-ws.nlm.nih.gov/rest/search/current?string={search_term}&ticket={st}')
    results = json.loads(x.text)['result']['results']
    results_df = pd.DataFrame(results)
    return results_df


def search_by_cui(cui, tgt):
    st = get_st(tgt)
    x = requests.get(
        f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}?ticket={st}')
    result = json.loads(x.text)['result']
    return result


def cui_to_atoms(cui, tgt):
    st = get_st(tgt)
    x = requests.get(
        f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms?ticket={st}')
    result = json.loads(x.text)['result']
    result_df = pd.DataFrame(result)
    result_df_eng = result_df[result_df.language == 'ENG']
    return result_df_eng.name.unique().tolist()


def atom_to_cui(search_term, tgt):
    st = get_st(tgt)
    x = requests.get(
        f'https://uts-ws.nlm.nih.gov/rest/search/current?string={search_term}&ticket={st}')
    results = json.loads(x.text)['result']['results']
    return results[0]['ui']
