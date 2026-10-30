"""Toutes les inférences Streamlit passent réellement par HTTP."""
import os
import requests


def api_call(path,body=None):
    url=os.environ.get('FRAUDINSIGHT_API_URL','http://127.0.0.1:8766').rstrip('/')+path
    with requests.Session() as session:
        session.trust_env=False
        response=session.get(url,timeout=120) if body is None else session.post(url,json=body,timeout=120)
        response.raise_for_status()
        return response.json()
