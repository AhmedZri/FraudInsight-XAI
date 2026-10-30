"""Service HTTP local : schémas fermés, registre contrôlé, chemin ML unique."""
import asyncio
from typing import Literal
import json
import logging
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime
from src.data import ROOT, validate
from src.features import FEATURES, build_features
from src.registry import manifest, load_bundle
from src.modeling import risk_scores, SUPERVISED
from src.explain import explain
from src.evaluation import evaluate

ModelID=Literal['logistic','boosting','isolation','lof']
app=FastAPI(title='FraudInsight XAI',description='API interne de détection explicable. Montants en USD.',version='1.0.0')
_scoring_gate=asyncio.Semaphore(1)


@app.middleware('http')
async def bound_concurrent_scoring(request: Request,call_next):
    # Évite plusieurs gros calculs simultanés sur un hébergement contraint.
    if request.method=='POST':
        async with _scoring_gate:
            return await call_next(request)
    return await call_next(request)


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)


class FeatureRow(Strict):
    amt: float=Field(ge=0)
    hour_sin: float=Field(ge=-1,le=1)
    hour_cos: float=Field(ge=-1,le=1)
    day_of_week: float=Field(ge=0,le=6)
    distance_km: float | None=Field(default=None,ge=0,le=20100)
    count_1h: float=Field(ge=0)
    count_24h: float=Field(ge=0)
    amount_ratio_30d: float | None=Field(default=None,ge=0)
    seconds_since_previous: float | None=Field(default=None,ge=0)
    history_missing: Literal[0,1]
    category: str=Field(min_length=1,max_length=100)


class FeatureRequest(Strict):
    model_id: ModelID
    rows: list[FeatureRow]=Field(min_length=1,max_length=5000)


class Transaction(Strict):
    trans_date_trans_time: AwareDatetime
    cc_num: str | None=Field(default=None,max_length=80)
    amt: float=Field(ge=0)
    category: str=Field(min_length=1,max_length=100)
    lat: float | None=Field(default=None,ge=-90,le=90)
    long: float | None=Field(default=None,ge=-180,le=180)
    merch_lat: float | None=Field(default=None,ge=-90,le=90)
    merch_long: float | None=Field(default=None,ge=-180,le=180)


class RawRequest(Strict):
    model_id: ModelID
    transaction: Transaction
    history: list[Transaction]=Field(default_factory=list,max_length=20000)


class EvaluationRow(FeatureRow):
    trans_date_trans_time: AwareDatetime
    is_fraud: Literal[0,1]


class EvaluationRequest(Strict):
    model_id: ModelID
    rows: list[EvaluationRow]=Field(min_length=1,max_length=5000)
    threshold: float | None=None
    capacity: int=Field(default=50,ge=0,le=1000)
    investigation: float=Field(default=5.,ge=0)
    effectiveness: float=Field(default=1.,ge=0,le=1)
    apply_capacity: bool=False


def safe_bundle(model_id):
    try:
        return load_bundle(model_id)
    except ValueError as exc:
        raise HTTPException(404,'Modèle absent ou artefact invalide.') from exc


def score_features(model_id,frame):
    bundle=safe_bundle(model_id)
    X=bundle['preprocessor'].transform(frame[FEATURES])
    scores=risk_scores(bundle['model'],model_id,X)
    return dict(model_id=model_id,score_kind=manifest()['models'][model_id]['score_kind'],
                scores=scores.tolist(),alerts=(scores>=bundle['policy']['threshold']).tolist(),
                threshold=bundle['policy']['threshold'],history='Variables fournies par le pipeline canonique ou le client.')


def raw_features(body):
    event=body.transaction.model_dump()
    history=[item.model_dump() for item in body.history]
    if any(item['trans_date_trans_time']>=event['trans_date_trans_time'] for item in history):
        raise HTTPException(422,'Historique requis strictement antérieur à la décision.')
    df=validate(pd.DataFrame(history+[event]),labelled=False)
    return build_features(df).iloc[[-1]]


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request,exc):
    # Ne jamais renvoyer les entrées (identifiants possibles) dans les erreurs.
    return JSONResponse(status_code=422,content={'detail':'Requête invalide : consulter le schéma OpenAPI.'})


@app.exception_handler(Exception)
async def internal_error(request: Request,exc):
    logging.getLogger(__name__).error('Erreur de service : %s',type(exc).__name__)
    return JSONResponse(status_code=500,content={'detail':'Erreur interne ; vérifier les artefacts locaux.'})


@app.get('/health')
def health():
    return dict(status='ok',service='fraudinsight-xai',artifact_ready=(ROOT/'artifacts/manifest.json').is_file())


@app.get('/models')
def models():
    return dict(**manifest(),features=FEATURES,schemas='/openapi.json',currency='USD',higher_is_riskier=True)


@app.post('/score')
def score(body: RawRequest):
    result=score_features(body.model_id,raw_features(body))
    result['history']='Historique fourni seulement ; état serveur absent. Sans historique : carte nouvelle.'
    return result


@app.post('/score/features')
def score_engineered(body: FeatureRequest):
    return score_features(body.model_id,pd.DataFrame([row.model_dump() for row in body.rows]))


@app.post('/explain')
def explain_raw(body: RawRequest):
    if body.model_id not in SUPERVISED:
        raise HTTPException(422,'Explication SHAP disponible pour logistic et boosting.')
    frame=raw_features(body)
    return dict(**score_features(body.model_id,frame),explanation=explain(safe_bundle(body.model_id),frame))


@app.post('/explain/features')
def explain_engineered(body: FeatureRequest):
    if body.model_id not in SUPERVISED or len(body.rows)>200:
        raise HTTPException(422,'SHAP : modèle supervisé et 200 observations maximum.')
    frame=pd.DataFrame([row.model_dump() for row in body.rows])
    return dict(**score_features(body.model_id,frame),explanation=explain(safe_bundle(body.model_id),frame))


@app.get('/reports')
def reports():
    return json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))


@app.post('/evaluate')
def evaluate_sample(body: EvaluationRequest):
    frame=pd.DataFrame([row.model_dump() for row in body.rows])
    scored=score_features(body.model_id,frame)
    policy=dict(safe_bundle(body.model_id)['policy'])
    policy.update(capacity=body.capacity,investigation=body.investigation,effectiveness=body.effectiveness)
    if body.threshold is not None:
        policy['threshold']=body.threshold
    metrics=evaluate(frame.is_fraud,scored['scores'],frame.amt,frame.trans_date_trans_time,policy,apply_capacity=body.apply_capacity)
    return dict(metrics=metrics,ranking=np.argsort(-np.array(scored['scores']),kind='stable').tolist(),
                notice='Exploration du lot fourni, sans sélection ni modification du seuil validé.')


@app.get('/demo')
def demo():
    # Données bornées pré-calculées, sans cartes, noms ni coordonnées.
    df=pd.read_parquet(ROOT/'data/demo/transactions.parquet')
    return json.loads(df.to_json(orient='records',date_format='iso'))
