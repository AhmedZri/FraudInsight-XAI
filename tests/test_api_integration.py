import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.data import ROOT,load_source,chronological_split
from src.features import build_features,FEATURES
from src.registry import load_bundle,manifest

client=TestClient(app)


def raw_event():
    return dict(trans_date_trans_time='2020-01-01T00:00:00Z',cc_num='test',amt=100.,category='grocery_pos')


def test_health_and_openapi():
    assert client.get('/health').json()['service']=='fraudinsight-xai'
    assert '/score' in client.get('/openapi.json').json()['paths']


def test_strict_schema_and_no_sensitive_error_echo():
    body=dict(model_id='logistic',transaction={**raw_event(),'secret':'ne_pas_renvoyer'})
    response=client.post('/score',json=body)
    assert response.status_code==422
    assert 'ne_pas_renvoyer' not in response.text


def test_history_future_rejected():
    body=dict(model_id='logistic',transaction=raw_event(),history=[raw_event()])
    assert client.post('/score',json=body).status_code==422


@pytest.mark.integration
def test_authentic_artifacts_api_and_raw_path():
    assert (ROOT/'reports/results.json').is_file(),'Exécuter scripts/train.py sur les données réelles.'
    assert set(manifest()['models'])=={'logistic','boosting','isolation','lof'}
    body=dict(model_id='logistic',transaction=raw_event())
    response=client.post('/score',json=body)
    assert response.status_code==200,response.text
    df=pd.DataFrame([raw_event()])
    df['trans_date_trans_time']=pd.to_datetime(df.trans_date_trans_time,utc=True)
    feature=build_features(df)
    rows=json.loads(feature.to_json(orient='records'))
    engineered=client.post('/score/features',json={'model_id':'logistic','rows':rows})
    assert engineered.status_code==200,engineered.text
    np.testing.assert_array_equal(response.json()['scores'],engineered.json()['scores'])
    explanation=client.post('/explain',json=body)
    assert explanation.status_code==200,explanation.text


@pytest.mark.integration
def test_real_demo_rescoring_all_models():
    demo=pd.read_parquet(ROOT/'data/demo/transactions.parquet')
    assert not {'cc_num','first','last','street','trans_num'}.intersection(demo)
    rows=json.loads(demo[FEATURES].iloc[:30].to_json(orient='records'))
    for name in manifest()['models']:
        response=client.post('/score/features',json={'model_id':name,'rows':rows})
        assert response.status_code==200,response.text
        np.testing.assert_allclose(response.json()['scores'],demo['score_'+name].iloc[:30],atol=1e-9)
    labelled=json.loads(demo[FEATURES+['trans_date_trans_time','is_fraud']].iloc[:30].to_json(orient='records',date_format='iso'))
    evaluation=client.post('/evaluate',json={'model_id':'boosting','rows':labelled})
    assert evaluation.status_code==200,evaluation.text
    assert len(evaluation.json()['ranking'])==30


@pytest.mark.integration
def test_real_data_split_and_prevalence():
    results=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    assert sum(p['rows'] for p in results['splits'].values())==results['rows']
    assert 0<results['prevalence']<.02
    assert results['splits']['train']['end']<results['splits']['validation']['start']
    assert results['splits']['validation']['end']<results['splits']['test']['start']
    for selection in results['selection'].values():
        assert selection['selected']['policy']['selected_on']=='validation'
