import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score, matthews_corrcoef
from src.evaluation import financial_cost,select_threshold,evaluate,daily_topk


def fixture():
    return np.array([0,1,0,1]),np.array([.1,.9,.8,.2]),np.array([5,100,10,40]),pd.date_range('2020-01-01',periods=4,freq='h',tz='UTC')


def test_financial_cost():
    y,s,a,t=fixture()
    assert financial_cost(y,a,[0,1,1,0],5,1)==50
    assert financial_cost(y,a,[0,1,1,0],5,.5)==100
    assert financial_cost(y,a,[0]*4)==140


def test_top_k_daily():
    y,s,a,t=fixture()
    assert daily_topk(s,t,2).tolist()==[False,True,True,False]
    p=select_threshold(y,s,a,t,capacity=2)
    m=evaluate(y,s,a,t,p)
    assert m['precision_at_k']==.5
    assert m['recall_at_k']==.5
    assert m['k']==2


def test_metrics_match_definitions():
    y,s,a,t=fixture()
    p=select_threshold(y,s,a,t,capacity=2)
    m=evaluate(y,s,a,t,p)
    pred=s>=p['threshold']
    assert m['average_precision']==average_precision_score(y,s)
    assert m['roc_auc']==roc_auc_score(y,s)
    assert m['f2']==fbeta_score(y,pred,beta=2)
    assert m['mcc']==matthews_corrcoef(y,pred)
    assert np.sum(m['confusion'])==4


def test_threshold_exact_cost_and_daily_capacity():
    y,s,a,t=fixture()
    p=select_threshold(y,s,a,t,capacity=2)
    feasible=[]
    for threshold in [np.nextafter(s.max(),np.inf),*np.unique(s)]:
        alert=s>=threshold
        if alert.sum()<=2:
            feasible.append(financial_cost(y,a,alert))
    assert p['cost']==min(feasible)
    assert p['threshold']==.9


def test_score_ties_atomic_and_capacity_zero():
    y,s,a,t=fixture()
    s[:]=.5
    p=select_threshold(y,s,a,t,capacity=2)
    assert p['alerts']==0
    p=select_threshold(y,s,a,t,capacity=0)
    assert p['alerts']==0


def test_threshold_rejects_holdout_and_evaluation_does_not_retune():
    y,s,a,t=fixture()
    with pytest.raises(ValueError):
        select_threshold(y,s,a,t,partition='test')
    p=select_threshold(y,s,a,t)
    original=p.copy()
    evaluate(1-y,s,a,t,p)
    assert p==original


def test_capacity_is_per_day():
    y,s,a,t=fixture()
    t=pd.date_range('2020-01-01',periods=4,freq='D',tz='UTC')
    p=select_threshold(y,s,a,t,capacity=1)
    assert p['alerts']>1
    assert not evaluate(y,s,a,t,p)['capacity_exceeded']


def test_exploratory_capacity_changes_alerts_without_changing_policy():
    y,s,a,t=fixture()
    p=select_threshold(y,s,a,t)
    p.update(threshold=0.,capacity=1)
    before=p.copy()
    uncapped=evaluate(y,s,a,t,p)
    capped=evaluate(y,s,a,t,p,apply_capacity=True)
    assert uncapped['alerts']==4
    assert capped['alerts']==1
    assert capped['cost']==financial_cost(y,a,[0,1,0,0])
    assert p==before
