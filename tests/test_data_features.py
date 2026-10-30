import sys
import types
import numpy as np
import pandas as pd
import pytest
from src.data import validate, chronological_split, acquire
from src.features import build_features, FEATURES


def test_acquire_uses_public_version_and_cache(tmp_path,monkeypatch):
    for name in ['fraudTrain.csv','fraudTest.csv']:
        (tmp_path/name).write_text('test',encoding='utf-8')
    calls=[]
    def download(source):
        calls.append(source)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules,'kagglehub',types.SimpleNamespace(dataset_download=download))
    assert acquire()==acquire()
    assert calls==['kartik2112/fraud-detection/versions/1']*2
    import os
    assert os.environ['KAGGLEHUB_CACHE'].endswith('data'+str(__import__('os').sep)+'cache')


@pytest.mark.parametrize('column,value',[('amt',-1),('amt',np.inf),('is_fraud',2),('trans_date_trans_time','invalide')])
def test_validation_rejects(events,column,value):
    events[column]=events[column].astype(object)
    events.loc[0,column]=value
    with pytest.raises(ValueError):
        validate(events)


def test_mapping_and_order(events):
    custom=events.rename(columns={'is_fraud':'cible','trans_date_trans_time':'date'}).iloc[::-1]
    result=validate(custom,'cible','date')
    assert result.trans_date_trans_time.is_monotonic_increasing
    assert result.is_fraud.sum()==events.is_fraud.sum()


def test_split_60_20_20(events):
    parts=chronological_split(events)
    assert [len(p) for p in parts]==[120,40,40]
    assert events.iloc[parts[0][-1]].trans_date_trans_time<events.iloc[parts[1][0]].trans_date_trans_time


def test_ties_cannot_cross_split(events):
    events.loc[119:121,'trans_date_trans_time']=events.loc[120,'trans_date_trans_time']
    a,b,c=chronological_split(events)
    assert events.iloc[a[-1]].trans_date_trans_time<events.iloc[b[0]].trans_date_trans_time
    assert set(np.r_[a,b,c])==set(range(len(events)))


def test_strict_windows_and_ties():
    df=validate(pd.DataFrame(dict(trans_date_trans_time=['2020-01-01 00:00','2020-01-01 00:00',
        '2020-01-01 01:00','2020-01-02 00:00','2020-02-03 00:00'],
        amt=[10,30,40,80,10],cc_num=['a']*5,category=['x']*5,is_fraud=[0]*5)))
    f=build_features(df)
    assert f.count_1h.tolist()==[0,0,2,0,0]
    assert f.count_24h.tolist()==[0,0,2,3,0]
    assert f.amount_ratio_30d.iloc[2]==2
    assert f.seconds_since_previous.iloc[2]==3600
    assert f.history_missing.tolist()==[1,1,0,0,0]
    assert np.isnan(f.amount_ratio_30d.iloc[-1])


def test_future_events_labels_and_current_amount_do_not_change_history(events):
    original=build_features(events)
    modified=events.copy()
    modified.loc[101:,'amt']=999999
    modified['is_fraud']=1-modified.is_fraud
    again=build_features(modified)
    pd.testing.assert_frame_equal(original.iloc[:101],again.iloc[:101])
    modified=events.copy()
    modified.loc[100,'amt']=100000
    f=build_features(modified)
    assert f.count_1h.iloc[100]==original.count_1h.iloc[100]
    assert f.amount_ratio_30d.iloc[100]/original.amount_ratio_30d.iloc[100]==pytest.approx(100000/101)


def test_prefix_invariance_and_identity_exclusion(events):
    full=build_features(events)
    for boundary in [120,160,175]:
        pd.testing.assert_frame_equal(full.iloc[:boundary],build_features(events.iloc[:boundary]))
    assert set(full)==set(FEATURES)
    assert not {'cc_num','is_fraud','trans_num','first','last'}.intersection(full.columns)


def test_zero_history_new_card_missing_coordinates():
    df=validate(pd.DataFrame(dict(trans_date_trans_time=['2020-01-01','2020-01-02','2020-01-03'],
        amt=[0,10,20],category=['x']*3,is_fraud=[0]*3,cc_num=['a','a','b'])))
    f=build_features(df)
    assert f.amount_ratio_30d.isna().all()
    assert f.history_missing.tolist()==[1,0,1]
    assert f.distance_km.isna().all()


def test_missing_card_is_not_shared_history(events):
    df=events.drop(columns='cc_num')
    f=build_features(validate(df))
    assert (f.count_24h==0).all()
