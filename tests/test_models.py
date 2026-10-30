import numpy as np
import pytest
from src.features import build_features
from src.modeling import preprocessor,make_model,risk_scores
from src.explain import explain


@pytest.mark.parametrize('name',['logistic','boosting','isolation','lof'])
def test_models_score_orientation_and_reload(events,tmp_path,name):
    import joblib
    f=build_features(events)
    prep=preprocessor()
    X=prep.fit_transform(f.iloc[:120])
    model=make_model(name)
    if name in ['logistic','boosting']:
        model.fit(X,events.is_fraud.iloc[:120])
        expected=model.predict_proba(X)[:,list(model.classes_).index(1)]
    else:
        model.fit(X)
        expected=-model.score_samples(X)
    np.testing.assert_allclose(risk_scores(model,name,X),expected)
    path=tmp_path/'model.joblib'
    joblib.dump(model,path)
    np.testing.assert_array_equal(risk_scores(model,name,X),risk_scores(joblib.load(path),name,X))
    if name=='lof':
        assert model.novelty


def test_train_only_transform_and_imbalance(events):
    f=build_features(events)
    prep=preprocessor().fit(f.iloc[:120])
    before=prep.named_transformers_['numeric']['scale'].mean_.copy()
    changed=f.iloc[120:].copy()
    changed['amt']=1e12
    prep.transform(changed)
    np.testing.assert_array_equal(before,prep.named_transformers_['numeric']['scale'].mean_)
    assert make_model('logistic').class_weight=='balanced'
    assert make_model('boosting',False).class_weight is None
    assert len(prep.transform(f.iloc[120:]))==80


@pytest.mark.parametrize('name',['logistic','boosting'])
def test_shap_additivity_and_one_hot_aggregation(events,name):
    f=build_features(events)
    prep=preprocessor()
    X=prep.fit_transform(f.iloc[:120])
    model=make_model(name).fit(X,events.is_fraud.iloc[:120])
    exp=explain(dict(name=name,model=model,preprocessor=prep,background=X[:50]),f.iloc[120:123])
    raw=np.array(exp['base_values'])+np.array(exp['values']).sum(axis=1)
    np.testing.assert_allclose(raw,exp['raw_output'],atol=1e-5)
    np.testing.assert_allclose(1/(1+np.exp(-raw)),model.predict_proba(prep.transform(f.iloc[120:123]))[:,1],atol=1e-5)
    assert len(exp['names'])==11
