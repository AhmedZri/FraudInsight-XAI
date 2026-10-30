"""Vérification académique indépendante : aucune fonction d'apprentissage appelée.

Les oracles ci-dessous sont exclusivement des contrôles QA, jamais un chemin
de production. Les modèles, seuils, résultats et sorties SHAP restent figés.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ.setdefault(key, '2')
from collections import deque
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data import ROOT, load_source, chronological_split
from src.features import build_features, FEATURES, NUMERIC
from src.modeling import risk_scores
from src.registry import load_bundle, manifest

OUT = ROOT / 'reports/academic_audit'


def independent_metrics(y, scores, amounts, times, threshold, capacity=50):
    """AP groupée aux ex aequo ; AUC par rangs ; F2/MCC par dénombrement."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores)
    positive, negative = int(y.sum()), int(len(y)-y.sum())
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(y)-1]
    cum_tp = np.cumsum(y[order])[ends]
    ap = float(np.sum(np.diff(np.r_[0, cum_tp]) * (cum_tp/(ends+1))) / positive)
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    auc = float((ranks[y == 1].sum()-positive*(positive+1)/2) / (positive*negative))
    decision = scores >= threshold
    tp, fp = int(y[decision].sum()), int((1-y[decision]).sum())
    fn, tn = positive-tp, negative-fp
    f2 = 5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc = (tp*tn-fp*fn)/math.sqrt(denom) if denom else 0.
    cents = np.rint(np.asarray(amounts)*100).astype('int64')
    assert np.allclose(cents/100, amounts, rtol=0, atol=1e-8)
    table = pd.DataFrame(dict(position=np.arange(len(y)), score=scores, y=y,
                             day=pd.DatetimeIndex(times).floor('D')))
    ranked = table.sort_values(['day', 'score', 'position'], ascending=[True, False, True], kind='stable')
    selected = ranked.groupby('day', sort=False).head(capacity)
    top = np.zeros(len(y), bool)
    top[selected.position.to_numpy()] = True
    daily = table.groupby('day').agg(transactions=('y', 'size'), frauds=('y', 'sum'))
    daily['topk_alerts'] = selected.groupby('day').size().reindex(daily.index, fill_value=0)
    daily['topk_true_frauds'] = selected.groupby('day').y.sum().reindex(daily.index, fill_value=0)
    threshold_table = table.loc[decision]
    daily['threshold_alerts'] = threshold_table.groupby('day').size().reindex(daily.index, fill_value=0)
    daily['threshold_true_frauds'] = threshold_table.groupby('day').y.sum().reindex(daily.index, fill_value=0)
    cutoff_ties = []
    for day, block in ranked.groupby('day', sort=False):
        if 0 < capacity < len(block):
            cutoff = block.score.iloc[capacity-1]
            equal = block[block.score == cutoff]
            if block.score.iloc[capacity] == cutoff:
                cutoff_ties.append(dict(day=str(day), score=float(cutoff), tied=int(len(equal)),
                                        selected_from_tie=int(top[equal.position].sum())))
    detected = int(y[top].sum())
    output = dict(average_precision=ap, roc_auc=auc, f2=f2, mcc=mcc,
        precision_at_k=detected/int(top.sum()) if top.any() else 0., recall_at_k=detected/positive,
        k=int(top.sum()), topk_true_frauds=detected, total_frauds=positive,
        confusion=[[tn, fp], [fn, tp]], alerts=tp+fp,
        missed_fraud_usd=int(cents[(y == 1) & ~decision].sum())/100,
        investigation_usd=(tp+fp)*5., cost=(int(cents[(y == 1) & ~decision].sum())+(tp+fp)*500)/100,
        no_alert_cost=int(cents[y == 1].sum())/100,
        days=len(daily), peak_alerts_per_day=int(daily.threshold_alerts.max()),
        cutoff_tie_days=len(cutoff_ties), cutoff_ties=cutoff_ties,
        first_day=dict(date=str(daily.index[0]), **{k:int(v) for k,v in daily.iloc[0].items()}),
        last_day=dict(date=str(daily.index[-1]), **{k:int(v) for k,v in daily.iloc[-1].items()}))
    return output, daily, top


def verify_behavior(df, features):
    """Oracle entier en centimes, files temporelles mises à jour APRÈS chaque ex aequo."""
    t = df.trans_date_trans_time.dt.as_unit('ns').astype('int64').to_numpy()
    cents = np.rint(df.amt.to_numpy()*100).astype('int64')
    names = ['count_1h', 'count_24h', 'amount_ratio_30d', 'seconds_since_previous', 'history_missing']
    oracle = np.full((len(df), 5), np.nan)
    for indices in df.groupby('cc_num', sort=False, dropna=True).indices.values():
        queues = [deque(), deque(), deque()]
        counts = [0, 0, 0]
        sum_30d = 0
        previous = None
        i = 0
        while i < len(indices):
            now = int(t[indices[i]])
            j = i+1
            while j < len(indices) and t[indices[j]] == now:
                j += 1
            for w, span in enumerate([3600, 86400, 30*86400]):
                while queues[w] and queues[w][0][0] < now-span*10**9:
                    _, count, amount = queues[w].popleft()
                    counts[w] -= count
                    if w == 2:
                        sum_30d -= amount
            rows = indices[i:j]
            oracle[rows, 0] = counts[0]
            oracle[rows, 1] = counts[1]
            if sum_30d > 0 and counts[2] > 0:
                oracle[rows, 2] = cents[rows] * counts[2] / sum_30d
            oracle[rows, 3] = (now-previous)/10**9 if previous is not None else np.nan
            oracle[rows, 4] = int(previous is None)
            item = (now, j-i, int(cents[rows].sum()))
            for w in range(3):
                queues[w].append(item)
                counts[w] += j-i
            sum_30d += item[2]
            previous, i = now, j
    actual = features[names].to_numpy(float)
    np.testing.assert_allclose(actual, oracle, rtol=1e-8, atol=1e-8, equal_nan=True)
    error = np.abs(actual-oracle)
    return dict(rows_checked=len(df), windows='[t-durée,t[ ; groupe simultané inséré après décision',
        oracle='files temporelles, sommes exactes en centimes, aucun label lu',
        max_absolute_errors={n:float(np.nanmax(error[:, i])) for i,n in enumerate(names)})


def confirm_validation_optimum(frame, name, frozen):
    """Vérifie le minimum figé en validation ; ne modifie aucune politique."""
    ranked = frame.sort_values('score_'+name, ascending=False, kind='stable').copy()
    daily_rank = ranked.groupby(ranked.trans_date_trans_time.dt.floor('D')).cumcount().to_numpy()+1
    maximum_daily = np.maximum.accumulate(daily_rank)
    s = ranked['score_'+name].to_numpy()
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s)-1]
    feasible = ends[maximum_daily[ends] <= 50]
    fraud_cents = np.rint(ranked.amt.to_numpy()*100).astype('int64')*ranked.is_fraud.to_numpy()
    baseline = int(fraud_cents.sum())
    costs = baseline+(feasible+1)*500-np.cumsum(fraud_cents)[feasible]
    best = int(costs.argmin())
    assert costs[best] < baseline
    threshold = float(s[feasible[best]])
    assert threshold == frozen['threshold']
    assert int(costs[best])/100 == round(frozen['cost'], 2)
    return dict(threshold=threshold, cost=int(costs[best])/100, admissible_thresholds=len(feasible),
                selected_on='validation seulement ; vérification, sans mise à jour')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    before = json.loads((OUT/'frozen_before.json').read_text(encoding='utf-8'))
    results = json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    paths = [ROOT/p for p in json.loads((ROOT/'data/source_paths.json').read_text())]
    source = json.loads((ROOT/'reports/source_audit.json').read_text(encoding='utf-8'))
    for path in paths:
        item = next(f for f in source['files'] if f['file'] == path.name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    print('Source locale vérifiée ; reconstruction des scores sans fit...', flush=True)
    df = load_source(paths)
    n = len(df)
    times = df.trans_date_trans_time
    # Frontières calculées indépendamment de chronological_split.
    boundaries = [int(np.flatnonzero((times >= times.iloc[int(n*f)]).to_numpy())[0]) for f in [.6, .8]]
    a, b = boundaries
    indices = chronological_split(df)
    for actual, expected in zip(indices, [np.arange(a), np.arange(a,b), np.arange(b,n)]):
        np.testing.assert_array_equal(actual, expected)
    assert times.iloc[a-1] < times.iloc[a] and times.iloc[b-1] < times.iloc[b]
    for name, idx in zip(['train', 'validation', 'test'], indices):
        assert len(idx) == results['splits'][name]['rows']
        assert int(df.is_fraud.iloc[idx].sum()) == results['splits'][name]['frauds']
    features = build_features(df)
    print('Oracle indépendant des historiques sur tous les événements...', flush=True)
    evidence = dict(prediction_provenance='Seul le préfixe démo de 5000 lignes était initialement sauvegardé. '
        'Scores complets reconstruits sans fit depuis les modèles figés, puis relus depuis le parquet d’audit. '
        'Pas de preuve cryptographique que les scores complets étaient archivés lors de l’expérience initiale.',
        splits=results['splits'], timestamp_tied_groups=int((times.value_counts()>1).sum()),
        tied_rows=int(times.duplicated(keep=False).sum()), behavior=verify_behavior(df,features))
    for boundary in boundaries:
        prefix = build_features(df.iloc[:boundary])
        np.testing.assert_allclose(prefix[NUMERIC], features.iloc[:boundary][NUMERIC], equal_nan=True)
    evidence['split_boundary_prefix_invariance'] = True
    prep = load_bundle('boosting')['preprocessor']
    medians = np.nanmedian(features.iloc[:a][NUMERIC].to_numpy(float), axis=0)
    numeric = prep.named_transformers_['numeric']
    np.testing.assert_allclose(numeric['imputer'].statistics_, medians)
    filled = np.where(np.isnan(features.iloc[:a][NUMERIC].to_numpy(float)), medians,
                      features.iloc[:a][NUMERIC].to_numpy(float))
    np.testing.assert_allclose(numeric['scale'].mean_, filled.mean(axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(numeric['scale'].var_, filled.var(axis=0), rtol=1e-10, atol=1e-10)
    assert int(numeric['scale'].n_samples_seen_) == a
    np.testing.assert_array_equal(prep.named_transformers_['category'].categories_[0],
                                  np.sort(features.category.iloc[:a].unique()))
    evidence['train_only_preprocessing'] = True
    predictions = df.iloc[a:][['trans_date_trans_time', 'amt', 'is_fraud']].copy()
    predictions['event_position'] = np.arange(a,n)
    predictions['partition'] = ['validation']*(b-a)+['test']*(n-b)
    model_details = {}
    for name in manifest()['models']:
        bundle = load_bundle(name)
        assert bundle['policy'] == manifest()['models'][name]['policy']
        assert bundle['policy'] == results['selection'][name]['selected']['policy']
        model = bundle['model']
        with threadpool_limits(limits=2):
            X = bundle['preprocessor'].transform(features.iloc[a:])
            scores = risk_scores(model, name, X)
            direct = (model.predict_proba(X[:100])[:, list(model.classes_).index(1)] if name in ['logistic','boosting']
                      else -model.score_samples(X[:100]))
        np.testing.assert_array_equal(scores[:100], direct)
        predictions['score_'+name] = scores
        model_details[name] = dict(type=type(model).__name__, training_rows=results['selection'][name]['selected']['training_rows'],
            score_orientation='classe 1' if name in ['logistic','boosting'] else '-score_samples',
            class_weight=getattr(model,'class_weight',None), contamination=getattr(model,'contamination',None),
            novelty=getattr(model,'novelty',None))
        if name == 'lof':
            assert model.novelty and model.n_samples_fit_ == 3000
        if name == 'isolation':
            assert model._n_samples == 50000
    evidence['models'] = model_details
    predictions.to_parquet(OUT/'predictions.parquet', index=False)
    # Toute réconciliation qui suit lit les prédictions désormais sauvegardées.
    saved = pd.read_parquet(OUT/'predictions.parquet')
    evidence['recalculated'] = {}
    daily_tables = []
    from src.evaluation import daily_topk
    for partition in ['validation', 'test']:
        frame = saved[saved.partition == partition].reset_index(drop=True)
        evidence['recalculated'][partition] = {}
        for name in manifest()['models']:
            policy = load_bundle(name)['policy']
            recalculated, daily, selected = independent_metrics(frame.is_fraud, frame['score_'+name],frame.amt,
                frame.trans_date_trans_time,policy['threshold'])
            original = results['test'][name] if partition == 'test' else results['selection'][name]['selected']['validation']
            for key in ['average_precision','roc_auc','f2','mcc','precision_at_k','recall_at_k','k',
                        'alerts','cost','no_alert_cost','peak_alerts_per_day']:
                assert math.isclose(recalculated[key],original[key],rel_tol=1e-11,abs_tol=1e-8), (partition,name,key)
            assert recalculated['confusion'] == original['confusion']
            np.testing.assert_array_equal(selected, daily_topk(frame['score_'+name],frame.trans_date_trans_time,50))
            if partition == 'validation':
                recalculated['validation_optimum'] = confirm_validation_optimum(frame,name,policy)
            evidence['recalculated'][partition][name] = recalculated
            daily = daily.reset_index().assign(partition=partition,model=name)
            daily_tables.append(daily)
    pd.concat(daily_tables).to_csv(OUT/'daily_alerts.csv',index=False)
    demo = pd.read_parquet(ROOT/'data/demo/transactions.parquet')
    test = saved[saved.partition=='test'].reset_index(drop=True)
    for name in manifest()['models']:
        np.testing.assert_array_equal(demo['score_'+name],test['score_'+name].iloc[:len(demo)])
    evidence['demo_predictions_equal_reconstruction'] = True
    print('Métriques, budgets et coûts réconciliés. SHAP et HTTP...', flush=True)
    from src.explain import explain
    example = json.loads((ROOT/'reports/figures/shap_example.json').read_text(encoding='utf-8'))
    e = example['explanation']
    row = example['row']
    fresh = explain(load_bundle('boosting'),demo.iloc[[row]][FEATURES])
    np.testing.assert_allclose(fresh['values'],e['values'],rtol=0,atol=1e-12)
    base, contributions = e['base_values'][0], np.array(e['values'][0])
    margin = base+float(contributions.sum())
    assert math.isclose(margin,e['raw_output'][0],abs_tol=1e-12)
    assert math.isclose(1/(1+math.exp(-margin)),example['score'],abs_tol=1e-12)
    top = [int(i) for i in np.argsort(-contributions,kind='stable') if contributions[i]>0][:5]
    assert 3 <= len(top) <= 5
    evidence['shap'] = dict(row=row,base=base,contribution_sum=float(contributions.sum()),margin=margin,
        score=example['score'],reason_codes=[dict(name=e['names'][i],contribution=float(contributions[i])) for i in top],
        saved_values_preserved=True)
    from src.plots import shap_figure
    import matplotlib.pyplot as plt
    figure=shap_figure(e,'waterfall')
    assert len(figure.axes)>0
    plt.close('all')
    from src.runtime import ApiService
    import requests
    service=ApiService(port=8770)
    url=service.ensure_running()
    checked=[]
    try:
        with requests.Session() as session:
            session.trust_env=False
            for position in [a,b,b+row]:
                event=df.iloc[position]
                relevant=df[(df.cc_num==event.cc_num)&(df.trans_date_trans_time<event.trans_date_trans_time)]
                columns=['trans_date_trans_time','cc_num','amt','category','lat','long','merch_lat','merch_long']
                history=json.loads(relevant[columns].to_json(orient='records',date_format='iso'))
                transaction=json.loads(df.iloc[[position]][columns].to_json(orient='records',date_format='iso'))[0]
                for name in manifest()['models']:
                    response=session.post(url+'/score',json=dict(model_id=name,transaction=transaction,history=history),timeout=60)
                    assert response.status_code==200,response.text
                    output=response.json()
                    expected=float(saved.iloc[position-a]['score_'+name])
                    assert math.isclose(output['scores'][0],expected,abs_tol=1e-9)
                    assert output['threshold']==load_bundle(name)['policy']['threshold']
                    checked.append(dict(event_position=position,model=name,history_rows=len(history),absolute_error=abs(output['scores'][0]-expected)))
    finally:
        service.stop()
    evidence['http_raw_history_checks']=checked
    evidence['threshold_sequence_audit']='scripts/train.py : fit/prétraitement train ; sélection de variantes et seuils validation ; winner gelé avant transformation/scoring test. Aucun fit pendant cet audit.'
    changed=[p for p,digest in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    evidence['frozen_hashes_unchanged']=len(before)
    evidence['predictions_sha256']=hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest()
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',frozen_files_unchanged=len(before),
        winner_test=evidence['recalculated']['test']['boosting'],
        winner_validation=evidence['recalculated']['validation']['boosting']),ensure_ascii=False,indent=2),flush=True)


def replay_saved():
    """Rejouer les dénombrements depuis le ZIP, sans CSV bruts ni entraînement."""
    evidence=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
    path=OUT/'predictions.parquet'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence['predictions_sha256']
    saved=pd.read_parquet(path)
    results=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    output={}
    for partition in ['validation','test']:
        frame=saved[saved.partition==partition].reset_index(drop=True)
        assert len(frame)==results['splits'][partition]['rows']
        assert int(frame.is_fraud.sum())==results['splits'][partition]['frauds']
        assert (frame.event_position.diff().dropna()==1).all()
        output[partition]={}
        for name in results['test']:
            frozen=results['selection'][name]['selected']['policy']
            values,_,_=independent_metrics(frame.is_fraud,frame['score_'+name],frame.amt,
                frame.trans_date_trans_time,frozen['threshold'])
            expected=evidence['recalculated'][partition][name]
            for key in ['average_precision','roc_auc','f2','mcc','precision_at_k','recall_at_k',
                        'k','topk_true_frauds','alerts','cost','no_alert_cost']:
                assert math.isclose(values[key],expected[key],rel_tol=1e-12,abs_tol=1e-10)
            output[partition][name]={key:values[key] for key in ['average_precision','roc_auc','f2','mcc',
                'precision_at_k','recall_at_k','k','topk_true_frauds','cost']}
    print(json.dumps(dict(status='PASS',replay='prédictions sauvegardées uniquement',metrics=output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true',help='Recalcul depuis les seules prédictions jointes, sans source brute.')
    args=parser.parse_args()
    replay_saved() if args.replay else run()
