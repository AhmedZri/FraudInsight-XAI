"""Entraînement reproductible ; sélection validation ; test consulté une fois."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(variable,'2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from src.data import ROOT, load_source, chronological_split
from src.features import build_features
from src.modeling import preprocessor, make_model, risk_scores, SUPERVISED
from src.evaluation import evaluate, select_threshold


def train(max_rows=None, capacity=50, investigation=5., effectiveness=1., methods=None,
          supervised_budget=None, isolation_budget=50000, lof_budget=3000,
          logistic_c=1.,boosting_iterations=100,lof_neighbors=25):
    if logistic_c<=0 or boosting_iterations<1 or lof_neighbors<2 or lof_budget<=lof_neighbors:
        raise ValueError('Hyperparamètres invalides : C>0, itérations>=1, 2<=voisins<budget LOF.')
    started=time.perf_counter()
    for folder in ['artifacts','data/demo','reports']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    methods=methods or ['logistic','boosting','isolation','lof']
    print('Chargement des données...',flush=True)
    df=load_source(max_rows=max_rows)
    train_idx,val_idx,test_idx=chronological_split(df)
    print(f'{len(df):,} événements ; variables historiques...',flush=True)
    features=build_features(df)
    prep=preprocessor()
    Xtrain=prep.fit_transform(features.iloc[train_idx])
    Xval=prep.transform(features.iloc[val_idx])
    # Le test transformé n'est encore ni scoré ni utilisé pour sélectionner.
    y=df.is_fraud.to_numpy()
    selection={}
    bundles={}
    artifact_entries={}
    for name in methods:
        print('Entraînement : '+name,flush=True)
        candidates=[]
        budget=supervised_budget if name in SUPERVISED else (isolation_budget if name=='isolation' else lof_budget)
        chosen=np.sort(rng.choice(len(train_idx),min(budget or len(train_idx),len(train_idx)),replace=False))
        for balanced in ([False,True] if name in SUPERVISED else [False]):
            begin=time.perf_counter()
            model=make_model(name,balanced,logistic_c,boosting_iterations,lof_neighbors)
            with threadpool_limits(limits=2):
                model.fit(Xtrain[chosen],y[train_idx][chosen]) if name in SUPERVISED else model.fit(Xtrain[chosen])
                scores=risk_scores(model,name,Xval)
            policy=select_threshold(y[val_idx],scores,df.amt.iloc[val_idx],df.trans_date_trans_time.iloc[val_idx],
                                    capacity,investigation,effectiveness)
            metrics=evaluate(y[val_idx],scores,df.amt.iloc[val_idx],df.trans_date_trans_time.iloc[val_idx],policy)
            candidate=dict(balanced=balanced,training_rows=len(chosen),training_seconds=time.perf_counter()-begin,
                           parameters=model.get_params(),policy=policy,validation=metrics)
            candidates.append((candidate,model))
        # Le coût constitue le critère primaire ; AP départage, sans test.
        best,model=min(candidates,key=lambda pair:(pair[0]['validation']['cost'],-(pair[0]['validation']['average_precision'] or 0)))
        selection[name]=dict(selected=best,candidates=[c for c,m in candidates])
        background=Xtrain[rng.choice(len(Xtrain),min(100,len(Xtrain)),replace=False)]
        bundles[name]=dict(name=name,model=model,preprocessor=prep,background=background,policy=best['policy'])
        print(name,'coût validation',round(best['validation']['cost'],2),flush=True)
    winner=min(methods,key=lambda n:(selection[n]['selected']['validation']['cost'],
                                    -(selection[n]['selected']['validation']['average_precision'] or 0)))
    print('Sélection figée : '+winner+' ; évaluation indépendante du test...',flush=True)
    Xtest=prep.transform(features.iloc[test_idx])
    test_results={}
    test_scores={}
    for name,bundle in bundles.items():
        with threadpool_limits(limits=2):
            scores=risk_scores(bundle['model'],name,Xtest)
        test_scores[name]=scores
        test_results[name]=evaluate(y[test_idx],scores,df.amt.iloc[test_idx],df.trans_date_trans_time.iloc[test_idx],bundle['policy'])
        path=ROOT/'artifacts'/f'{name}.joblib'
        joblib.dump(bundle,path,compress=3)
        artifact_entries[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            score_kind='probabilité non calibrée' if name in SUPERVISED else 'score d’anomalie (pas une probabilité)',
            policy=bundle['policy'])
    splits={}
    for name,idx in [('train',train_idx),('validation',val_idx),('test',test_idx)]:
        part=df.iloc[idx]
        splits[name]=dict(rows=len(idx),fraction=len(idx)/len(df),frauds=int(part.is_fraud.sum()),
                         prevalence=float(part.is_fraud.mean()),start=str(part.trans_date_trans_time.min()),end=str(part.trans_date_trans_time.max()))
    results=dict(source='https://www.kaggle.com/datasets/kartik2112/fraud-detection',version=1,
        mode='complet' if max_rows is None else 'développement borné : préfixe temporel',
        rows=len(df),frauds=int(y.sum()),prevalence=float(y.mean()),currency='USD',splits=splits,
        config=dict(seed=42,max_rows=max_rows,capacity=capacity,investigation=investigation,effectiveness=effectiveness,
                    supervised_budget=supervised_budget,isolation_budget=isolation_budget,lof_budget=lof_budget,
                    logistic_c=logistic_c,boosting_iterations=boosting_iterations,lof_neighbors=lof_neighbors),
        selection=selection,winner=winner,test=test_results,
        duration_seconds=time.perf_counter()-started,
        peak_rss_bytes=int(psutil.Process().memory_info().peak_wset if os.name=='nt' else psutil.Process().memory_info().rss),
        python=platform.python_version(),missing_values={c:int(v) for c,v in df.isna().sum().items()},
        feature_missing_values={c:int(v) for c,v in features.isna().sum().items()})
    (ROOT/'reports/results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'artifacts/manifest.json').write_text(json.dumps(dict(models=artifact_entries,winner=winner),ensure_ascii=False,indent=2),encoding='utf-8')
    # Préfixe contigu du test : distribution naturelle, jamais enrichi en fraudes.
    demo_idx=test_idx[:min(5000,len(test_idx))]
    demo=features.iloc[demo_idx].reset_index(drop=True)
    demo['trans_date_trans_time']=df.trans_date_trans_time.iloc[demo_idx].to_numpy()
    demo['is_fraud']=y[demo_idx]
    for name,scores in test_scores.items():
        demo[f'score_{name}']=scores[:len(demo_idx)]
    demo.to_parquet(ROOT/'data/demo/transactions.parquet',index=False)
    # Petit fond d'explication sans identité ; aucune donnée de test dans le fond SHAP.
    features.iloc[train_idx[:200]].to_parquet(ROOT/'artifacts/explanation_background.parquet',index=False)
    print(json.dumps(dict(winner=winner,rows=len(df),seconds=results['duration_seconds']),ensure_ascii=False),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-rows',type=int)
    parser.add_argument('--capacity',type=int,default=50)
    parser.add_argument('--investigation',type=float,default=5.)
    parser.add_argument('--effectiveness',type=float,default=1.)
    parser.add_argument('--supervised-budget',type=int)
    parser.add_argument('--isolation-budget',type=int,default=50000)
    parser.add_argument('--lof-budget',type=int,default=3000)
    parser.add_argument('--logistic-c',type=float,default=1.)
    parser.add_argument('--boosting-iterations',type=int,default=100)
    parser.add_argument('--lof-neighbors',type=int,default=25)
    parser.add_argument('--methods',nargs='+',choices=['logistic','boosting','isolation','lof'])
    args=vars(parser.parse_args())
    train(**args)
