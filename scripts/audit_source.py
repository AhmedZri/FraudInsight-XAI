"""Inventaire réel complet sans afficher d'identités synthétiques."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data import ROOT,acquire

if __name__=='__main__':
    audit=[]
    for file in acquire():
        rows=frauds=0
        dates=[]
        missing={}
        columns=list(pd.read_csv(file,nrows=0).columns)
        for df in pd.read_csv(file,usecols=['trans_date_trans_time','is_fraud','amt'],chunksize=100000):
            rows+=len(df)
            frauds+=int(df.is_fraud.sum())
            dates.extend([df.trans_date_trans_time.min(),df.trans_date_trans_time.max()])
            for key,value in df.isna().sum().items(): missing[key]=missing.get(key,0)+int(value)
        digest=hashlib.file_digest(file.open('rb'),'sha256').hexdigest()
        audit.append(dict(file=file.name,bytes=file.stat().st_size,rows=rows,frauds=frauds,prevalence=frauds/rows,
                          start=min(dates),end=max(dates),columns=columns,missing_values=missing,sha256=digest))
    result=dict(source='kartik2112/fraud-detection',version=1,licence='CC0: Public Domain',
                acquisition='kagglehub public, sans connexion ni jeton fourni',files=audit)
    (ROOT/'reports/source_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
