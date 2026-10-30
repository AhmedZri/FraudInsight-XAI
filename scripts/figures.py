"""Figures scientifiques sur les résultats réels, sans identité brute."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.data import ROOT
from src.registry import load_bundle
from src.features import FEATURES
from src.explain import explain
from src.plots import shap_figure


if __name__=='__main__':
    folder=ROOT/'reports/figures'
    folder.mkdir(exist_ok=True)
    results=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    plt.figure(figsize=(8,4.5))
    for name,m in results['test'].items():
        plt.plot(m['pr_curve']['recall'],m['pr_curve']['precision'],label=f"{name} · AP={m['average_precision']:.3f}")
    plt.xlabel('Rappel');plt.ylabel('Précision');plt.title('Test chronologique intégral - courbes précision-rappel')
    plt.legend();plt.grid(alpha=.2);plt.tight_layout()
    plt.savefig(folder/'precision_rappel.png',dpi=160);plt.close()
    demo=pd.read_parquet(ROOT/'data/demo/transactions.parquet')
    positions=np.unique(np.linspace(0,len(demo)-1,100).astype(int))
    name='boosting'
    e=explain(load_bundle(name),demo.iloc[positions][FEATURES])
    shap_figure(e)
    plt.title('Contributions SHAP - extrait de démonstration')
    plt.savefig(folder/'shap_beeswarm.png',dpi=150,bbox_inches='tight');plt.close('all')
    row=int(demo['score_boosting'].argmax())
    e=explain(load_bundle(name),demo.iloc[[row]][FEATURES])
    shap_figure(e,'waterfall')
    plt.title('Alerte : contributions en log-odds (boosting)')
    plt.savefig(folder/'shap_waterfall.png',dpi=150,bbox_inches='tight');plt.close('all')
    (folder/'shap_example.json').write_text(json.dumps(dict(row=row,score=float(demo.iloc[row]['score_boosting']),explanation=e),ensure_ascii=False,indent=2),encoding='utf-8')
