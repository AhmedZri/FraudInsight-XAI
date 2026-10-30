"""Audit d'acceptation réel : HTTP, six vues Streamlit et ressources locales."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key,'2')
import json
from pathlib import Path
import sys
import threading
import time
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psutil
import pandas as pd
import requests
from src.data import ROOT
from src.runtime import ApiService
from src.features import FEATURES
from src.client import api_call


def acceptance():
    started=time.perf_counter()
    (ROOT/'tmp').mkdir(parents=True,exist_ok=True)
    service=ApiService(port=8769)
    os.environ['FRAUDINSIGHT_INTERNAL_API_PORT']='8769'
    os.environ['FRAUDINSIGHT_API_URL']=service.ensure_running()
    peak=0
    stop=threading.Event()
    def monitor():
        nonlocal peak
        parent=psutil.Process()
        while not stop.wait(.1):
            processes=[parent]+parent.children(recursive=True)
            total=0
            for process in processes:
                try: total+=process.memory_info().rss
                except psutil.NoSuchProcess: pass
            peak=max(peak,total)
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    assert service.process is not None,'L’audit ressources doit posséder son processus API.'
    output={'api':{},'streamlit':{},'screenshots':'non réalisées : aucun navigateur disponible dans la session ; figures réelles séparées dans reports/figures',
            'environment':sys.platform,'public_deployment':'non exécuté, autorisation de publication absente'}
    try:
        models=api_call('/models')['models']
        output['api']['health']=api_call('/health')
        demo=pd.DataFrame(api_call('/demo'))
        rows=json.loads(demo[FEATURES].iloc[:50].to_json(orient='records'))
        for name in models:
            scored=api_call('/score/features',{'model_id':name,'rows':rows})
            assert len(scored['scores'])==50
            output['api'][name]='score HTTP OK'
            if name in ['logistic','boosting']:
                exp=api_call('/explain/features',{'model_id':name,'rows':rows[:3]})
                assert len(exp['explanation']['values'])==3
                output['api'][name]+=' ; SHAP HTTP OK'
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/'cloud_app.py'),default_timeout=120)
        at.run()
        options=list(at.sidebar.radio[0].options)
        for module in options:
            at.sidebar.radio[0].set_value(module).run(timeout=120)
            if len(at.exception):
                raise AssertionError(module+': '+str(at.exception[0].message))
            output['streamlit'][module]='rendu sans exception ; appels HTTP réels'
        # Les curseurs du module 4 sont aussi exercés, pas uniquement sa vue initiale.
        at.sidebar.radio[0].set_value(options[3]).run(timeout=120)
        at.slider[1].set_value(25).run(timeout=120)
        assert not len(at.exception)
        output['streamlit']['curseur_budget']='25 alertes/jour, rendu réussi'
        log=(ROOT/'tmp/streamlit-smoke.log').open('w',encoding='utf-8')
        options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        web=subprocess.Popen([sys.executable,'-m','streamlit','run','cloud_app.py',
            '--server.port=8511','--server.headless=true'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,**options)
        try:
            with requests.Session() as session:
                session.trust_env=False
                deadline=time.monotonic()+30
                while True:
                    try:
                        response=session.get('http://127.0.0.1:8511/_stcore/health',timeout=1)
                        if response.status_code==200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic()>deadline:
                        raise AssertionError('Le serveur Streamlit réel ne répond pas.')
                    time.sleep(.2)
            output['streamlit']['serveur_réel']='HTTP /_stcore/health 200, commande streamlit run cloud_app.py'
        finally:
            try:
                owner=psutil.Process(web.pid)
                children=owner.children(recursive=True)
                for process in children+[owner]:
                    try: process.terminate()
                    except psutil.NoSuchProcess: pass
                psutil.wait_procs(children+[owner],timeout=5)
            except psutil.NoSuchProcess:
                pass
            log.close()
        output['duration_seconds']=round(time.perf_counter()-started,2)
    finally:
        stop.set(); thread.join(timeout=2)
        output['peak_combined_rss_bytes']=peak
        output['under_1_gib']=peak<1024**3
        output['resource_scope']='Processus audit (Streamlit AppTest + imports) et descendants FastAPI, un visiteur, pas une garantie Linux/cloud.'
        service.stop()
        (ROOT/'reports/acceptance.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':
    started=time.perf_counter()
    acceptance()
