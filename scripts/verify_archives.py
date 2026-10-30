"""Extraction isolée et exécution effective des instructions pytest livrées."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import ROOT


if __name__=='__main__':
    (ROOT/'tmp').mkdir(parents=True,exist_ok=True)
    results=[]
    for name in ['Challenge2_Detection_Fraude_Explicable_ENSI.zip','FraudInsight-XAI_publication.zip']:
        with tempfile.TemporaryDirectory(prefix='extraction-',dir=ROOT/'tmp') as folder:
            target=Path(folder).resolve()
            assert target.is_relative_to((ROOT/'tmp').resolve())
            with zipfile.ZipFile(ROOT/name) as archive:
                assert archive.testzip() is None
                archive.extractall(target)
            env=os.environ.copy()
            env['PYTHONIOENCODING']='utf-8'
            env['PYTHONPATH']=str(target)
            env['OMP_NUM_THREADS']='2'
            run=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=target,env=env,capture_output=True,text=True,timeout=180)
            summary=run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr[-1000:]
            print(name,run.returncode,summary,flush=True)
            results.append(dict(archive=name,exit_code=run.returncode,summary=summary))
            if run.returncode:
                print(run.stdout,run.stderr)
                raise RuntimeError('Échec des tests après extraction.')
    (ROOT/'reports/extraction_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
