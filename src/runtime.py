"""Cycle de vie HTTP interne, aucune logique d'inférence dupliquée."""
import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import requests
import psutil

ROOT=Path(__file__).resolve().parents[1]


class ApiService:
    def __init__(self,port=8766):
        self.url=f'http://127.0.0.1:{port}'
        self.port=port
        self.process=None
        self.lock=threading.Lock()
        atexit.register(self.stop)

    def healthy(self):
        try:
            with requests.Session() as session:
                session.trust_env=False
                data=session.get(self.url+'/health',timeout=1).json()
            return data.get('service')=='fraudinsight-xai'
        except (requests.RequestException,ValueError):
            return False

    def ensure_running(self):
        with self.lock:
            if self.healthy():
                return self.url
            if self.process is None or self.process.poll() is not None:
                options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                self.process=subprocess.Popen([sys.executable,'-m','src.runtime','--port',str(self.port),
                    '--parent-pid',str(os.getpid())],cwd=ROOT,stdin=subprocess.DEVNULL,**options)
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                if self.healthy():
                    return self.url
                if self.process.poll() is not None:
                    raise RuntimeError('Le service HTTP interne a quitté.')
                time.sleep(.1)
            raise RuntimeError('Le service HTTP interne ne répond pas.')

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                parent=psutil.Process(self.process.pid)
                children=parent.children(recursive=True)
                for process in children+[parent]:
                    try: process.terminate()
                    except psutil.NoSuchProcess: pass
                _,alive=psutil.wait_procs(children+[parent],timeout=5)
                for process in alive:
                    try: process.kill()
                    except psutil.NoSuchProcess: pass
            except psutil.NoSuchProcess:
                pass


if __name__=='__main__':
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--parent-pid',type=int,required=True)
    args=parser.parse_args()
    server=uvicorn.Server(uvicorn.Config('src.api:app',host='127.0.0.1',port=args.port,access_log=False))
    def lease():
        # La lecture bloquante de stdin peut retarder le démarrage sous Windows.
        # Observer le processus parent fonctionne aussi après son arrêt brutal.
        try:
            owner=psutil.Process(args.parent_pid)
            while owner.is_running():
                time.sleep(.5)
        except psutil.NoSuchProcess:
            pass
        server.should_exit=True
    threading.Thread(target=lease,daemon=True).start()
    server.run()
