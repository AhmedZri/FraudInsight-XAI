"""Télécharger la version publique figée sans identifiants incorporés."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import acquire, ROOT

if __name__=='__main__':
    files=acquire()
    for file in files:
        print(file.name, file.stat().st_size)
    (ROOT/'data/source_paths.json').write_text(json.dumps([str(f.relative_to(ROOT)) for f in files]),encoding='utf-8')
