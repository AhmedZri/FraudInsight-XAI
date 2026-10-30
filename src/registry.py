"""Chargement de seuls artefacts locaux de confiance, vérifiés par empreinte."""
import hashlib
import json
from functools import lru_cache
import joblib
from src.data import ROOT


def manifest():
    return json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=4)
def load_bundle(model_id):
    entries=manifest()['models']
    if model_id not in entries:
        raise ValueError('Identifiant de modèle inconnu.')
    path=ROOT/'artifacts'/f'{model_id}.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entries[model_id]['sha256']:
        raise ValueError('Artefact corrompu : empreinte invalide.')
    # Ne jamais charger un pickle transmis par un utilisateur.
    return joblib.load(path)
