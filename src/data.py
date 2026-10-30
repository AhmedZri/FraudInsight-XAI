"""Acquisition publique et validation explicite des données."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'kartik2112/fraud-detection'
RAW_COLUMNS = ['trans_date_trans_time', 'cc_num', 'amt', 'category',
               'lat', 'long', 'merch_lat', 'merch_long', 'is_fraud']


def acquire():
    # Le cache et les téléchargements restent exclusivement dans ce dépôt.
    os.environ['KAGGLEHUB_CACHE'] = str(ROOT / 'data/cache')
    import kagglehub
    folder = Path(kagglehub.dataset_download(SOURCE + '/versions/1'))
    files = [folder / name for name in ('fraudTrain.csv', 'fraudTest.csv')]
    if not all(p.is_file() for p in files):
        raise ValueError('Archive Kaggle incomplète : deux CSV attendus.')
    return files


def validate(frame, target='is_fraud', timestamp='trans_date_trans_time', labelled=True):
    df = frame.rename(columns={target: 'is_fraud', timestamp: 'trans_date_trans_time'}).copy()
    if df.columns.duplicated().any():
        raise ValueError('Colonnes ou mappings ambigus : noms dupliqués.')
    required = ['amt', 'category', 'trans_date_trans_time'] + (['is_fraud'] if labelled else [])
    if set(required) - set(df):
        raise ValueError('Colonnes manquantes : ' + ', '.join(sorted(set(required)-set(df))))
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce', utc=True,format='mixed')
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')
    if df.empty or df['trans_date_trans_time'].isna().any():
        raise ValueError('Données vides ou horodatages invalides.')
    if not np.isfinite(df['amt']).all() or (df['amt'] < 0).any():
        raise ValueError('Les montants doivent être finis, positifs ou nuls, dans une seule devise.')
    if labelled:
        if df['is_fraud'].isna().any() or not df['is_fraud'].isin([0, 1]).all():
            raise ValueError('Cible binaire requise : 0 légitime, 1 fraude.')
        df['is_fraud'] = df['is_fraud'].astype('int8')
    df['category'] = df['category'].fillna('inconnue').astype(str)
    # Sans identifiant, aucune histoire n'est supposée ; pas de carte fictive commune.
    if 'cc_num' not in df:
        df['cc_num'] = None
    return df.sort_values('trans_date_trans_time', kind='stable').reset_index(drop=True)


def load_source(files=None, max_rows=None):
    files = files or acquire()
    frames = []
    for path in files:
        for chunk in pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS,
                                 dtype={'cc_num': 'string'}, chunksize=100_000):
            frames.append(chunk)
    df = validate(pd.concat(frames, ignore_index=True))
    if max_rows is not None and len(df) > max_rows:
        # Préfixe temporel contigu, jamais de sélection selon les étiquettes.
        boundary = df.iloc[max_rows]['trans_date_trans_time']
        df = df[df.trans_date_trans_time < boundary].copy()
    return df


def chronological_split(df):
    t = df['trans_date_trans_time']
    if len(df) < 10 or not t.is_monotonic_increasing:
        raise ValueError('Au moins 10 observations ordonnées sont nécessaires.')
    a = int(t.searchsorted(t.iloc[int(.6*len(df))], side='left'))
    b = int(t.searchsorted(t.iloc[int(.8*len(df))], side='left'))
    if not 0 < a < b < len(df):
        raise ValueError('Horodatages distincts insuffisants pour trois périodes.')
    return np.arange(a), np.arange(a, b), np.arange(b, len(df))
