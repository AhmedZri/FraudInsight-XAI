"""Variables disponibles à la décision, mêmes définitions hors ligne et en API."""
import numpy as np
import pandas as pd

NUMERIC = ['amt', 'hour_sin', 'hour_cos', 'day_of_week', 'distance_km',
           'count_1h', 'count_24h', 'amount_ratio_30d', 'seconds_since_previous',
           'history_missing']
FEATURES = NUMERIC + ['category']
LABELS = dict(zip(NUMERIC, ['Montant (USD)', 'Heure (sinus)', 'Heure (cosinus)',
    'Jour de semaine', 'Distance domicile-marchand (km)', 'Transactions antérieures sur 1 h',
    'Transactions antérieures sur 24 h', 'Montant / moyenne antérieure sur 30 j',
    'Secondes depuis la transaction précédente', 'Historique absent']))


def build_features(df):
    """Tri préalable requis. Fenêtres [t-fenêtre,t[, ex aequo exclus en bloc.

    Les historiques contiennent des événements observables, jamais leurs labels.
    Les fenêtres sont calculées sans apprendre un paramètre sur les données.
    """
    if not df.trans_date_trans_time.is_monotonic_increasing:
        raise ValueError('Les événements doivent être triés.')
    out = pd.DataFrame(index=df.index)
    hour = df.trans_date_trans_time.dt.hour + df.trans_date_trans_time.dt.minute/60
    out['amt'] = df.amt.astype(float)
    out['hour_sin'] = np.sin(2*np.pi*hour/24)
    out['hour_cos'] = np.cos(2*np.pi*hour/24)
    out['day_of_week'] = df.trans_date_trans_time.dt.day_of_week.astype(float)
    out['distance_km'] = np.nan
    coords = ['lat', 'long', 'merch_lat', 'merch_long']
    if set(coords).issubset(df):
        c = df[coords].apply(pd.to_numeric, errors='coerce')
        valid = (c.lat.abs()<=90)&(c.merch_lat.abs()<=90)&(c['long'].abs()<=180)&(c.merch_long.abs()<=180)
        la, lo, ma, mo = [np.radians(c[x]) for x in coords]
        h = np.sin((ma-la)/2)**2+np.cos(la)*np.cos(ma)*np.sin((mo-lo)/2)**2
        out['distance_km'] = (6371*2*np.arcsin(np.sqrt(h.clip(0,1)))).where(valid)
    out['count_1h'], out['count_24h'] = 0., 0.
    out['amount_ratio_30d'], out['seconds_since_previous'] = np.nan, np.nan
    out['history_missing'] = 1.
    # Conversion explicite en secondes : pandas 2/3 peuvent utiliser ns ou us.
    seconds = df.trans_date_trans_time.dt.as_unit('ns').astype('int64').to_numpy() / 10**9
    amounts = df.amt.to_numpy(float)
    arrays = {c: out[c].to_numpy(copy=True) for c in NUMERIC[5:]}
    for indices in df.groupby('cc_num', sort=False, dropna=True).indices.values():
        t, amt = seconds[indices], amounts[indices]
        previous_end = np.searchsorted(t, t, side='left')
        for window, name in [(3600,'count_1h'), (86400,'count_24h')]:
            arrays[name][indices] = previous_end - np.searchsorted(t, t-window, side='left')
        start = np.searchsorted(t, t-30*86400, side='left')
        counts = previous_end-start
        sums = np.r_[0., np.cumsum(amt)]
        mean = np.divide(sums[previous_end]-sums[start], counts,
                         out=np.full(len(t),np.nan), where=counts>0)
        arrays['amount_ratio_30d'][indices] = np.divide(amt, mean, out=np.full(len(t),np.nan), where=mean>0)
        exists = previous_end>0
        arrays['seconds_since_previous'][indices] = np.where(exists, t-t[np.maximum(previous_end-1,0)], np.nan)
        arrays['history_missing'][indices] = (~exists).astype(float)
    for name, values in arrays.items():
        out[name] = values
    out['category'] = df.category.astype(str)
    return out[FEATURES].replace([np.inf,-np.inf], np.nan)
