"""Mesures et politique figée, sans optimisation sur le test."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, fbeta_score, matthews_corrcoef, confusion_matrix, precision_recall_curve)


def financial_cost(y, amounts, alerts, investigation=5., effectiveness=1.):
    if investigation < 0 or not 0<=effectiveness<=1:
        raise ValueError('Coût positif et efficacité entre 0 et 1 requis.')
    y, amounts, alerts = np.asarray(y), np.asarray(amounts), np.asarray(alerts, bool)
    return float(np.sum(amounts*y*(1-effectiveness*alerts)) + investigation*alerts.sum())


def daily_topk(scores, times, capacity):
    if capacity < 0:
        raise ValueError('Capacité négative.')
    days = pd.DatetimeIndex(times).floor('D')
    alerts = np.zeros(len(scores), bool)
    for day in days.unique():
        idx = np.flatnonzero(days==day)
        alerts[idx[np.argsort(-np.asarray(scores)[idx], kind='stable')[:capacity]]] = True
    return alerts


def select_threshold(y, scores, amounts, times, capacity=50, investigation=5., effectiveness=1., *, partition='validation'):
    """Minimisation exacte parmi les scores uniques, sous plafond PAR JOUR.

    Une politique à seuil fixe est réalisable si aucun jour de validation ne
    dépasse la capacité. Aucun classement rétrospectif ne modifie ce seuil.
    """
    if partition != 'validation':
        raise ValueError('La sélection du seuil est autorisée en validation seulement.')
    if capacity < 0 or investigation < 0 or not 0 <= effectiveness <= 1:
        raise ValueError('Capacité/coût positifs et efficacité entre 0 et 1 requis.')
    scores, y, amounts = np.asarray(scores), np.asarray(y), np.asarray(amounts)
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    codes, _ = pd.factorize(pd.DatetimeIndex(times).floor('D'))
    daily = np.zeros(codes.max()+1, int)
    baseline = float(np.sum(y*amounts))
    best = dict(threshold=float(np.nextafter(s.max(), np.inf)), cost=baseline, alerts=0,
                capacity=capacity, investigation=investigation, effectiveness=effectiveness, selected_on='validation')
    cost, i = baseline, 0
    while i < len(s):
        j = int(np.searchsorted(-s, -s[i], side='right'))
        rows = order[i:j]
        daily += np.bincount(codes[rows], minlength=len(daily))
        if daily.max() > capacity:
            break
        cost += investigation*len(rows)-effectiveness*float(np.sum(y[rows]*amounts[rows]))
        if cost < best['cost']:
            best.update(threshold=float(s[i]), cost=float(cost), alerts=j)
        i = j
    return best


def evaluate(y, scores, amounts, times, policy, *, apply_capacity=False):
    y, scores = np.asarray(y), np.asarray(scores)
    alerts = scores >= policy['threshold']
    top = daily_topk(scores, times, policy['capacity'])
    if apply_capacity:
        # Simulation rétrospective seulement ; ne retouche jamais la politique gelée.
        alerts &= top
    days = pd.DatetimeIndex(times).floor('D')
    peak = int(pd.Series(alerts.astype(int)).groupby(days).sum().max())
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # Courbe allégée pour la démonstration, métriques calculées sur TOUS les scores.
    idx = np.unique(np.linspace(0,len(precision)-1,min(500,len(precision))).astype(int))
    two = len(np.unique(y))==2
    return dict(average_precision=float(average_precision_score(y,scores)) if two else None,
        roc_auc=float(roc_auc_score(y,scores)) if two else None,
        precision_at_k=float(y[top].mean()) if top.any() else 0.,
        recall_at_k=float(y[top].sum()/max(1,y.sum())), k=int(top.sum()),
        precision=float(precision_score(y,alerts,zero_division=0)),
        recall=float(recall_score(y,alerts,zero_division=0)),
        f2=float(fbeta_score(y,alerts,beta=2,zero_division=0)),
        mcc=float(matthews_corrcoef(y,alerts)),
        confusion=confusion_matrix(y,alerts,labels=[0,1]).tolist(),
        alerts=int(alerts.sum()), peak_alerts_per_day=peak, capacity_exceeded=peak>policy['capacity'],
        cost=financial_cost(y,amounts,alerts,policy['investigation'],policy['effectiveness']),
        no_alert_cost=float(np.sum(y*np.asarray(amounts))),
        pr_curve=dict(precision=precision[idx].tolist(),recall=recall[idx].tolist()))
