"""Rendu SHAP partagé, libellés français et catégorie non ordinale en gris."""
import numpy as np


def shap_figure(result,kind='beeswarm',index=0):
    import shap
    import matplotlib.pyplot as plt
    exp=shap.Explanation(values=np.array(result['values']),base_values=np.array(result['base_values']),
        data=np.array(result['data'],dtype=float),display_data=np.array(result['display_data'],dtype=object),
        feature_names=result['names'])
    if kind=='beeswarm':
        shap.plots.beeswarm(exp,show=False,max_display=11)
    elif kind=='waterfall':
        shap.plots.waterfall(exp[index],show=False,max_display=11)
    else:
        raise ValueError('Type de figure SHAP inconnu.')
    figure=plt.gcf()
    for axis in figure.axes:
        if axis.get_xlabel().startswith('SHAP value'):
            axis.set_xlabel('Contribution SHAP au score (log-odds)')
        if axis.get_ylabel()=='Feature value':
            axis.set_ylabel('Valeur de la variable')
        labels=[t.get_text() for t in axis.get_yticklabels()]
        if set(labels)=={'Low','High'}:
            axis.set_yticks(axis.get_yticks(),['Faible' if label=='Low' else 'Élevée' for label in labels])
    return figure
