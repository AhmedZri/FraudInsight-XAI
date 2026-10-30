"""SHAP réel en log-odds pour les modèles supervisés."""
import numpy as np
from src.features import LABELS


def explain(bundle, features):
    import shap
    name, model = bundle['name'], bundle['model']
    if name not in ('logistic','boosting'):
        raise ValueError('SHAP disponible pour les modèles supervisés seulement.')
    transformed = bundle['preprocessor'].transform(features)
    names = list(bundle['preprocessor'].get_feature_names_out())
    if name == 'logistic':
        explainer = shap.LinearExplainer(model,bundle['background'])
    else:
        explainer = shap.TreeExplainer(model,model_output='raw')
    values = explainer(transformed)
    # Somme signée des modalités one-hot : additivité conservée.
    base_names = list(features.columns)
    aggregated = np.zeros((len(features),len(base_names)))
    for j, feature in enumerate(base_names):
        columns=[k for k,n in enumerate(names) if n==feature or (feature=='category' and n.startswith('category_'))]
        aggregated[:,j]=values.values[:,columns].sum(axis=1)
    raw = model.decision_function(transformed) if name=='logistic' else model._raw_predict(transformed).ravel()
    bases = np.broadcast_to(values.base_values,(len(features),)).astype(float)
    if not np.allclose(bases+aggregated.sum(axis=1),raw,atol=1e-5):
        raise ValueError('Échec du contrôle d’additivité SHAP.')
    numeric_display=features.assign(category=np.nan).to_numpy(float)
    display_values=[[float(v) if np.isfinite(v) else None for v in row] for row in numeric_display]
    return dict(values=aggregated.tolist(),base_values=bases.tolist(),raw_output=raw.tolist(),
                names=[LABELS.get(n,'Catégorie marchande') for n in base_names],
                units='log-odds de la classe fraude (1)',
                display_data=features.astype(object).where(features.notna(),'indisponible').to_numpy().tolist(),
                data=display_values)
