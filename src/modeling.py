"""Prétraitement appris uniquement sur train, scores orientés vers le risque."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.features import NUMERIC

SUPERVISED = ['logistic', 'boosting']


def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                              ('scale',StandardScaler())]), NUMERIC),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['category'])],
        verbose_feature_names_out=False)


def make_model(name, balanced=True, logistic_c=1., boosting_iterations=100, lof_neighbors=25):
    weight = 'balanced' if balanced else None
    if name == 'logistic':
        return LogisticRegression(class_weight=weight,max_iter=400,C=logistic_c,random_state=42)
    if name == 'boosting':
        return HistGradientBoostingClassifier(class_weight=weight,max_iter=boosting_iterations,
            max_leaf_nodes=15,learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=42)
    if name == 'isolation':
        return IsolationForest(n_estimators=100,max_samples=256,contamination='auto',random_state=42,n_jobs=2)
    if name == 'lof':
        return LocalOutlierFactor(n_neighbors=lof_neighbors,novelty=True,contamination='auto',n_jobs=2)
    raise ValueError('Modèle inconnu.')


def risk_scores(model, name, X, batch_size=10000):
    result=[]
    for start in range(0,len(X),batch_size):
        block=X[start:start+batch_size]
        if name in SUPERVISED:
            positive = list(model.classes_).index(1)
            result.append(model.predict_proba(block)[:,positive])
        else:
            result.append(-model.score_samples(block))
    return np.concatenate(result)
