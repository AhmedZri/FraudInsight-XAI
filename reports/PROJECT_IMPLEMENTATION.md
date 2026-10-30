# PROJECT_IMPLEMENTATION

## Données, chronologie et fuite

- Source : [Kaggle, kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection), version 1 ; auteur Kartik Shenoy ; simulation Sparkov, Brandon Harris.
- Licence : CC0 Public Domain confirmée par les métadonnées publiques Kaggle enregistrées. Le code du générateur a une licence MIT distincte.
- Acquisition : kagglehub réellement exécuté, sans login ni jeton fourni ; deux CSV téléchargés dans le cache du présent dépôt. Aucun compte requis lors de cette exécution.
- Volume : 1,852,394 transactions ; 9,651 fraudes ; prévalence 0.521001%. CSV décompressés : 501,592,535 octets.
- Données intégralement évaluées ; budgets d’apprentissage bornés des anomalies déclarés. L’extrait public est un préfixe contigu de 5 000 observations du test, sans enrichissement en fraudes.

| Période | Transactions | Fraudes | Prévalence | Début | Fin |
|---|---:|---:|---:|---|---|
| train | 1,111,436 | 6,401 | 0.5759% | 2019-01-01 00:00:18+00:00 | 2020-04-07 07:00:14+00:00 |
| validation | 370,479 | 1,901 | 0.5131% | 2020-04-07 07:00:36+00:00 | 2020-08-25 01:31:21+00:00 |
| test | 370,479 | 1,349 | 0.3641% | 2020-08-25 01:31:59+00:00 | 2020-12-31 23:59:34+00:00 |

Frontières 60/20/20 vérifiées, sans ex aequo entre périodes. Le test conserve ses 370 479 observations et ses 1 349 fraudes. Les deux fichiers Kaggle sont réunis puis redécoupés ; leurs noms historiques ne constituent pas notre protocole.

Audit de fuite : fenêtres [t-fenêtre,t[, ex aequo exclus en bloc, jamais de labels dans les agrégats. L’historique des événements antérieurs de validation/test est observable à la décision ; les transformations apprises utilisent seulement train. Les tests de perturbation des événements futurs, du label et des frontières passent. Il s’agit d’un audit ciblé, sans garantie universelle.

Variables : montant, heure sin/cos, jour de semaine, distance haversine valide, compteurs carte 1 h/24 h, ratio montant/moyenne des 30 jours antérieurs, temps depuis la dernière transaction strictement antérieure, indicateur d’historique absent, catégorie. Carte nouvelle, moyenne nulle, coordonnées manquantes et catégorie inconnue traitées explicitement. Aucun identifiant, nom, adresse ni label comme prédicteur.

## Modèles et sélection validation

| Modèle | Pondération retenue | Observations train | Coût validation USD | AP validation |
|---|---|---:|---:|---:|
| logistic | équilibrée | 1,111,436 | 190,855.96 | 0.207497 |
| boosting | équilibrée | 1,111,436 | 31,405.93 | 0.876744 |
| isolation | aucune | 50,000 | 780,922.17 | 0.082843 |
| lof | aucune | 3,000 | 811,810.84 | 0.068869 |

Logistique et boosting histogramme réellement appris ; variantes sans pondération et équilibrées comparées sur validation. Isolation Forest (100 arbres) et LOF (25 voisins, novelty=True) réellement ajustés sans cible, sur un sous-ensemble uniforme du train naturel. Contamination auto : seuil interne des détecteurs distinct de la politique métier. Scores orientés vers le risque croissant ; anomalies sans fausse probabilité.

Sélection validation : **boosting pondéré**, coût comme critère primaire, AP pour départager. Seuil **0.882625926759**, figé avant scoring test. Recherche exacte parmi les scores distincts, plus option aucune alerte, sous plafond de 50 par jour de validation. Les autres hyperparamètres sont conservateurs et fixés, sans revendication de recherche exhaustive. Les probabilités ne sont pas calibrées.

## Résultats du test indépendant

| Modèle | AP | ROC-AUC | Précision@k | Rappel@k | F2 | MCC | Coût USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.164813 | 0.949243 | 0.148217 | 0.708673 | 0.423216 | 0.346816 | 138,433.84 |
| boosting | 0.860069 | 0.997717 | 0.202016 | 0.965901 | 0.776838 | 0.659926 | 24,534.66 |
| isolation | 0.040791 | 0.858243 | 0.093643 | 0.447739 | 0.108373 | 0.089894 | 593,574.98 |
| lof | 0.038621 | 0.851285 | 0.068992 | 0.329874 | 0.117607 | 0.091564 | 588,006.77 |

AP désigne Average Precision, somme des précisions pondérées par les incréments de rappel, pas une intégration trapézoïdale. @k : jusqu’à 50 dossiers/jour, soit 6,450 au total ; classement rétrospectif, pas ordonnanceur en ligne. Au seuil retenu : précision 47.2569%, rappel 92.5871%, 2,643 alertes, pic 47/jour. La logistique, IF et LOF dépassent la capacité certains jours test ; cela est signalé sans tronquer les résultats.

Matrice du boosting : TN=367,736, FP=1,394, FN=100, TP=1,249.

## Hypothèses financières

`Coût = Σ montant × is_fraud × (1 − efficacité × alerte) + coût_unitaire × nombre_alertes`.

Hypothèses configurables et hypothétiques : 5.00 USD/investigation, efficacité 100%, capacité 50/jour. Coût validation : **31,405.93 USD** ; coût test : **24,534.66 USD** ; sans alerte : **733,911.72 USD** ; écart hypothétique : **709,377.06 USD**. Ce n’est pas un bénéfice bancaire observé. Pas de coût de friction, coût fixe ou calendrier de récupération. La simulation interactive peut plafonner les alertes par classement quotidien, explicitement comme exploration a posteriori ; elle ne modifie jamais la sélection validée.

## SHAP, API, interface et notebook

- SHAP : vrai LinearExplainer et TreeExplainer, unités log-odds de fraude, one-hot agrégé, additivité et sigmoid vérifiées. Beeswarm, importance, waterfall et jusqu’à cinq contributions positives réelles. Motifs français et interprétation sans causalité.
- FastAPI : santé, modèles/schémas, score brut avec historique, score sur variables, explications, évaluation/coût/classement, résultats et démonstration ; OpenAPI. Schémas fermés, erreurs sans entrée sensible, artefacts locaux SHA256. Scoring vérifié par HTTP réel pour les quatre modèles ; SHAP HTTP pour les deux supervisés.
- Streamlit : six modules français vérifiés via cloud_app.py, appels HTTP réels, curseur de budget exercé ; serveur réel streamlit run cloud_app.py vérifié par HTTP santé 200. Processus interne corrigé et vérifié sous Windows ; API interne non publique.
- Notebook : dix sections, exécuté réellement par nbclient, toutes les cellules de code exécutées sans erreur. Il réutilise les modèles authentiques et les résultats figés, sans entraîner à chaque ouverture.
- Documentation : README français détaillé, matrice de conformité par page du PDF enseignant, provenance et publication séparée. Rapport une page correctement identifié comme supplémentaire.

## Tests, ressources et livraison

- Tests : **39 passés, 0 échec**, jeu complet incluant données réelles, SHAP, rechargement, API, notebook, PDF et ZIP. Sept avertissements de dépendances/petites fixtures : dépréciations, absence de positifs sur un extrait de test de 30 lignes et max_samples réduit sur fixture IF ; pas d’échec masqué.
- Extraction : les deux archives reconstruites dans des dossiers isolés passent chacune les 39 tests. Les scripts créent les dossiers temporaires nécessaires après extraction. CRC et extraction vérifiés. Les archives excluent venv, Git, secrets, jeux bruts, cache et téléversements.
- Ressources : apprentissage complet 61.95 s ; pic mémoire train Windows 1036.28 MiB. Audit Streamlit + API + serveur de santé : pic RSS combiné 713.80 MiB, inférieur à l’objectif 1 GiB. Un visiteur simulé, Windows ; aucune garantie pour Linux, charge concurrente ou limites futures Community Cloud.
- Cloud : modèles gelés, démonstration bornée, rapports précalculés, pas d’entraînement visiteur. Configuration prête à vérifier après publication autorisée ; aucun GitHub public ni déploiement public créé.
- Rapport une page : `reports/FraudInsight_XAI_une_page.pdf`, une page et rendu visuel vérifiés.
- Soumission privée : `Challenge2_Detection_Fraude_Explicable_ENSI.zip`.
- Publication préparée : `FraudInsight-XAI_publication.zip`, sans PDF enseignant. Empreintes finales : `reports/archives.json` (hors archives pour éviter circularité).

## Questions non résolues et limites

Les captures navigateur de l’interface restent indisponibles : aucun navigateur fourni par la session ; aucun faux screenshot livré. Les figures PR et SHAP réelles sont dans `reports/figures`. Aucun test public cloud exécuté, conformément à l’interdiction de publier. Les résultats restent propres à la simulation Sparkov : pas de garantie bancaire, de calibration, d’intervalle de confiance, de validation temporelle répétée ou d’authentification de production. L’API sans état exige l’historique du client pour retrouver les mêmes variables ; sans historique elle applique explicitement la convention carte nouvelle. Challenge 1 n’a pas été modifié.

## Commandes exactes

Depuis la racine du dépôt, PowerShell :

```powershell
.\.venv\Scripts\python.exe -m streamlit run cloud_app.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/acceptance.py
.\.venv\Scripts\python.exe scripts/verify_archives.py
```

Reproduire l’apprentissage et la livraison :

```powershell
.\.venv\Scripts\python.exe scripts/acquire.py
.\.venv\Scripts\python.exe scripts/audit_source.py
.\.venv\Scripts\python.exe scripts/train.py --capacity 50 --investigation 5 --effectiveness 1
.\.venv\Scripts\python.exe scripts/figures.py
.\.venv\Scripts\python.exe scripts/deliver.py
```
