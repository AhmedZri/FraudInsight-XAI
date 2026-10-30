# Publication GitHub indépendante

Destination autorisée : **https://github.com/AhmedZri/FraudInsight-XAI**.

Le compte de publication est **AhmedZri**, vérifié par la CLI GitHub ; le dépôt cible est public et initialement vide. Le remote `origin` doit désigner uniquement `https://github.com/AhmedZri/FraudInsight-XAI.git`. Aucun historique ou artefact de Challenge 1 n’est incorporé.

## Périmètre public

Le dépôt contient les sources Streamlit/FastAPI, la chaîne ML canonique, les scripts reproductibles, les tests, le notebook exécuté, les dépendances, les modèles figés vérifiés et la démonstration légère de 5 000 transactions synthétiques sans carte ni identité brute. Les rapports scientifiques, le PDF de synthèse créé pour le projet et les figures SHAP/PR sont conservés. Les preuves d’audit légères restent consultables.

Sont exclus du suivi Git :

- Le **PDF de l’enseignante** dans `docs/` ; sa référence dans les documents sert uniquement à identifier le cahier des charges privé.
- Les environnements virtuels, fichiers de configuration privée, secrets, clés, identifiants Kaggle et métadonnées locales de travail.
- Les CSV bruts, le cache Kaggle, les téléversements privés et les données temporaires.
- Toutes les archives ZIP, dont la soumission académique privée ; les rapports locaux de CRC/extraction de ces archives.
- Le fichier exhaustif `reports/academic_audit/predictions.parquet` et le journal local de l’interface. Le recalcul `--replay` nécessite ce fichier de la livraison privée ; les métriques et preuves légères de réconciliation sont publiques.

## Intégrité scientifique

La publication ne réentraîne aucun modèle et ne change ni seuil, ni score, ni résultat, ni calcul métier. Les fins de ligne sont conservées par `.gitattributes` pour préserver les empreintes des fichiers scientifiques. Les artefacts sont vérifiés par SHA256 avant publication et après récupération du commit depuis GitHub ; le manifeste des modèles reste inchangé.

## Vérification et usage

Une copie isolée des seuls fichiers publiables est testée avant envoi. Après publication, le commit et l’arbre des fichiers sont récupérés à nouveau depuis GitHub pour vérifier leur identité avec la version locale, les exclusions et les empreintes.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run cloud_app.py
.\.venv\Scripts\python.exe -m pytest -q
```

La publication GitHub ne constitue pas un déploiement web. Aucun hébergement public de l’application n’est créé par cette opération.
