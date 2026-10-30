# Matrice de conformité au Challenge 2

Référence privée : `Projet_Challenge_2_Detetion de Fraude_Explicable.pdf`, quatre pages lues intégralement. Les pages 3-4 constituent le catalogue, sans nouvelles fonctionnalités imposées. « Vérifié » signifie couvert par code et contrôles exécutés ; voir résultats d’exécution dans `reports/acceptance.json` et `reports/tests.txt`. Aucun statut de déploiement public n’est revendiqué.

| Exigence exacte / synthèse fidèle | Référence | Implémentation et preuve |
|---|---|---|
| Application web interactive de fraude explicable | p.1 contexte | `app.py`, `cloud_app.py`, six vues françaises ; audit Streamlit |
| Choix d’un jeu du catalogue | p.1 ; p.3 n°1 | Sparkov Kaggle version 1, `src/data.py`, `reports/dataset_metadata.json`, audit source |
| Vérifier licence et taille | p.3 introduction | CC0 données, tailles observées et colonnes : `reports/source_audit.json` |
| Chargement automatique ou import | p.1 module 1 | kagglehub `acquire`, upload CSV validé, mappings cible/date |
| Taux de fraude et distribution des montants par classe | p.1 module 1 | KPIs, histogramme interactif, montants réels ; aucune donnée simulée en guise de résultat |
| Évolution du taux dans le temps | p.1 module 1 | Taux quotidien ; extrait démo annoncé |
| Taux par heure, catégorie, canal selon disponibilité | p.1 module 1 | Heure/catégorie disponibles, canal absent explicitement signalé |
| Heure cyclique et jour de semaine | p.1 module 2 | `src/features.py` ; sinus/cosinus et jour |
| Comportement si identifiant : 1 h, 24 h, ratio 30 j, temps précédent | p.1 module 2 | Groupement carte, fenêtres strictes ; tests des ex aequo, futur et historique absent |
| Chronologie 60/20/20 ; aléatoire seulement sans temps | p.1 module 2 | Chronologie, frontières sans ex aequo ; données sans temps refusées, repli aléatoire non nécessaire sur Sparkov |
| Déséquilibre uniquement train, comparaison validation | p.1 module 2 | Témoin sans pondération contre pondération équilibrée, candidats et coûts sauvegardés |
| Au moins un supervisé | p.2 module 3 | Logistique ET boosting histogramme (deuxième modèle ajouté par le commanditaire) |
| Deux non supervisés dont IF et choix LOF/DBSCAN/OCSVM/AE | p.2 module 3 | Isolation Forest + LOF novelty, fit sans cible ; budgets bornés documentés |
| Tableau comparatif de toutes les méthodes | p.2 module 3 | Comparaison quatre méthodes, validation/test séparés |
| PR-AUC, ROC-AUC, précision@k, rappel@k, budget justifié, F2, MCC | p.2 module 4 | AP explicitement définie, k/jour hypothétique, mesures complètes `results.json` ; tests formules |
| Courbes PR superposées | p.2 module 4 | Module 4 et notebook, courbes allégées après calcul sur test intégral |
| Seuil validation minimisant coût montants manqués + analyse, confirmé test | p.2 module 4 | Recherche exacte des scores distincts sous plafond quotidien ; référence aucune alerte ; seuil gelé avant scoring test |
| Curseurs seuil/budget, alertes, précision, rappel, coût en direct | p.2 module 4 | Simulation HTTP sur extrait annoncé ; plafond quotidien rétrospectif explicite, budget et seuil modifient décisions, mesures et coûts sans retuning |
| Explication globale SHAP beeswarm OU permutation | p.2 module 5 | SHAP beeswarm supervisé et importance absolue moyenne ; exigence beeswarm stricte ajoutée par le commanditaire |
| SHAP waterfall et 3-5 motifs d’une alerte | p.2 module 5 | Waterfall vraie et jusqu’à cinq contributions positives ; moins si insuffisantes, sans invention ; unités log-odds |
| API (FastAPI) | p.2 module 6, titre sans détails supplémentaires | API HTTP réelle, schémas fermés, modèles, score et SHAP, OpenAPI. Les routes et validations détaillées sont des ajouts du commanditaire |

## Ajouts du commanditaire, pas des obligations attribuées à l’enseignante

| Ajout demandé | Livraison / limite |
|---|---|
| Deux supervisés, géographie et historique strictement antérieur | Deux modèles + distance haversine + tests de causalité temporelle |
| Provenance détaillée et accès sans jetons inutiles | Métadonnées Kaggle enregistrées, version figée, cache local ; résultat d’acquisition dans audit |
| Coût avec efficacité configurable et plafond de capacité | Fonction exacte, absence d’alerte, coûts validation/test, diagnostic de dérive |
| SHAP et agrégation one-hot, contrôle additif | `src/explain.py` et tests numériques ; anomalies non expliquées plutôt qu’une fausse probabilité |
| Notebook de dix sections | `notebooks/FraudInsight_XAI.ipynb`, exécution réelle par nbclient |
| Rapport professionnel une page | `reports/FraudInsight_XAI_une_page.pdf`, chiffres réels ; contrôle d’une page et rendu visuel |
| Tests automatisés et intégration réelle | `tests/`, `scripts/acceptance.py`, journaux de tests et ressources |
| Guide Windows, captures, cloud et limites | README, `.streamlit`, captures et audit ; démarrage interne vérifié localement |
| ZIP privé et configuration publique distincte | `scripts/deliver.py`, liste blanche ; PDF enseignant exclu de l’archive publique, CRC/extraction |
| Déploiement gratuit et GitHub | Préparation seulement ; aucune publication autorisée ni exécutée |

Le PDF ne demande explicitement ni notebook, ni rapport une page, ni ZIP, ni hébergement public, ni un second modèle supervisé. Ces livrables répondent au présent mandat, pas à une obligation inventée de l’enseignante.
