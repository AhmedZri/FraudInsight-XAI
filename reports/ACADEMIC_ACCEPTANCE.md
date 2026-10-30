# ACADEMIC_ACCEPTANCE

**Décision : PASS — prêt pour la remise académique.** Audit du 30 octobre 2026, limité au Challenge 2. Aucun défaut de correction scientifique identifié ; aucune modification du code de production, aucun entraînement ou réajustement du seuil. Les 34 fichiers inventoriés avant audit (source brute, code canonique, modèles, manifeste, démonstration, résultats, notebook, PDF et sorties SHAP) sont conservés à l’identique, vérification SHA256.

## Référentiel et six modules obligatoires

Seule autorité normative : `docs/Projet_Challenge_2_Detetion de Fraude_Explicable.pdf`, **quatre pages relues intégralement**. Les pages 3–4 constituent le catalogue ; le jeu retenu est son entrée n°1. Aucune exigence de Challenge 1 n’est importée.

| Module imposé | Référence PDF | Constat d’acceptation | Statut |
|---|---|---|---|
| 1. Données | p.1 | Acquisition/import, taux de fraude, montants par classe, évolution temporelle, heure et catégorie. Canal absent signalé, conformément à « selon les variables disponibles ». Source synthétique, licence et taille documentées. | PASS |
| 2. Préparation | p.1 | Heure cyclique, jour, variables comportementales, chronologie 60/20/20 ; pondération limitée au train et comparaison en validation. | PASS |
| 3. Modélisation | p.2 | Logistique et boosting supervisés ; Isolation Forest et LOF novelty non supervisés ; tableau comparatif. Le PDF exige au moins **un** supervisé et **deux** non supervisés. | PASS |
| 4. Évaluation/seuil | p.2 | AP explicitement définie, ROC-AUC, précision/rappel@k, F2, MCC, courbes PR ; coût et seuil choisis en validation, confirmation sur test ; curseurs interactifs. | PASS |
| 5. Explicabilité | p.2 | SHAP global et waterfall locale, contributions et cinq motifs réels sur l’alerte contrôlée, sans causalité revendiquée. | PASS |
| 6. API | p.2 | FastAPI opérationnelle ; prédictions brutes avec historique contrôlées par HTTP. Le PDF ne détaille pas d’autres obligations de routes. | PASS |

Le notebook, le rapport d’une page et les modalités de packaging sont vérifiés comme livrables supplémentaires demandés. Publication GitHub, déploiement public et captures ne sont **ni requis pour cette acceptation, ni réalisés**.

## Chronologie et disponibilité des variables

| Période | Transactions | Fraudes | Première transaction | Dernière transaction |
|---|---:|---:|---|---|
| Train | 1 111 436 | 6 401 | 2019-01-01 00:00:18 | 2020-04-07 07:00:14 |
| Validation | 370 479 | 1 901 | 2020-04-07 07:00:36 | 2020-08-25 01:31:21 |
| Test | 370 479 | 1 349 | 2020-08-25 01:31:59 | 2020-12-31 23:59:34 |

Fractions : 59,9999784 % / 20,0000108 % / 20,0000108 %. Les 32 333 groupes d’horodatages répétés (65 176 lignes) restent dans une seule partition. Les horloges naïves de la source sont traitées selon la convention UTC commune documentée, sans affirmer leur fuseau réel.

Un oracle indépendant, à files temporelles et sommes en centimes, a contrôlé les **1 852 394 événements** : fenêtres `[t−1 h,t[`, `[t−24 h,t[`, `[t−30 j,t[`, insertion des événements simultanés seulement après leur décision. Écart nul sur les compteurs, le délai précédent et l’indicateur d’historique ; écart maximal du ratio : **5,6843 × 10⁻¹²**, dû à l’arithmétique flottante. L’invariance des préfixes est également vérifiée aux deux frontières réelles. Aucun label n’entre dans les historiques ; les événements antérieurs de validation/test sont observables sans leur étiquette.

Les médianes, moyennes/variances du scaler et catégories du prétraitement persistant correspondent au **seul train**. Les identifiants de carte ne sont que des clés d’historique. Cas sans historique, moyenne nulle et ex aequo couverts par les tests.

## Modèles, scores et recalcul indépendant

Les modèles supervisés utilisent le score de la classe **1 = fraude** ; IF et LOF utilisent **−score_samples**, jamais une fausse probabilité. IF a appris sur 50 000 observations du train et LOF sur 3 000, sans cible ; LOF a `novelty=True`. Les variantes pondérées/non pondérées sont comparées en validation. Aucun rééquilibrage ni filtrage des 370 479 lignes du test naturel (prévalence **0,364123 %**).

**Traçabilité :** l’archive initiale ne conservait que 5 000 prédictions de démonstration, pas toutes les prédictions test. L’audit a reconstruit les scores complets depuis les modèles figés et le cache source vérifié, **sans fit**, puis sauvegardé et relu `academic_audit/predictions.parquet` : 740 958 lignes validation+test, sans carte ni identité. Les 5 000 scores initiaux sont identiques. Cette reconstruction ne constitue pas une preuve d’archivage contemporain des prédictions complètes initiales.

Les formules QA n’appellent pas les métriques de production : AP aux groupes de scores distincts, ROC-AUC par rangs moyens, F2/MCC par dénombrement, classement quotidien indépendant. Tous les résultats publiés concordent à la précision numérique près.

| Test | AP / PR-AUC¹ | ROC-AUC | F2 au seuil figé | MCC | Fraudes@k | Précision@k | Rappel@k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistique | 0,164813 | 0,949243 | 0,423216 | 0,346816 | 956 | 0,148217 | 0,708673 |
| Boosting | 0,860069 | 0,997717 | 0,776838 | 0,659926 | 1 303 | 0,202016 | 0,965901 |
| Isolation Forest | 0,040791 | 0,858243 | 0,108373 | 0,089894 | 604 | 0,093643 | 0,447739 |
| LOF | 0,038621 | 0,851285 | 0,117607 | 0,091564 | 445 | 0,068992 | 0,329874 |

¹ Average Precision : somme des précisions pondérées par les incréments de rappel, sans interpolation trapézoïdale.

## Réconciliation du budget de 50 dossiers/jour

Le budget est un **scénario hypothétique**. Le diagnostic @k classe rétrospectivement les scores dans chaque jour calendaire **de la partition évaluée**, puis sélectionne `min(50, nombre d’événements du jour)`. Une égalité de score est départagée par l’ordre stable des lignes, donc la chronologie puis l’ordre source pour les événements simultanés ; jamais par la cible.

- **Test : 129 dates observées**, du 25 août au 31 décembre inclus ; chacune possède au moins 50 événements. Sélection exacte : **129 × 50 = 6 450 dossiers**, 50 chaque jour, sans suppression de lignes d’évaluation.
- Boosting : **1 303 fraudes détectées**, **5 147 dossiers légitimes** dans ces 6 450. Précision@k = **1 303 / 6 450 = 20,201550 %** ; rappel@k = **1 303 / 1 349 = 96,590067 %**. Aucun ex aequo au rang de coupure sur le test pour les quatre détecteurs.
- Premier jour test, **partiel** : début 01:31:59, 3 526 événements, 18 fraudes ; 50 sélectionnés, dont 18 fraudes pour le boosting. Dernier jour : 3 377 événements, 0 fraude ; 50 sélectionnés. **Aucune proratisation horaire**.
- Validation : 141 dates, **7 050 dossiers**, dont 1 836 fraudes pour le boosting. Le dernier jour, 25 août, n’a que 220 événements avant la frontière : il reçoit néanmoins 50 places. Un ex aequo à la coupure ce jour-là est départagé selon l’ordre stable.
- Les budgets sont remis à zéro entre partitions : le 25 août figure dans les deux évaluations. Les deux diagnostics peuvent donc sélectionner ensemble 100 dossiers ce jour-là ; **ils ne représentent pas un budget opérationnel commun de 50 sur un flux validation+test continu**.
- Fixture supplémentaire vérifiée : journée partielle de 3 événements puis journée de 55, tous ex aequo ; sélection **3 + 50 = 53**, avec choix des premiers ex aequo. Détail par date et modèle : `academic_audit/daily_alerts.csv`.

Le **seuil figé** est une autre politique : sur test, le boosting produit **2 643 alertes**, dont **1 249 fraudes**, avec **5 à 47 alertes/jour**. Les autres seuils figés atteignent 66/jour (logistique), 192 (IF), 269 (LOF) : ces dépassements déjà signalés ne sont pas masqués. Un seuil fixe ne garantit pas la capacité future. Le plafonnement rétrospectif de l’interface est explicitement une simulation distincte.

## Coût et gel du seuil

Hypothèses : montants USD, **5 USD/investigation**, **100 % d’efficacité hypothétique**. Formule : pertes des fraudes non alertées + coût unitaire × alertes. Recalcul exact en centimes :

| Période du boosting | Fraudes manquées | Alertes | Pertes manquées USD | Investigations USD | Coût total USD | Sans alerte USD |
|---|---:|---:|---:|---:|---:|---:|
| **Validation** | 159 | 3 222 | 15 295,93 | 16 110,00 | **31 405,93** | 989 313,46 |
| **Test indépendant** | 100 | 2 643 | 11 319,66 | 13 215,00 | **24 534,66** | **733 911,72** |

Le seuil **0,8826259267588804** est identique dans le modèle, le manifeste et les résultats. La vérification indépendante des seuils admissibles de validation retrouve exactement ce minimum et le plafond validation de 50/jour ; elle n’écrit aucune nouvelle politique. Le boosting pondéré reste le meilleur coût de validation enregistré. Le script d’entraînement sélectionne variantes, seuils et modèle avant le scoring test ; cet audit n’a exécuté aucun entraînement. Limite de provenance : pas de journal signé permettant de certifier rétrospectivement l’ordre historique des opérations au-delà du code et des artefacts concordants.

## SHAP, API et livrables

- Alerte sauvegardée n°3423 de la démo : base **−5,0162602801661** + contributions **11,2863864860372** = log-odds **6,2701262058711** ; sigmoid = **0,9981115830776104**, conforme au score. Waterfall rendue en mémoire ; contributions one-hot agrégées conservées.
- Les cinq motifs positifs contrôlés sont le montant (+9,544815), l’heure cosinus (+1,179439), le compteur 1 h (+0,561166), l’heure sinus (+0,414661), le délai précédent (+0,269818), en log-odds. Pas de motifs inventés ni de causalité. Toutes les sorties SHAP initiales restent inchangées.
- Douze contrôles HTTP réels, quatre détecteurs × trois transactions (dont les premières de validation et test), avec historique brut strictement antérieur : scores conformes au calcul hors ligne, écart maximal **6,7 × 10⁻¹⁶**. Les six vues Streamlit et le serveur local `streamlit run cloud_app.py` passent l’acceptation ; OpenAPI et commandes README vérifiés.
- **39 tests existants exécutés : 39 PASS, 0 FAIL**, sept avertissements de dépendances/petites fixtures ; `pip check` sans dépendance cassée. Le notebook est réexécuté sur une copie en mémoire : **11 cellules, 0 erreur**, fichier et sorties enregistrées inchangés. Un premier démarrage de kernel a expiré pendant l’audit concurrent ; reprise isolée avec boucle Windows Selector réussie.
- PDF une page : nombre de pages, chiffres, seuil et rendu visuel vérifiés ; fichier inchangé. README et matrice de conformité cohérents avec le seul PDF du Challenge 2.

## Livraison et reproduction

Rapport et preuves ajoutés au ZIP professeur **`Challenge2_Detection_Fraude_Explicable_ENSI.zip`** ; aucune régénération des modèles, du notebook, des figures ou du PDF. CRC, extraction, fichiers requis, égalité des fichiers livrés et tests dans l’extraction vérifiés. L’archive publique préexistante est laissée inchangée. Empreinte finale du ZIP privé : `reports/archives.json`, conservé hors ZIP pour éviter l’auto-référence.

```powershell
# Démarrage local et tests existants
.\.venv\Scripts\python.exe -m streamlit run cloud_app.py
.\.venv\Scripts\python.exe -m pytest -q
# Recalcul indépendant depuis les prédictions jointes, sans téléchargement ni entraînement
.\.venv\Scripts\python.exe scripts/academic_audit.py --replay
```

Preuves : `academic_audit/evidence.json`, `predictions.parquet`, `daily_alerts.csv`, `budget_edge_cases.json`, `tests.txt`, `notebook.json`, `pdf.json`, `interface.txt`, `frozen_before.json`. Le script d’audit complet sans `--replay` nécessite le cache source local initial ; il ne réentraîne pas.

**Questions restantes : aucune non-conformité bloquante au PDF du Challenge 2.** La reconstruction tardive des prédictions complètes et les conventions de budget aux journées partielles sont déclarées ci-dessus. Simulation, probabilités non calibrées, hypothèses financières et validation temporelle unique limitent l’extrapolation à une banque réelle ; aucune recherche ou phase de déploiement supplémentaire n’est engagée.
