"""Notebook exécutable, rapport une page et archives vérifiées."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import zipfile
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import ROOT


def write_notebook():
    import nbformat as nbf
    nb=nbf.v4.new_notebook()
    cells=[nbf.v4.new_markdown_cell('# FraudInsight XAI\nENSI · Data Mining · II3-IA / II3-IF · 2026-2027\n\n'
        'Analyse reproductible des artefacts réellement entraînés. Pour refaire l’entraînement : `python scripts/acquire.py` puis `python scripts/train.py`. '
        'L’exécution de ce notebook ne retouche jamais le seuil et ne relance pas implicitement l’apprentissage.'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport sys, json\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display\nresults=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))\ndemo=pd.read_parquet(ROOT/'data/demo/transactions.parquet')" )]
    sections=[
        ('1. Problème métier','Classer des transactions pour une capacité hypothétique de revue humaine ; une anomalie n’est pas une preuve de fraude.',"print('Modèle retenu en validation :',results['winner'])"),
        ('2. Données et provenance','Kartik Shenoy, Sparkov, Kaggle version 1. Données synthétiques sous CC0. Source : https://www.kaggle.com/datasets/kartik2112/fraud-detection . Montants USD. Aucun nom ni identifiant brut dans cet extrait.',"print(results['source'], results['mode'], results['rows'])\nprint('Période :',results['splits']['train']['start'],results['splits']['test']['end'])\ndisplay(demo[['amt','category','is_fraud']].describe(include='all'))"),
        ('3. Déséquilibre','La prévalence naturelle est conservée en validation et test. Pondération comparée au témoin sans pondération seulement sur train.',"print('Fraudes :',results['frauds'],'; prévalence :',results['prevalence'])\nfor name,selection in results['selection'].items():\n    display(pd.DataFrame([{'modèle':name,'pondération':c['balanced'],'AP validation':c['validation']['average_precision'],'coût validation':c['validation']['cost']} for c in selection['candidates']]))"),
        ('4. Séparation chronologique','60/20/20 approchés à l’événement ; tous les ex aequo restent ensemble. Les deux CSV source sont réunis puis découpés à nouveau selon le cahier des charges.',"display(pd.DataFrame(results['splits']).T)\nassert results['splits']['train']['end']<results['splits']['validation']['start']\nassert results['splits']['validation']['end']<results['splits']['test']['start']"),
        ('5. Variables sans fuite','Fenêtres [t-1h,t[, [t-24h,t[, [t-30j,t[. Aucun label dans les agrégations ; ex aequo exclus. Carte nouvelle et moyenne nulle traitées explicitement. Prétraitement ajusté sur train uniquement.',"from src.features import FEATURES,build_features\nfrom src.data import validate\nedge=validate(pd.DataFrame({'trans_date_trans_time':['2020-01-01 00:00','2020-01-01 00:00','2020-01-01 01:00'],'cc_num':['a']*3,'amt':[10,30,40],'category':['x']*3,'is_fraud':[0,1,0]}))\nf=build_features(edge)\nassert f.count_1h.tolist()==[0,0,2]\ndisplay(f)"),
        ('6. Modèles supervisés et anomalies','Logistique pondérée et boosting histogramme, comparés à des témoins non pondérés. Isolation Forest et LOF novelty ne reçoivent pas les étiquettes. Contamination auto ; seuil métier distinct. Budgets de calcul déclarés.',"for name,s in results['selection'].items():\n    print(name, s['selected']['training_rows'], 'observations entraînement,',round(s['selected']['training_seconds'],2),'secondes')\n    display(s['selected']['parameters'])"),
        ('7. Comparaison','AP = Average Precision (somme pondérée des précisions aux incréments de rappel), distincte de l’aire trapézoïdale. Scores orientés vers un risque croissant. Probabilités supervisées non calibrées.',"columns=['average_precision','roc_auc','precision_at_k','recall_at_k','f2','mcc','cost']\ndisplay(pd.DataFrame(results['test']).T[columns])\nimport matplotlib.pyplot as plt\nfor name,m in results['test'].items():\n    plt.plot(m['pr_curve']['recall'],m['pr_curve']['precision'],label=name)\nplt.xlabel('Rappel'); plt.ylabel('Précision'); plt.legend(); plt.show()"),
        ('8. Seuil optimisé en validation','Coût = somme montants frauduleux × (1 − efficacité × alerte) + coût d’analyse × alertes. Seuil exact parmi les scores distincts, sous capacité par jour de validation. Choix du modèle par coût validation, AP pour départager. Hyperparamètres conservateurs fixés ; pas de recherche exhaustive.',"winner=results['winner']\npolicy=results['selection'][winner]['selected']['policy']\ndisplay(policy)\nassert policy['selected_on']=='validation'"),
        ('9. Test indépendant','Le seuil est gelé avant scoring du test. @k classe a posteriori k dossiers par jour ; une telle sélection exige de voir le jour entier. Le seuil fixe peut dépasser la capacité sous dérive. L’extrait de 5 000 lignes est une démonstration, pas l’évaluation complète.',"test=results['test'][winner]\ndisplay({k:v for k,v in test.items() if k!='pr_curve'})\nprint('Économie face à aucune alerte (USD) :',test['no_alert_cost']-test['cost'])"),
        ('10. SHAP et interprétation critique','Contributions réelles en log-odds de fraude. La somme des modalités one-hot de catégorie conserve l’additivité. Le fond de la logistique provient du train ; TreeSHAP du boosting suit les chemins appris sur train. Associations, sans causalité ni certitude forensique. Les catégories et valeurs manquantes sont grises, sans ordre fictif.',"from src.registry import load_bundle\nfrom src.explain import explain\nfrom src.plots import shap_figure\nname='boosting'\ne=explain(load_bundle(name),demo[FEATURES].iloc[:40])\nv=np.array(e['values']); base=np.array(e['base_values'])\nassert np.allclose(base+v.sum(axis=1),e['raw_output'],atol=1e-5)\nshap_figure(e); plt.show()\nrow=int(np.argmax(demo['score_'+name].iloc[:40]))\nshap_figure(e,'waterfall',row); plt.show()")]
    for title,prose,code in sections:
        cells.extend([nbf.v4.new_markdown_cell('## '+title+'\n'+prose),nbf.v4.new_code_cell(code)])
    nb.cells=cells
    nb.metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    path=ROOT/'notebooks/FraudInsight_XAI.ipynb'
    nbf.write(nb,path)
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=240,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(nb,path)


def write_pdf():
    (ROOT/'tmp').mkdir(parents=True,exist_ok=True)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
    from reportlab.lib.enums import TA_LEFT
    from pypdf import PdfReader
    r=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    winner=r['winner']; m=r['test'][winner]; policy=r['selection'][winner]['selected']['policy']
    path=ROOT/'reports/FraudInsight_XAI_une_page.pdf'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyFR',fontName='Helvetica',fontSize=9,leading=12,textColor=colors.HexColor('#263B50'),spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallFR',fontName='Helvetica',fontSize=8,leading=10,textColor=colors.HexColor('#526879'),spaceAfter=5))
    story=[]
    def para(text,style='BodyFR'):
        story.append(Paragraph(text,styles[style]))
    para('FraudInsight XAI','Title')
    para('DÉTECTION DE FRAUDE EXPLICABLE | ENSI | 2026-2027','SmallFR')
    para('Data Mining - II3-IA / II3-IF - Challenge 2 - Mme Rym Besrour','SmallFR')
    para('<b>Décision étudiée.</b> Prioriser les transactions à examiner avec une capacité hypothétique de '
        f"{policy['capacity']} alertes/jour. Les résultats ci-dessous proviennent de l’exécution réelle ({r['mode']}).")
    para('<b>Données.</b> Sparkov / Kartik Shenoy, Kaggle version 1, CC0 Public Domain. '
        f"{r['rows']:,} transactions synthétiques, {r['frauds']:,} fraudes ({r['prevalence']:.3%}). "
        f"Période : {r['splits']['train']['start'][:10]} au {r['splits']['test']['end'][:10]}. Devise : USD.")
    para('<b>Protocole.</b> Séparation chronologique 60/20/20, sans ex aequo entre périodes. '
        'Fenêtres strictement antérieures sur 1 h, 24 h et 30 j ; encodeurs, médianes et échelles appris sur train. '
        'Pondération comparée au témoin non pondéré. Logistique, boosting, Isolation Forest et LOF novelty ; anomalies ajustées sans labels.')
    rows=[['Détecteur','AP test','ROC-AUC','F2','MCC','Coût test USD']]
    for name,d in r['test'].items():
        rows.append([name,f"{d['average_precision']:.4f}",f"{d['roc_auc']:.4f}",f"{d['f2']:.4f}",f"{d['mcc']:.4f}",f"{d['cost']:,.0f}"])
    table=Table(rows,colWidths=[90,62,66,53,53,105])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#15364B')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EDF4F7'),colors.white])]))
    story.extend([table,Spacer(1,12)])
    para(f"<b>Choix validation : {winner}.</b> Seuil figé : {policy['threshold']:.6f}. "
        f"Coût validation : {policy['cost']:,.2f} USD ; coût test indépendant : {m['cost']:,.2f} USD ; "
        f"sans alerte : {m['no_alert_cost']:,.2f} USD. Test : {m['alerts']:,} alertes, pic {m['peak_alerts_per_day']}/jour.")
    para(f"<b>Budget @k.</b> Précision@k : {m['precision_at_k']:.3%} ; rappel@k : {m['recall_at_k']:.3%} ; "
        f"k total : {m['k']:,} dossiers, jusqu’à {policy['capacity']}/jour (classement rétrospectif). AP = Average Precision.")
    para(f"<b>Hypothèses financières.</b> Coût = montants frauduleux × (1 - efficacité × alerte) + coût unitaire × alertes. "
        f"Analyse : {policy['investigation']:.2f} USD/alerte ; prévention/récupération : {policy['effectiveness']:.0%}. "
        'Le seuil minimise ce coût en validation sous plafond quotidien ; aucun réglage sur le test.')
    para('<b>Explication et service.</b> SHAP global et local sur les modèles supervisés, contributions en log-odds, '
        'additivité vérifiée ; catégorie one-hot agrégée. Interface française à six modules, FastAPI réellement appelée par HTTP interne. '
        'Modèles figés et extrait sans identités pour la démonstration.')
    para('<b>Limites.</b> Données simulées ; probabilités non calibrées ; budgets bornés pour les anomalies ; '
        'capacité future non garantie par un seuil fixe ; efficacité hypothétique ; pas de coûts de friction ni intervalles de confiance. '
        'SHAP décrit des associations, sans preuve causale. Déploiement public non réalisé.','SmallFR')
    para('Source : kaggle.com/datasets/kartik2112/fraud-detection | Résultats détaillés : reports/results.json | '
         'Rapport une page supplémentaire demandé par le commanditaire.','SmallFR')
    SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=43,leftMargin=43,topMargin=32,bottomMargin=28).build(story)
    assert len(PdfReader(path).pages)==1,'Le rapport doit tenir sur une page.'
    import fitz
    doc=fitz.open(path)
    doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(ROOT/'tmp/report-preview.png')


def package(public=False):
    (ROOT/'tmp').mkdir(parents=True,exist_ok=True)
    # Liste blanche, jamais de copie de tout le dépôt.
    roots=['src','scripts','tests','notebooks','.streamlit','deploy','artifacts','data/demo','reports','docs']
    names=['app.py','cloud_app.py','README.md','requirements.txt','requirements-lock.txt','pytest.ini','.gitignore','LICENSE','NOTICE.md']
    files=[ROOT/n for n in names if (ROOT/n).is_file()]
    for name in roots:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='archives.json')
    if public:
        files=[p for p in files if p.suffix.lower()!='.pdf' or p.parent.name=='reports']
    name='FraudInsight-XAI_publication.zip' if public else 'Challenge2_Detection_Fraude_Explicable_ENSI.zip'
    path=ROOT/name
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(files)):
            archive.write(file,file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            archive.extractall(folder)
            assert (Path(folder)/'src/api.py').is_file()
            assert (Path(folder)/'artifacts/manifest.json').is_file()
    print(name,path.stat().st_size,'octets ; CRC et extraction vérifiés')
    return dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crc='ok',extraction='ok')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-only',action='store_true')
    args=parser.parse_args()
    if not args.package_only:
        write_notebook()
        write_pdf()
    info=[package(False),package(True)]
    (ROOT/'reports/archives.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
