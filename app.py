"""Tableau de bord académique français - six modules, inférence HTTP réelle."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from src.client import api_call
from src.data import ROOT, validate, chronological_split
from src.features import FEATURES, build_features

st.set_page_config(page_title='FraudInsight XAI',page_icon='◈',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem}h1{letter-spacing:-1.2px}
[data-testid="stMetric"]{background:white;border:1px solid #dce5ed;border-radius:12px;padding:16px}
[data-testid="stSidebar"]{border-right:1px solid #dce5ed}</style>''',unsafe_allow_html=True)
st.sidebar.markdown('## ◈ FraudInsight XAI')
st.sidebar.caption('ENSI · Data Mining · 2026-2027')
module=st.sidebar.radio('Navigation',['1 · Données','2 · Préparation','3 · Modélisation',
    '4 · Évaluation et coût','5 · Explications SHAP','6 · API et méthode'])
st.sidebar.info('Simulation Sparkov · USD\n\nAide à la revue humaine. Une alerte ne prouve pas une fraude.')
st.title('Détection de fraude explicable')
st.caption('Observer les transactions. Prioriser les alertes. Comprendre les décisions.')


@st.cache_data(ttl=120)
def get_reports():
    return api_call('/reports')


@st.cache_data(ttl=120)
def get_demo():
    df=pd.DataFrame(api_call('/demo'))
    df['trans_date_trans_time']=pd.to_datetime(df.trans_date_trans_time,utc=True)
    return df


def rows_payload(frame):
    return json.loads(frame[FEATURES].to_json(orient='records'))


def comparison(results):
    rows=[]
    for name,metrics in results['test'].items():
        rows.append({'Modèle':name,'AP (test)':metrics['average_precision'],'ROC-AUC':metrics['roc_auc'],
            'Précision@k':metrics['precision_at_k'],'Rappel@k':metrics['recall_at_k'],
            'F2':metrics['f2'],'MCC':metrics['mcc'],'Coût test (USD)':metrics['cost'],
            'Coût validation (USD)':results['selection'][name]['selected']['validation']['cost']})
    return pd.DataFrame(rows)


try:
    results=get_reports()
    demo=get_demo()
    registry=api_call('/models')
except (requests.RequestException,ValueError,KeyError) as exc:
    st.error('API indisponible ou artefacts absents. Utiliser cloud_app.py pour le démarrage intégré, après entraînement.')
    st.code('python scripts/acquire.py\npython scripts/train.py\npython -m streamlit run cloud_app.py')
    st.stop()

st.sidebar.caption(f"Évaluation : {results['mode']} · {results['rows']:,} événements")
st.sidebar.caption(f"Modèle retenu en validation : {results['winner']}")
data=st.session_state.get('uploaded_data',demo)
is_custom='uploaded_data' in st.session_state

if module.startswith('1'):
    st.subheader('Données et qualité')
    st.markdown('[Kaggle · kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) · '
                '[Générateur Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation) · '
                'jeu synthétique, licence de données CC0 ; aucun canal de paiement disponible.')
    with st.expander('Importer un CSV privé (en mémoire uniquement)'):
        upload=st.file_uploader('CSV : montants dans une seule devise (USD pour cette étude)',type='csv')
        a,b=st.columns(2)
        target=a.text_input('Colonne cible','is_fraud')
        timestamp=b.text_input('Colonne temporelle','trans_date_trans_time')
        if upload is not None and st.button('Valider et charger'):
            try:
                raw=pd.read_csv(upload,dtype={'cc_num':'string'})
                st.session_state['uploaded_data']=validate(raw,target,timestamp)
                st.rerun()
            except (ValueError,pd.errors.ParserError) as exc:
                st.error(str(exc))
        if st.button('Revenir à la démonstration'):
            st.session_state.pop('uploaded_data',None)
            st.rerun()
        st.caption('Les identités et cartes servent au calcul local d’historique seulement et sont masquées à l’écran. Aucun fichier privé n’est enregistré.')
    st.info('CSV utilisateur' if is_custom else 'Démonstration : préfixe contigu de 5 000 transactions du test. Les résultats scientifiques complets sont au module 4.')
    c=st.columns(4)
    for col,label,value in zip(c,['Transactions','Taux de fraude','Montant total (USD)','Montant frauduleux (USD)'],
        [f'{len(data):,}',f'{data.is_fraud.mean():.3%}',f'{data.amt.sum():,.2f}',f'{(data.amt*data.is_fraud).sum():,.2f}']):
        col.metric(label,value)
    a,b=st.columns(2)
    a.plotly_chart(px.histogram(data,x='amt',color=data.is_fraud.astype(str),nbins=60,
        labels={'amt':'Montant (USD)','color':'Fraude'},title='Montants par classe',log_y=True),width='stretch')
    daily=data.assign(jour=data.trans_date_trans_time.dt.floor('D')).groupby('jour').is_fraud.mean().reset_index()
    b.plotly_chart(px.line(daily,x='jour',y='is_fraud',labels={'is_fraud':'Taux de fraude'},title='Évolution quotidienne'),width='stretch')
    a,b=st.columns(2)
    hourly=data.assign(heure=data.trans_date_trans_time.dt.hour).groupby('heure').is_fraud.agg(['mean','size']).reset_index()
    a.plotly_chart(px.bar(hourly,x='heure',y='mean',hover_data=['size'],title='Fraude par heure',labels={'mean':'Taux'}),width='stretch')
    category=data.groupby('category').is_fraud.agg(['mean','size']).reset_index()
    b.plotly_chart(px.bar(category,x='category',y='mean',hover_data=['size'],title='Fraude par catégorie',labels={'mean':'Taux'}),width='stretch')
    st.caption('Canal : visualisation indisponible dans Sparkov. Horodatage naïf de la source traité comme une horloge commune UTC ; fuseau réel non documenté.')
    st.dataframe(pd.DataFrame({'Valeurs manquantes':data.isna().sum(),'Taux manquant':data.isna().mean()}),width='stretch')
    safe=[c for c in ['trans_date_trans_time','amt','category','is_fraud'] if c in data]
    st.dataframe(data[safe].head(200),width='stretch')

elif module.startswith('2'):
    st.subheader('Une chronologie sans fuite')
    st.dataframe(pd.DataFrame(results['splits']).T,width='stretch')
    st.markdown('Les frontières visent **60 / 20 / 20 %** des événements et conservent chaque groupe d’horodatages dans une seule période. '
        'Les encodeurs, médianes et échelles sont ajustés sur l’entraînement uniquement. Les événements antérieurs de validation et de test '
        'peuvent alimenter l’historique, sans leurs étiquettes, comme dans un flux réel.')
    st.dataframe(pd.DataFrame({'Variable':FEATURES,'Disponibilité':[
        'Montant courant','Heure courante, sinus','Heure courante, cosinus','Jour courant','Coordonnées valides à la décision',
        'Événements dans [t-1h,t[','Événements dans [t-24h,t[','Moyenne des montants dans [t-30j,t[',
        'Événement strictement antérieur','Indicateur carte nouvelle / historique absent','Catégorie courante']}),hide_index=True,width='stretch')
    st.warning('Ex aequo exclus de l’historique. Moyenne nulle ou historique insuffisant : ratio manquant, imputé par la médiane du train. '
               'Les cartes, identités, adresses, numéros de transaction et labels futurs ne sont jamais des prédicteurs.')
    if is_custom:
        f=build_features(data)
        split=chronological_split(data)
        st.write('Tailles du découpage utilisateur :', [len(x) for x in split])
        st.dataframe(f.head(100),width='stretch')

elif module.startswith('3'):
    st.subheader('Modèles et reproductibilité')
    st.dataframe(comparison(results),hide_index=True,width='stretch')
    st.markdown('Régression logistique et gradient boosting histogramme : comparaison sans pondération / pondération équilibrée, '
        'choix par coût de validation puis AP. Isolation Forest et LOF novelty : ajustement sans cible, '
        'contamination « auto » (seuil interne non utilisé pour la politique métier). Scores élevés = suspicion élevée.')
    for name,details in results['selection'].items():
        with st.expander(name):
            st.json(details['selected']['parameters'])
            st.dataframe(pd.DataFrame([{'Pondéré':c['balanced'],'Observations train':c['training_rows'],
                'Durée (s)':c['training_seconds'],'AP validation':c['validation']['average_precision'],
                'Coût validation':c['validation']['cost']} for c in details['candidates']]),hide_index=True)
    st.caption('Les probabilités supervisées ne sont pas calibrées, notamment après pondération. Les scores d’anomalie ne sont pas des probabilités.')
    with st.expander('Entraînement local contrôlé'):
        st.caption('Désactivé en hébergement. Sur votre machine, activer FRAUDINSIGHT_ALLOW_TRAIN=1. Aucun téléversement n’est utilisé.')
        methods=st.multiselect('Méthodes',['logistic','boosting','isolation','lof'],default=['logistic','boosting','isolation','lof'])
        mode=st.selectbox('Volume',['Complet','Développement : 100 000 premiers événements'])
        budget=st.number_input('Budget LOF (entraînement)',min_value=100,max_value=10000,value=3000,step=100)
        capacity=st.number_input('Capacité hypothétique / jour',min_value=0,max_value=1000,value=50)
        a,b,c=st.columns(3)
        logistic_c=a.number_input('Régularisation logistique C',min_value=.01,max_value=100.,value=1.)
        iterations=b.number_input('Itérations du boosting',min_value=10,max_value=500,value=100,step=10)
        neighbors=c.number_input('Voisins LOF',min_value=2,max_value=99,value=25)
        command=[sys.executable,'scripts/train.py','--capacity',str(capacity),'--lof-budget',str(budget),
            '--logistic-c',str(logistic_c),'--boosting-iterations',str(iterations),'--lof-neighbors',str(neighbors),'--methods',*methods]
        if mode.startswith('Développement'):
            command+=['--max-rows','100000']
        st.code(subprocess.list2cmdline(command))
        if st.button('Lancer le processus local',disabled=os.environ.get('FRAUDINSIGHT_ALLOW_TRAIN')!='1' or not methods):
            # Processus explicite, jamais à chaque visite. Redémarrer l'API après achèvement.
            (ROOT/'tmp').mkdir(parents=True,exist_ok=True)
            log=open(ROOT/'tmp/training.log','w',encoding='utf-8')
            options={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,**options)
            log.close()
            st.success(f'Processus {process.pid} démarré. Journal : tmp/training.log. Redémarrer l’application après achèvement.')

elif module.startswith('4'):
    st.subheader('Évaluation indépendante et politique financière')
    st.dataframe(comparison(results),hide_index=True,width='stretch')
    st.caption('AP = Average Precision, somme des incréments de rappel pondérés par la précision ; pas une aire trapézoïdale. '
        '@k = classement rétrospectif de capacité k par jour ; ce diagnostic n’est pas un ordonnanceur temps réel.')
    fig=go.Figure()
    for name,m in results['test'].items():
        curve=m['pr_curve']
        fig.add_trace(go.Scatter(x=curve['recall'],y=curve['precision'],name=name,mode='lines'))
    fig.update_layout(title='Précision-rappel sur tout le test',xaxis_title='Rappel',yaxis_title='Précision')
    st.plotly_chart(fig,width='stretch')
    name=st.selectbox('Détecteur',list(registry['models']),index=list(registry['models']).index(results['winner']))
    policy=registry['models'][name]['policy']
    st.write('Politique figée en validation',policy)
    independent=results['test'][name]
    st.write(f"Coût validation : {policy['cost']:,.2f} USD · coût test : {independent['cost']:,.2f} USD · "
        f"sans alerte : {independent['no_alert_cost']:,.2f} USD")
    if independent['capacity_exceeded']:
        st.warning(f"Dérive de capacité : pic test {independent['peak_alerts_per_day']} alertes/jour. Le seuil figé n’impose pas de plafond futur.")
    st.markdown('**Simulation interactive sur un échantillon** - cette exploration ne revalide pas un seuil sur le test.')
    frame=build_features(data) if is_custom else data[FEATURES]
    scored=api_call('/score/features',{'model_id':name,'rows':rows_payload(frame.iloc[:5000])})
    observed=data.iloc[:len(scored['scores'])].copy()
    scores=np.array(scored['scores'])
    lo,hi=float(min(scores.min(),policy['threshold'])),float(max(scores.max(),policy['threshold']))
    threshold=st.slider('Seuil de risque',lo,hi if hi>lo else lo+1.,float(policy['threshold']),format='%.6f')
    capacity=st.slider('Capacité hypothétique (alertes/jour)',0,max(200,policy['capacity']),policy['capacity'])
    apply_capacity=st.checkbox('Simuler un plafond quotidien après classement rétrospectif',value=True)
    investigation=st.number_input('Coût d’analyse / alerte (USD)',min_value=0.,value=float(policy['investigation']))
    effectiveness=st.slider('Efficacité hypothétique de prévention / récupération',0.,1.,float(policy['effectiveness']))
    evaluation_frame=frame.iloc[:len(observed)].reset_index(drop=True).copy()
    evaluation_frame['trans_date_trans_time']=observed.trans_date_trans_time.to_numpy()
    evaluation_frame['is_fraud']=observed.is_fraud.to_numpy()
    evaluation_rows=json.loads(evaluation_frame.to_json(orient='records',date_format='iso'))
    m=api_call('/evaluate',dict(model_id=name,rows=evaluation_rows,threshold=threshold,capacity=capacity,
        investigation=investigation,effectiveness=effectiveness,apply_capacity=apply_capacity))['metrics']
    c=st.columns(6)
    for col,label,value in zip(c,['Alertes','Précision','Rappel','F2','MCC','Coût (USD)'],
        [str(m['alerts']),f"{m['precision']:.3f}",f"{m['recall']:.3f}",f"{m['f2']:.3f}",f"{m['mcc']:.3f}",f"{m['cost']:,.2f}"]):
        col.metric(label,value)
    st.write(f"Capacité @k : {m['k']} dossiers ; précision@k {m['precision_at_k']:.3f} ; rappel@k {m['recall_at_k']:.3f}. "
        f"Pic d’alertes au seuil : {m['peak_alerts_per_day']}/jour.")
    st.caption('Coût = montants frauduleux × (1 − efficacité × alerte) + coût unitaire × nombre d’alertes. '
        'Sans alerte : tous les montants frauduleux sont perdus. Pas de coût de friction client, ni de récupération différée modélisés.')
    a,b=st.columns([1,2])
    a.plotly_chart(px.imshow(m['confusion'],text_auto=True,x=['Sans alerte','Alerte'],y=['Légitime','Fraude'],title='Matrice de confusion'),width='stretch')
    observed['Risque']=scores
    observed['Alerte']=scores>=threshold
    if apply_capacity:
        from src.evaluation import daily_topk
        observed['Alerte'] &= daily_topk(scores,observed.trans_date_trans_time,capacity)
    b.dataframe(observed.sort_values('Risque',ascending=False)[['trans_date_trans_time','amt','category','is_fraud','Risque','Alerte']].head(100),hide_index=True,width='stretch')

elif module.startswith('5'):
    st.subheader('SHAP : comprendre une contribution, sans causalité')
    name=st.selectbox('Modèle supervisé',[n for n in registry['models'] if n in ['logistic','boosting']])
    frame=build_features(data) if is_custom else data[FEATURES]
    frame=frame.iloc[:5000]
    scored=api_call('/score/features',{'model_id':name,'rows':rows_payload(frame)})
    scores=np.array(scored['scores'])
    flagged=np.flatnonzero(np.array(scored['alerts']))
    candidates=flagged[np.argsort(-scores[flagged])] if len(flagged) else np.argsort(-scores)[:100]
    if not len(flagged):
        st.info('Aucune alerte au seuil figé dans cet extrait ; sélection des scores les plus élevés pour illustration.')
    row=st.selectbox('Transaction (index sans identité)',candidates.tolist(),format_func=lambda i:f'Événement {i} · {scores[i]:.4f} · {data.iloc[i].amt:.2f} USD')
    global_rows=np.unique(np.linspace(0,len(frame)-1,min(100,len(frame))).astype(int))
    with st.spinner('Calcul SHAP réel par l’API...'):
        global_exp=api_call('/explain/features',{'model_id':name,'rows':rows_payload(frame.iloc[global_rows])})['explanation']
        local=api_call('/explain/features',{'model_id':name,'rows':rows_payload(frame.iloc[[row]])})
    # Import des bibliothèques de rendu seulement dans ce module.
    import matplotlib.pyplot as plt
    from src.plots import shap_figure
    st.caption('Vue globale sur au plus 100 événements uniformément espacés dans l’extrait affiché. Importance ≠ causalité. '
               'La catégorie one-hot est agrégée ; sa couleur numérique est neutralisée et ne représente pas un ordre.')
    a,b=st.columns(2)
    a.pyplot(shap_figure(global_exp)); plt.close('all')
    importance=pd.DataFrame({'Variable':global_exp['names'],'Contribution absolue moyenne':np.abs(np.array(global_exp['values'])).mean(axis=0)}).sort_values('Contribution absolue moyenne')
    b.plotly_chart(px.bar(importance,x='Contribution absolue moyenne',y='Variable',orientation='h'),width='stretch')
    exp=local['explanation']
    st.write(f"Score : {local['scores'][0]:.5f} · Alerte : {'oui' if local['alerts'][0] else 'non'} · unité SHAP : {exp['units']}")
    st.pyplot(shap_figure(exp,'waterfall')); plt.close('all')
    values=np.array(exp['values'][0])
    positive=np.flatnonzero(values>0)
    top=positive[np.argsort(-values[positive])][:5]
    reasons=[f"{exp['names'][i]} : +{values[i]:.3f} log-odds" for i in top]
    st.markdown('**Motifs associés à une hausse du score**\n\n'+'\n'.join('- '+r for r in reasons) if reasons else 'Aucune contribution positive dans ce cas.')
    state='signalée pour revue' if local['alerts'][0] else 'sous le seuil d’alerte'
    main=', '.join(exp['names'][i] for i in top[:3])
    st.write(f'Cette transaction est {state} au seuil figé. '+
        (f'Les principales contributions à la hausse du score sont : {main}. ' if main else '')+
        'L’analyste doit confronter ces associations au contexte du dossier avant de conclure.')
    st.caption('Ces associations décrivent le modèle. Elles ne prouvent ni l’intention ni une relation causale. '
        'Base + contributions = log-odds ; la fonction logistique donne le score supervisé non calibré. '
        'SHAP des détecteurs non supervisés non proposé.')

else:
    st.subheader('API HTTP et cadre scientifique')
    st.json(api_call('/health'))
    url=os.environ.get('FRAUDINSIGHT_API_URL','http://127.0.0.1:8766')
    st.code(f'{url}/docs')
    st.caption('API interne en Community Cloud : ce lien n’est accessible que depuis le serveur. En local, OpenAPI est accessible dans votre navigateur.')
    st.dataframe(pd.DataFrame({'Route':['GET /health','GET /models','POST /score','POST /score/features',
        'POST /explain','POST /explain/features','POST /evaluate','GET /reports','GET /demo'],
        'Usage':['Santé','Modèles et schémas','Transaction et historique strictement antérieur','Variables déjà construites',
        'SHAP sur transaction brute','SHAP sur variables validées','Mesures, coût et classement exploratoires','Résultats figés','Extrait sans identités']}),hide_index=True,width='stretch')
    st.markdown('**Limites** : simulation américaine ; absence de garantie de généralisation bancaire ; période et fuseau source imparfaitement documentés ; '
        'LOF appris sur un budget borné ; validation unique sans intervalle de confiance ; probabilités non calibrées ; '
        'capacité quotidienne indicative ; historique API limité aux événements fournis ; pas de service public déployé ou authentifié.')
    st.download_button('Télécharger les résultats vérifiés',json.dumps(results,ensure_ascii=False,indent=2),'resultats.json','application/json')
