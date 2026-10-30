import json
import zipfile
from pathlib import Path
import pytest
from pypdf import PdfReader
from src.data import ROOT


@pytest.mark.integration
def test_pdf_real_results_and_one_page():
    pdf=ROOT/'reports/FraudInsight_XAI_une_page.pdf'
    reader=PdfReader(pdf)
    assert len(reader.pages)==1
    text=reader.pages[0].extract_text()
    result=json.loads((ROOT/'reports/results.json').read_text(encoding='utf-8'))
    assert result['winner'] in text
    assert f"{result['rows']:,}" in text


@pytest.mark.integration
def test_notebook_executed_without_errors():
    nb=json.loads((ROOT/'notebooks/FraudInsight_XAI.ipynb').read_text(encoding='utf-8'))
    codes=[c for c in nb['cells'] if c['cell_type']=='code']
    assert len(codes)>=10
    assert all(c['execution_count'] is not None for c in codes)
    assert not any(o.get('output_type')=='error' for c in codes for o in c['outputs'])


@pytest.mark.integration
@pytest.mark.parametrize('name,private',[
    ('Challenge2_Detection_Fraude_Explicable_ENSI.zip',True),('FraudInsight-XAI_publication.zip',False)])
def test_archives_integrity_and_exclusions(name,private):
    if not (ROOT/name).is_file():
        # Une archive ne peut pas se contenir elle-même. Dans une extraction,
        # reconstruire les paquets depuis les fichiers effectivement livrés.
        from scripts.deliver import package
        package(public=not private)
    with zipfile.ZipFile(ROOT/name) as archive:
        assert archive.testzip() is None
        names=archive.namelist()
        assert 'src/api.py' in names
        assert 'artifacts/manifest.json' in names
        assert 'reports/FraudInsight_XAI_une_page.pdf' in names
        assert all(not any(x in Path(n).parts for x in ['.git','.venv','__pycache__','cache','raw','private']) for n in names)
        professor=[n for n in names if n.startswith('docs/') and n.endswith('.pdf')]
        private_source_present=any((ROOT/'docs').glob('*.pdf'))
        assert bool(professor)==(private and private_source_present)


@pytest.mark.integration
def test_actual_http_and_streamlit_acceptance():
    audit=json.loads((ROOT/'reports/acceptance.json').read_text(encoding='utf-8'))
    assert len([k for k in audit['streamlit'] if ' · ' in k])==6
    assert audit['api']['health']['status']=='ok'
    assert 'SHAP HTTP OK' in audit['api']['boosting']
