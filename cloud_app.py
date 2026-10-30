"""Entrée Community Cloud : un service FastAPI interne partagé entre sessions."""
import os
import runpy
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ.setdefault(key,'2')
import streamlit as st
from src.runtime import ApiService


@st.cache_resource
def service(port):
    return ApiService(port=port)


try:
    os.environ['FRAUDINSIGHT_API_URL']=service(int(os.environ.get('FRAUDINSIGHT_INTERNAL_API_PORT','8766'))).ensure_running()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
runpy.run_path(str(Path(__file__).with_name('app.py')),run_name='__main__')
