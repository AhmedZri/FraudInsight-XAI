import numpy as np
import pandas as pd
import pytest
from src.data import validate


@pytest.fixture
def events():
    n=200
    return validate(pd.DataFrame(dict(trans_date_trans_time=pd.date_range('2020-01-01',periods=n,freq='h',tz='UTC'),
        cc_num=['carte-A']*n,amt=np.arange(n,dtype=float)+1,category=['alimentation' if i%2 else 'voyage' for i in range(n)],
        is_fraud=[int(i%17==0) for i in range(n)],lat=[36.]*n,long=[10.]*n,merch_lat=[36.1]*n,merch_long=[10.1]*n)))
