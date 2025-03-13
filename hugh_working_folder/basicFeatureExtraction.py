import pandas as pd
import numpy as np

def create_tsfresh_flat_dataframe(series: pd.Series, id: str) -> pd.DataFrame:
    return pd.DataFrame({'id': id, 'time': series.index, 'x': series, 'y': np.sign(series.shift(-1))}).dropna(how='any', axis=0).reset_index(drop=True)

