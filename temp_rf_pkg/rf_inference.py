# Minimal inference for RF påstigningsmodell
import json, joblib, pandas as pd
from pathlib import Path
MODEL_PATH = Path('rf_model.joblib')
FEATURES_PATH = Path('rf_feature_cols.json')
model = joblib.load(MODEL_PATH)
features = json.loads(FEATURES_PATH.read_text(encoding='utf-8'))['features']

def predict_rows(rows:list[dict]):
    df = pd.DataFrame(rows)
    # Sikre kolonnerekkefølge og fyll manglende features med 0
    for f in features:
        if f not in df.columns:
            df[f] = 0
    df = df[features]
    preds = model.predict(df)
    return preds.tolist()

if __name__ == '__main__':
    # Eksempel (sett inn realistiske verdier)
    sample = {'anall innbyggere': 0, 'år': 0, 't_index': 0, 'sin_q': 0, 'cos_q': 0}
    print(predict_rows([sample]))
