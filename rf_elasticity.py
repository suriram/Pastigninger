"""Beregn befolkningselastisitet per delmarkedsområde for en Random Forest modell.
Bruker samme logikk som compute_population_elasticity i main.py, men trener en separat RF.
Output:
 - artifacts/elasticity/rf_population_elasticity_rows.csv
 - artifacts/elasticity/rf_population_elasticity_by_area.csv
 - artifacts/elasticity/rf_population_elasticity_summary.json
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
import hashlib
import zipfile
import sklearn
from datetime import datetime
import json

DATA_PATH = Path('reg.csv')
ART = Path('artifacts')
(ART / 'elasticity').mkdir(parents=True, exist_ok=True)

POP_FEATURE = 'anall innbyggere'
AREA_COL = 'delmarkedsområde'
TARGET = 'påstigninger'


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    # Numeric casts
    for c in ['år','kvartall',TARGET, POP_FEATURE]:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                              .str.replace('"','', regex=False)
                              .str.replace(',','', regex=False)
                              .str.strip())
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['år','kvartall',TARGET, POP_FEATURE])
    df = df.astype({'år':int,'kvartall':int,TARGET:int,POP_FEATURE:int})
    # Fjern aggregat Area 12
    if AREA_COL in df.columns:
        df[AREA_COL] = df[AREA_COL].astype(str).str.strip()
        df = df[df[AREA_COL] != '12']
    df = df.sort_values(['år','kvartall']).reset_index(drop=True)
    # Features (speiler main.make_features men minimal variant)
    df['t_index'] = range(1, len(df)+1)
    if AREA_COL in df.columns:
        df['t_index_area'] = df.groupby(AREA_COL).cumcount() + 1
    season = pd.get_dummies(df['kvartall'].astype(int), prefix='Q')
    df = pd.concat([df, season], axis=1)
    df['sin_q'] = np.sin(2*np.pi*(df['kvartall']/4.0))
    df['cos_q'] = np.cos(2*np.pi*(df['kvartall']/4.0))
    if AREA_COL in df.columns:
        area_dummies = pd.get_dummies(df[AREA_COL], prefix='AREA')
        df = pd.concat([df, area_dummies], axis=1)
    return df


def build_matrix(df: pd.DataFrame):
    feature_cols = [POP_FEATURE, 'år', 't_index', 'sin_q', 'cos_q']
    if 't_index_area' in df.columns:
        feature_cols.append('t_index_area')
    feature_cols.extend([c for c in df.columns if c.startswith('Q_')])
    feature_cols.extend([c for c in df.columns if c.startswith('AREA_')])
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y, feature_cols


def compute_point_elasticity(model, X: pd.DataFrame, pop_feature: str):
    preds = model.predict(X)
    records = []
    for i, (pop, y_hat) in enumerate(zip(X[pop_feature].values, preds)):
        pop_val = float(pop)
        y_val = float(y_hat)
        if pop_val <= 0 or y_val <= 0:
            records.append({'index': X.index[i], pop_feature: pop_val, 'pred': y_val, 'elasticity': np.nan})
            continue
        delta = max(1.0, pop_val * 0.01)
        row_plus = X.iloc[i].copy(); row_plus[pop_feature] = pop_val + delta
        row_minus = X.iloc[i].copy(); row_minus[pop_feature] = max(0.0, pop_val - delta)
        try:
            y_plus = model.predict(pd.DataFrame([row_plus], columns=X.columns))[0]
            y_minus = model.predict(pd.DataFrame([row_minus], columns=X.columns))[0]
            denom = (row_plus[pop_feature] - row_minus[pop_feature])
            deriv = np.nan if denom == 0 else (y_plus - y_minus) / denom
            elasticity = deriv * (pop_val / y_val) if (deriv is not None and y_val != 0) else np.nan
        except Exception:
            y_plus = y_minus = np.nan
            deriv = np.nan
            elasticity = np.nan
        records.append({
            'index': X.index[i],
            pop_feature: pop_val,
            'pred': y_val,
            'delta_pop': delta,
            'pred_plus': y_plus,
            'pred_minus': y_minus,
            'derivative_dY_dPop': deriv,
            'elasticity': elasticity
        })
    df_el = pd.DataFrame(records)
    return df_el


def aggregate_per_area(el_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    if AREA_COL not in original_df.columns:
        return pd.DataFrame()
    tmp = el_df.merge(original_df[[AREA_COL]].reset_index().rename(columns={'index':'orig_index'}), left_on='index', right_on='orig_index', how='left')
    grp = tmp.groupby(AREA_COL)['elasticity']
    rows = []
    for area, series in grp:
        v = series.replace([np.inf, -np.inf], np.nan).dropna()
        if v.empty:
            rows.append({AREA_COL: area, 'n_valid': 0, 'mean': None, 'median': None, 'p25': None, 'p75': None})
        else:
            rows.append({
                AREA_COL: area,
                'n_valid': int(v.shape[0]),
                'mean': float(v.mean()),
                'median': float(v.median()),
                'p25': float(v.quantile(0.25)),
                'p75': float(v.quantile(0.75))
            })
    return pd.DataFrame(rows)


def main():
    print("Beregner Random Forest befolkningselastisitet per område...")
    df = load_clean()
    X, y, feature_cols = build_matrix(df)
    rf = RandomForestRegressor(random_state=42, n_estimators=500, max_depth=None, n_jobs=-1)
    rf.fit(X, y)
    el_rows = compute_point_elasticity(rf, X, POP_FEATURE)
    rows_path = ART / 'elasticity' / 'rf_population_elasticity_rows.csv'
    el_rows.to_csv(rows_path, index=False, encoding='utf-8')
    area_df = aggregate_per_area(el_rows, df)
    area_path = ART / 'elasticity' / 'rf_population_elasticity_by_area.csv'
    area_df.to_csv(area_path, index=False, encoding='utf-8')
    valid = el_rows['elasticity'].replace([np.inf,-np.inf], np.nan).dropna()
    summary = {
        'n_valid': int(valid.shape[0]),
        'mean': float(valid.mean()) if not valid.empty else None,
        'median': float(valid.median()) if not valid.empty else None,
        'p25': float(valid.quantile(0.25)) if not valid.empty else None,
        'p75': float(valid.quantile(0.75)) if not valid.empty else None
    }
    (ART / 'elasticity' / 'rf_population_elasticity_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("Lagret:")
    print(f" - {rows_path}")
    print(f" - {area_path}")
    print("Sammendrag:")
    print(summary)

    # --- Modellpakking ---
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    models_dir = ART / 'models'
    metadata_dir = ART / 'metadata'
    dist_dir = ART / 'distribution'
    for d in [models_dir, metadata_dir, dist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_version_name = f'rf_model_{ts}.joblib'
    model_path_versioned = models_dir / model_version_name
    model_path_latest = models_dir / 'rf_model.joblib'
    joblib.dump(rf, model_path_versioned)
    # Oppdater "latest" (kopi)
    try:
        if model_path_latest.exists():
            model_path_latest.unlink()
    except Exception:
        pass
    joblib.dump(rf, model_path_latest)

    # Feature metadata
    feat_meta = {
        'created_utc': ts,
        'pop_feature': POP_FEATURE,
        'target': TARGET,
        'n_training_rows': int(df.shape[0]),
        'features': feature_cols
    }
    feature_meta_path = metadata_dir / 'rf_feature_cols.json'
    feature_meta_path.write_text(json.dumps(feat_meta, ensure_ascii=False, indent=2), encoding='utf-8')

    # Manifest (append eller opprett)
    manifest_path = models_dir / 'rf_model_manifest.json'
    rf_params = rf.get_params()
    entry = {
        'timestamp_utc': ts,
        'model_file': model_version_name,
        'n_estimators': rf_params.get('n_estimators'),
        'max_depth': rf_params.get('max_depth'),
        'random_state': rf_params.get('random_state'),
        'sklearn_version': sklearn.__version__,
        'n_training_rows': int(df.shape[0]),
        'n_features': len(feature_cols),
        'features_sha256': hashlib.sha256('\n'.join(feature_cols).encode('utf-8')).hexdigest(),
        'elasticity_summary_mean': summary.get('mean'),
        'elasticity_summary_median': summary.get('median')
    }
    manifest_data = []
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            manifest_data = []
    manifest_data.append(entry)
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding='utf-8')

    # Hash av modellfil
    model_bytes = model_path_versioned.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    hash_path = models_dir / f'{model_version_name}.sha256'
    hash_path.write_text(model_sha, encoding='utf-8')

    # Minimum requirements
    requirements_text = f"scikit-learn=={sklearn.__version__}\njoblib=={joblib.__version__}\npandas=={pd.__version__}\nnumpy=={np.__version__}\n"
    req_path = dist_dir / 'requirements_min.txt'
    req_path.write_text(requirements_text, encoding='utf-8')

    # Inference skript
    inference_code = f"""# Minimal inference for RF påstigningsmodell\nimport json, joblib, pandas as pd\nfrom pathlib import Path\nMODEL_PATH = Path('rf_model.joblib')\nFEATURES_PATH = Path('rf_feature_cols.json')\nmodel = joblib.load(MODEL_PATH)\nfeatures = json.loads(FEATURES_PATH.read_text(encoding='utf-8'))['features']\n\ndef predict_rows(rows:list[dict]):\n    df = pd.DataFrame(rows)\n    # Sikre kolonnerekkefølge og fyll manglende features med 0\n    for f in features:\n        if f not in df.columns:\n            df[f] = 0\n    df = df[features]\n    preds = model.predict(df)\n    return preds.tolist()\n\nif __name__ == '__main__':\n    # Eksempel (sett inn realistiske verdier)\n    sample = {{{', '.join([repr(fc)+': 0' for fc in feature_cols[:5]])}}}\n    print(predict_rows([sample]))\n"""
    inference_path = dist_dir / 'rf_inference.py'
    inference_path.write_text(inference_code, encoding='utf-8')

    # README
    readme_text = f"""# Random Forest påstigningsmodell\n\nDenne pakken inneholder en trent RandomForestRegressor for å estimere påstigninger basert på befolkning og tids-/sesongfeatures.\n\n## Innhold\n- rf_model.joblib (siste versjon)\n- rf_feature_cols.json (feature-liste og metadata)\n- rf_inference.py (enkelt prediksjonsskript)\n- requirements_min.txt (minimal avhengigheter)\n- rf_model_manifest.json (historikk)\n- *rf_model_...*.joblib (versjonert modell)\n- *.sha256 (integritetshash)\n\n## Bruk\n```bash\npython -m venv .venv\nsource .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1\npip install -r requirements_min.txt\npython rf_inference.py\n```\n\n## Elastisitet (RF punktvis)\nMean: {summary.get('mean')}  Median: {summary.get('median')}  (n={summary.get('n_valid')})\nTolkning: 1% økning i befolkning ~ {summary.get('mean')}% endring i påstigninger (gj.snitt av lokale punkter).\n\n## Forutsetninger\n- Ikke-kausal modell; beskriver mønstre i historikk.\n- Punktvis elastisitet kan variere betydelig mellom perioder.\n\n## Reproduserbarhet\n- sklearn {sklearn.__version__}\n- hash modell: {model_sha}\n\n"""
    readme_path = dist_dir / 'README_MODEL.md'
    readme_path.write_text(readme_text, encoding='utf-8')

    # Kopier nødvendige filer til dist_dir for zipping
    # (Bruk siste modell og feature metadata)
    # Lag en "flat" kopi av modell og feature metadata hvis ikke allerede
    flat_model_path = dist_dir / 'rf_model.joblib'
    if flat_model_path.exists():
        try:
            flat_model_path.unlink()
        except Exception:
            pass
    # Kopier binært
    flat_model_path.write_bytes(model_bytes)
    flat_feat_path = dist_dir / 'rf_feature_cols.json'
    flat_feat_path.write_text(feature_meta_path.read_text(encoding='utf-8'), encoding='utf-8')
    # Kopier manifest
    manifest_copy_path = dist_dir / 'rf_model_manifest.json'
    manifest_copy_path.write_text(manifest_path.read_text(encoding='utf-8'), encoding='utf-8')
    # Lag integritetshashfil samlet
    integrity_path = dist_dir / 'integrity_sha256.txt'
    integrity_path.write_text(f"rf_model.joblib  SHA256  {model_sha}\n", encoding='utf-8')

    # Zip
    zip_name = dist_dir / f'rf_model_package_{ts}.zip'
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [flat_model_path, flat_feat_path, inference_path, req_path, readme_path, manifest_copy_path, integrity_path]:
            if p.exists():
                zf.write(p, arcname=p.name)
        # Legg ved versjonert modell og hash
        zf.write(model_path_versioned, arcname=model_path_versioned.name)
        zf.write(hash_path, arcname=hash_path.name)

    print(f"Modellpakke skrevet -> {zip_name}")

if __name__ == '__main__':
    main()
