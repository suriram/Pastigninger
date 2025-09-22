"""Konsollverktøy for å predikere påstigninger.
Bruk:
  python predict.py --år 2026 --kvartall 2 --innbyggere 270000
Valgfritt: --modell artifacts/modell_pastigninger.joblib
"""
import argparse
import json
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

DATA_PATH = Path('reg.csv')

ARTIFACT_DIR = Path('artifacts')


def load_model(path: Path):
    return joblib.load(path)


def load_feature_cols():
    p = ARTIFACT_DIR / 'feature_cols.json'
    if not p.exists():
        raise FileNotFoundError('feature_cols.json mangler – kjør main.py først.')
    return json.loads(p.read_text(encoding='utf-8'))


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    num_cols = ['år', 'kvartall', 'påstigninger', 'anall innbyggere']
    for col in num_cols:
        df[col] = (df[col].astype(str)
                             .str.replace('"', '', regex=False)
                             .str.replace(',', '', regex=False)
                             .str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)
    df = df.astype({'år': int, 'kvartall': int, 'påstigninger': int, 'anall innbyggere': int})
    return df


def compute_t_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['år', 'kvartall']).copy()
    df['t_index'] = range(1, len(df) + 1)
    return df


def build_feature_row(år: int, kvartall: int, innbyggere: int, delomrade: str | None, feature_cols: List[str], hist_df: pd.DataFrame):
    hist_sorted = hist_df.sort_values(['år', 'kvartall'])
    # Global t_index
    match_global = hist_sorted[(hist_sorted['år'] == år) & (hist_sorted['kvartall'] == kvartall)]
    if not match_global.empty:
        t_index = match_global['t_index'].iloc[0]
    else:
        t_index = hist_sorted['t_index'].max() + 1
    # Per-område t_index_area
    t_index_area = None
    if delomrade is not None and 't_index_area' in hist_sorted.columns:
        sub = hist_sorted[hist_sorted['delmarkedsområde'].astype(str) == str(delomrade)]
        match_area = sub[(sub['år'] == år) & (sub['kvartall'] == kvartall)]
        if not match_area.empty:
            t_index_area = match_area['t_index_area'].iloc[0]
        else:
            t_index_area = (sub['t_index_area'].max() + 1) if not sub.empty else 1
    row = {
        'år': år,
        'kvartall': kvartall,
        'anall innbyggere': innbyggere,
        't_index': t_index,
        'sin_q': np.sin(2 * np.pi * (kvartall / 4.0)),
        'cos_q': np.cos(2 * np.pi * (kvartall / 4.0)),
    }
    if t_index_area is not None:
        row['t_index_area'] = t_index_area
    for q in [1,2,3,4]:
        row[f'Q_{q}'] = 1 if kvartall == q else 0
    # Area dummies
    if delomrade is not None:
        area_col = f'AREA_{delomrade}'
        row[area_col] = 1
    # Fyll ut manglende
    for f in feature_cols:
        if f not in row:
            row[f] = 0
    return pd.DataFrame([row])[feature_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--år', type=int, required=True)
    parser.add_argument('--kvartall', type=int, required=True, choices=[1,2,3,4])
    parser.add_argument('--innbyggere', type=int, required=True)
    parser.add_argument('--modell', type=str, default=str(ARTIFACT_DIR / 'modell_pastigninger.joblib'))
    parser.add_argument('--delomrade', type=str, help='Angi delmarkedsområde (samme verdi som i data)')
    parser.add_argument('--vis-faktisk', action='store_true', help='Slå opp faktisk verdi i historikken hvis tilgjengelig')
    args = parser.parse_args()

    feature_cols = load_feature_cols()
    model = load_model(Path(args.modell))

    # Les historikk for korrekt t_index + ev. faktisk verdi
    if not DATA_PATH.exists():
        raise FileNotFoundError('Finner ikke reg.csv for historisk kontekst.')
    raw = pd.read_csv(DATA_PATH)
    raw_clean = clean_raw(raw)
    hist_df = compute_t_index(raw_clean)

    X_row = build_feature_row(args.år, args.kvartall, args.innbyggere, args.delomrade, feature_cols, hist_df)
    pred = model.predict(X_row)[0]

    print('\nPrediksjon:')
    area_txt = f" Delområde={args.delomrade}" if args.delomrade else ""
    print(f"  År={args.år} Kvartall={args.kvartall} Innbyggere={args.innbyggere}{area_txt}")
    print(f"  Estimert påstigninger: {pred:,.0f}")
    if args.vis_faktisk:
        m = hist_df[(hist_df['år'] == args.år) & (hist_df['kvartall'] == args.kvartall)]
        if not m.empty:
            print(f"  Faktisk historisk: {m['påstigninger'].iloc[0]:,}")
        else:
            print("  (Ingen historisk verdi – fremtidspunkt)")
    print('\n(t_index beregnet dynamisk fra historikk)')


if __name__ == '__main__':
    main()
