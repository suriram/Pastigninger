"""
Avansert versjon: tren, tun og generer prognoser for påstigninger.

Steg:
1. Leser og renser data
2. Lager ekstra features (år, sesong-dummies, kontinuerlig tidsindeks)
3. Kjører baseline modeller
4. Hyperparameter-tuner Random Forest og Gradient Boosting (grid search)
5. Velger beste modell og trener på hele datasettet
6. Lagrer modell og genererer scenarioprediksjoner
7. (Valgfritt) Bootstrap-intervaller (enkel implementasjon)

Output:
- Konsoll: resultattabell + valgte parametre
- Fil: model_report.txt, modell .joblib, scenario_predictions.csv
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.base import clone
import joblib
import shap

DATA_PATH = Path("reg.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    # Fjern aggregert område (Area 12) hvis kolonnen finnes (skal ikke brukes i modellering / analyser)
    if 'delmarkedsområde' in df.columns:
        df['delmarkedsområde'] = df['delmarkedsområde'].astype(str).str.strip()
        df = df[df['delmarkedsområde'] != '12']
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """Enkel datavalidering: sjekk duplikater, negative verdier, outliers (IQR), manglende områder."""
    issues = {}
    # Duplikatnøkler (år, kvartall, delmarkedsområde)
    key_cols = [c for c in ['delmarkedsområde','år','kvartall'] if c in df.columns]
    if key_cols:
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        dups = df[dup_mask]
        if not dups.empty:
            issues['duplicates'] = len(dups)
    # Negative tall
    for col in ['påstigninger','anall innbyggere']:
        if col in df.columns and (df[col] < 0).any():
            issues[f'negative_{col}'] = int((df[col] < 0).sum())
    # Outliers (IQR) for påstigninger
    if 'påstigninger' in df.columns:
        q1 = df['påstigninger'].quantile(0.25)
        q3 = df['påstigninger'].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            upper = q3 + 1.5 * iqr
            lower = q1 - 1.5 * iqr
            out_count = int(((df['påstigninger'] > upper) | (df['påstigninger'] < lower)).sum())
            if out_count:
                issues['outliers_påstigninger'] = out_count
    # Manglende sekvens i kvartall per år
    gaps = []
    for (år), grp in df.groupby('år'):
        expected = set([1,2,3,4])
        present = set(grp['kvartall'].unique())
        missing = expected - present
        if missing:
            gaps.append({'år': år, 'manglende_kvartal': sorted(missing)})
    if gaps:
        issues['missing_quarters'] = gaps
    # Returner + skriv rapport
    report_lines = ["DATA VALIDERING:"]
    if not issues:
        report_lines.append("Ingen problemer funnet.")
    else:
        for k,v in issues.items():
            report_lines.append(f"- {k}: {v}")
    (ARTIFACT_DIR / 'validation_report.txt').write_text("\n".join(report_lines), encoding='utf-8')
    return issues


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Kontinuerlig tidsindeks (sorter for sikkerhet)
    df = df.sort_values(['år', 'kvartall'])
    df['t_index'] = range(1, len(df) + 1)
    # Per-område tidsindeks (teller separat innen hvert delmarkedsområde hvis kolonnen finnes)
    if 'delmarkedsområde' in df.columns:
        df['delmarkedsområde'] = df['delmarkedsområde'].astype(str).str.strip()
        df['t_index_area'] = df.groupby('delmarkedsområde').cumcount() + 1
    # Sesong (one-hot)
    season_dummies = pd.get_dummies(df['kvartall'].astype(int), prefix='Q')
    df = pd.concat([df, season_dummies], axis=1)
    # Sinus/cos for jevn sesong
    df['sin_q'] = np.sin(2 * np.pi * (df['kvartall'] / 4.0))
    df['cos_q'] = np.cos(2 * np.pi * (df['kvartall'] / 4.0))
    # One-hot for delmarkedsområde
    if 'delmarkedsområde' in df.columns:
        area_dummies = pd.get_dummies(df['delmarkedsområde'], prefix='AREA')
        df = pd.concat([df, area_dummies], axis=1)
    return df


def train_baselines(df: pd.DataFrame):
    feature_cols = ['anall innbyggere', 'år', 't_index', 'sin_q', 'cos_q']
    if 't_index_area' in df.columns:
        feature_cols.append('t_index_area')
    # Legg til one-hot kvartal (Q_1..Q_4) i tillegg
    feature_cols.extend([c for c in df.columns if c.startswith('Q_')])
    # Legg til area-dummies (AREA_*)
    feature_cols.extend([c for c in df.columns if c.startswith('AREA_')])
    X = df[feature_cols]
    y = df['påstigninger']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        'Lineær regresjon': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        r2 = r2_score(y_test, pred)
        # Eldre sklearn-versjon støtter ikke squared=False
        rmse = mean_squared_error(y_test, pred) ** 0.5
        results.append((name, r2, rmse, m))
    return results, X, y, feature_cols


def tune_models(X, y):
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    rf_grid = {
        'n_estimators': [400, 700],
        'max_depth': [None, 8],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 0.8]
    }
    gb_grid = {
        'n_estimators': [400, 700],
        'learning_rate': [0.05, 0.1],
        'max_depth': [2, 3],
        'subsample': [0.9, 1.0]
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=5, scoring='r2', n_jobs=-1)
    gb_search = GridSearchCV(gb, gb_grid, cv=5, scoring='r2', n_jobs=-1)
    rf_search.fit(X, y)
    gb_search.fit(X, y)
    return rf_search, gb_search


def select_final(rf_search, gb_search):
    rf_score = rf_search.best_score_
    gb_score = gb_search.best_score_
    if gb_score >= rf_score:
        return 'Gradient Boosting', gb_search.best_estimator_, gb_score, gb_search.best_params_
    else:
        return 'Random Forest', rf_search.best_estimator_, rf_score, rf_search.best_params_


def bootstrap_intervals(model_class, X, y, X_future, n=150, random_state=42):
    rng = np.random.default_rng(random_state)
    preds = []
    for _ in range(n):
        idx = rng.integers(0, len(X), len(X))
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        m = model_class()
        m.fit(Xb, yb)
        preds.append(m.predict(X_future))
    arr = np.vstack(preds)
    mean_ = arr.mean(axis=0)
    low = np.percentile(arr, 5, axis=0)
    high = np.percentile(arr, 95, axis=0)
    return mean_, low, high


def build_scenarios(base_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # Utvidet: lag scenarier per delmarkedsområde (hvis kolonnen finnes)
    has_area = 'delmarkedsområde' in base_df.columns
    future_rows = []
    if has_area:
        areas = sorted(base_df['delmarkedsområde'].astype(str).unique())
    else:
        areas = [None]

    last_year = base_df['år'].max()
    last_q = base_df[base_df['år'] == last_year]['kvartall'].max()
    t_start_global = base_df['t_index'].max()

    for area in areas:
        sub = None
        if has_area:
            sub = base_df[base_df['delmarkedsområde'] == area]
            # Bruk per-område trend hvis mulig, ellers global
            pop_src = sub if len(sub) >= 3 else base_df
        else:
            pop_src = base_df
        pop = pop_src[['t_index', 'anall innbyggere']].drop_duplicates()
        try:
            pop_fit = np.polyfit(pop['t_index'], pop['anall innbyggere'], 1)
            slope = pop_fit[0]
            intercept = pop_fit[1]
        except Exception:
            slope = 0.0
            intercept = pop['anall innbyggere'].iloc[-1]
        # t_index_area start
        if has_area and sub is not None and 't_index_area' in base_df.columns:
            t_area_start = sub['t_index_area'].max()
        else:
            t_area_start = None
        for step in range(1, 5):
            q = (last_q + step - 1) % 4 + 1
            year_add = (last_q + step - 1) // 4
            year = last_year + year_add
            t_val = t_start_global + step
            base_pop = slope * t_val + intercept
            for mult, label in [(0.0, 'basis'), (0.02, 'pluss2pct'), (-0.02, 'minus2pct')]:
                adj_pop = int(base_pop * (1 + mult))
                row = {
                    'år': year,
                    'kvartall': q,
                    't_index': t_val,
                    'anall innbyggere': adj_pop,
                    'scenario': label
                }
                if has_area:
                    row['delmarkedsområde'] = area
                    if t_area_start is not None:
                        row['t_index_area'] = t_area_start + step
                future_rows.append(row)
    fut = pd.DataFrame(future_rows)
    # Legg til trig/season
    fut['sin_q'] = np.sin(2 * np.pi * (fut['kvartall'] / 4.0))
    fut['cos_q'] = np.cos(2 * np.pi * (fut['kvartall'] / 4.0))
    for q in [1, 2, 3, 4]:
        fut[f'Q_{q}'] = (fut['kvartall'] == q).astype(int)
    if has_area:
        area_dummies = pd.get_dummies(fut['delmarkedsområde'].astype(str), prefix='AREA')
        fut = pd.concat([fut, area_dummies], axis=1)
    # Sikre alle feature_cols
    for c in feature_cols:
        if c not in fut.columns:
            fut[c] = 0
    return fut


def write_report(baseline_results, final_name, final_score, final_params, feature_cols, model_version: str, top_features: list[tuple[str, float]] | None = None, elasticity_summary: dict | None = None):
    lines = []
    lines.append("BASELINE MODELLER:")
    for name, r2, rmse, _ in baseline_results:
        lines.append(f"  {name:18} R²={r2:.3f} RMSE={rmse:.0f}")
    lines.append("")
    lines.append("VALGT MODELL:")
    lines.append(f"  {final_name} (CV R²={final_score:.3f})")
    lines.append(f"  Versjon: {model_version}")
    lines.append(f"  Parametre: {final_params}")
    lines.append("")
    lines.append("FEATURES BRUKT:")
    for f in feature_cols:
        lines.append(f"  - {f}")
    if top_features:
        lines.append("")
        lines.append("TOPP FEATURES (SHAP gj.snitt abs, topp 10):")
        for name, val in top_features[:10]:
            lines.append(f"  {name:15} {val:,.0f}")
    if elasticity_summary:
        lines.append("")
        lines.append("BEFOLKNINGSELASTISITET:")
        try:
            lines.append(f"  Gyldige observasjoner: {elasticity_summary.get('n_valid')}")
            lines.append(f"  Mean elastisitet: {elasticity_summary.get('mean'):.4f}")
            lines.append(f"  Median: {elasticity_summary.get('median'):.4f} (P25={elasticity_summary.get('p25'):.4f}, P75={elasticity_summary.get('p75'):.4f})")
            if 'linear_coeff' in elasticity_summary and elasticity_summary.get('linear_coeff') is not None:
                lines.append(f"  Lineær modell koeff (dY/dPop): {elasticity_summary.get('linear_coeff'):.6f}")
            if 'linear_mean_elasticity_at_means' in elasticity_summary and elasticity_summary.get('linear_mean_elasticity_at_means') is not None:
                lines.append(f"  Lineær elastisitet (ved gjennomsnitt): {elasticity_summary.get('linear_mean_elasticity_at_means'):.4f}")
            lines.append("  Tolkning: En elastisitet på 0.80 betyr at 1% økning i befolkning gir ca 0.8% økning i påstigninger gitt dagens nivå.")
        except Exception as e:
            lines.append(f"  (Kunne ikke skrive elastisitetsseksjon: {e})")
    # Område-metrikker hvis eksisterer
    area_metrics_path = ARTIFACT_DIR / 'area_metrics.csv'
    if area_metrics_path.exists():
        try:
            import pandas as pd
            am = pd.read_csv(area_metrics_path)
            if not am.empty:
                lines.append("")
                lines.append("PER-OMRÅDE METRIKKER (R² / RMSE):")
                # sorter lavest R2 først
                am_sorted = am.sort_values('r2')
                for _, r in am_sorted.iterrows():
                    lines.append(f"  Område {r['delmarkedsområde']}: R²={r['r2']:.3f} RMSE={r['rmse']:.0f} N={int(r['n'])}")
        except Exception as e:
            lines.append(f"(Kunne ikke lese area_metrics.csv: {e})")
    (ARTIFACT_DIR / 'model_report.txt').write_text("\n".join(lines), encoding='utf-8')


def compute_and_save_shap(model, X: pd.DataFrame):
    """Beregner SHAP verdier for tre-baserte modeller og lagrer importance.
       Returnerer (shap_values, importance_df) eller (None, None) ved feil."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        vals = np.abs(shap_values).mean(axis=0)
        imp = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': vals}) \
                .sort_values('mean_abs_shap', ascending=False)
        imp.to_csv(ARTIFACT_DIR / 'shap_importance.csv', index=False, encoding='utf-8')
        # Summary plot
        try:
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            plt.savefig(ARTIFACT_DIR / 'shap_summary.png', dpi=120)
            plt.close()
        except Exception as e:
            print(f"Kunne ikke lage shap_summary.png: {e}")
        print("SHAP importance lagret -> artifacts/shap_importance.csv")
        return shap_values, imp
    except Exception as e:
        print(f"SHAP beregning hoppet over (feil): {e}")
        return None, None


def enrich_global_shap(imp: pd.DataFrame):
    total = imp['mean_abs_shap'].sum()
    if total <= 0:
        imp['pct'] = 0.0
    else:
        imp['pct'] = imp['mean_abs_shap'] / total * 100.0
    imp.to_csv(ARTIFACT_DIR / 'shap_importance_enriched.csv', index=False, encoding='utf-8')
    return imp


def per_area_shap(shap_values, X: pd.DataFrame):
    """Aggregerer gj.snitt abs SHAP per feature innen hvert område (hvis area-dummies finnes)."""
    area_cols = [c for c in X.columns if c.startswith('AREA_')]
    if not area_cols or shap_values is None:
        return None
    # shap_values shape: (n_samples, n_features)
    sv = np.abs(shap_values)
    feat_names = list(X.columns)
    rows = []
    # Finn mapping rad->område basert på hvilken AREA_* er 1, hvis flere -> første
    area_assign = []
    area_matrix = X[area_cols].values
    for r in area_matrix:
        if r.sum() == 1:
            idx = int(np.where(r == 1)[0][0])
            area_assign.append(area_cols[idx].replace('AREA_', ''))
        else:
            area_assign.append('NA')
    area_assign = np.array(area_assign)
    for area in sorted(set(area_assign)):
        mask = area_assign == area
        if mask.sum() == 0:
            continue
        mean_abs = sv[mask].mean(axis=0)
        df_area = pd.DataFrame({
            'area': area,
            'feature': feat_names,
            'mean_abs_shap': mean_abs
        })
        rows.append(df_area)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ARTIFACT_DIR / 'shap_per_area.csv', index=False, encoding='utf-8')
    return out


def generate_pdp(model, X: pd.DataFrame, features: list[str], prefix: str = 'pdp'):
    """Generer enkel partial dependence for utvalgte features og lagre som CSV.
       Bruker sklearn.inspection.partial_dependence (kan være treg ved mange punkter)."""
    rows = []
    for feat in features:
        if feat not in X.columns:
            continue
        try:
            disp = PartialDependenceDisplay.from_estimator(model, X, [feat])
            # Uthent data fra lines_ (støttet av stable sklearn versjoner)
            line = disp.lines_[0][0]
            xs = line.get_xdata()
            ys = line.get_ydata()
            for v, avg in zip(xs, ys):
                rows.append({'feature': feat, 'value': float(v), 'pdp': float(avg)})
        except Exception as e:
            rows.append({'feature': feat, 'value': None, 'pdp': None, 'error': str(e)})
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(ARTIFACT_DIR / f'{prefix}_values.csv', index=False, encoding='utf-8')


def save_feature_metadata(feature_cols):
    (ARTIFACT_DIR / 'feature_cols.json').write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding='utf-8')
    # Lagre liste over delmarkedsområder hvis tilgjengelig (fra feature navn)
    areas = sorted({c.replace('AREA_', '') for c in feature_cols if c.startswith('AREA_')})
    if areas:
        (ARTIFACT_DIR / 'areas.json').write_text(json.dumps(areas, ensure_ascii=False, indent=2), encoding='utf-8')


# ----------------- Nye avanserte analysetrinn (A-D) -----------------
def compute_out_of_time(X: pd.DataFrame, y: pd.Series, n_holdout_quarters: int = 4, index_order: pd.Index | None = None):
    """Lag en enkel out-of-time holdout ved å ta de siste n kvartaler.
       Returnerer dict med metrics + trenings/holdout-indekser.
    """
    if index_order is None:
        index_order = X.index
    ordered = list(index_order)
    if len(ordered) <= n_holdout_quarters + 5:
        # For lite data – fallback: siste 2
        n_hold = min(2, max(1, len(ordered)//5))
    else:
        n_hold = n_holdout_quarters
    hold_idx = ordered[-n_hold:]
    train_idx = ordered[:-n_hold]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_hold, y_hold = X.loc[hold_idx], y.loc[hold_idx]
    return {
        'train_idx': train_idx,
        'hold_idx': hold_idx,
        'X_train': X_train,
        'y_train': y_train,
        'X_hold': X_hold,
        'y_hold': y_hold
    }


def evaluate_out_of_time(estimator, oot_split: dict):
    """Tren klon av estimator på treningsdel og evaluer på holdout."""
    est = clone(estimator)
    est.fit(oot_split['X_train'], oot_split['y_train'])
    preds = est.predict(oot_split['X_hold'])
    if len(set(oot_split['y_hold'])) > 1:
        r2 = r2_score(oot_split['y_hold'], preds)
    else:
        r2 = float('nan')
    rmse = mean_squared_error(oot_split['y_hold'], preds) ** 0.5
    return {'oot_r2': r2, 'oot_rmse': rmse, 'n_holdout': len(oot_split['y_hold'])}


def drop_column_bootstrap(estimator, X: pd.DataFrame, y: pd.Series, oot_split: dict, features: list[str], n_boot: int = 30, random_state: int = 42, max_features: int = 15):
    """Bootstrap-basert drop-column importance: mål hvor mye OOT R² faller når feature fjernes.
       Returnerer DataFrame med mean/median/p05/p95 delta R².
    """
    rng = np.random.default_rng(random_state)
    target_feats = features[:max_features]
    results = {f: [] for f in target_feats}
    X_train_full = oot_split['X_train']
    y_train_full = oot_split['y_train']
    X_hold = oot_split['X_hold']
    y_hold = oot_split['y_hold']
    # Precompute baseline each bootstrap
    for b in range(n_boot):
        idx = rng.integers(0, len(X_train_full), len(X_train_full))
        Xb = X_train_full.iloc[idx]
        yb = y_train_full.iloc[idx]
        base_est = clone(estimator)
        base_est.fit(Xb, yb)
        base_pred = base_est.predict(X_hold)
        if len(set(y_hold)) > 1:
            base_r2 = r2_score(y_hold, base_pred)
        else:
            base_r2 = float('nan')
        for f in target_feats:
            # Dropp kolonne
            if f not in Xb.columns:
                continue
            Xb_drop = Xb.drop(columns=[f])
            X_hold_drop = X_hold.drop(columns=[f]) if f in X_hold.columns else X_hold
            est_drop = clone(estimator)
            est_drop.fit(Xb_drop, yb)
            pred_drop = est_drop.predict(X_hold_drop)
            if len(set(y_hold)) > 1:
                r2_drop = r2_score(y_hold, pred_drop)
            else:
                r2_drop = float('nan')
            delta = base_r2 - r2_drop
            results[f].append(delta)
    rows = []
    for f, deltas in results.items():
        if not deltas:
            continue
        arr = np.array(deltas)
        rows.append({
            'feature': f,
            'mean_delta_r2': float(np.nanmean(arr)),
            'median_delta_r2': float(np.nanmedian(arr)),
            'p05_delta_r2': float(np.nanpercentile(arr, 5)),
            'p95_delta_r2': float(np.nanpercentile(arr, 95)),
            'n_boot': len(arr)
        })
    if not rows:
        return None
    import pandas as pd
    out = pd.DataFrame(rows).sort_values('mean_delta_r2', ascending=False)
    out.to_csv(ARTIFACT_DIR / 'drop_column_importance.csv', index=False, encoding='utf-8')
    return out


def shap_bootstrap_intervals(estimator, X: pd.DataFrame, y: pd.Series, n_boot: int = 30, sample_frac: float = 0.75, random_state: int = 42):
    try:
        rng = np.random.default_rng(random_state)
        feats = list(X.columns)
        vals_collect = {f: [] for f in feats}
        for b in range(n_boot):
            idx = rng.choice(len(X), size=max(5, int(len(X)*sample_frac)), replace=True)
            Xb = X.iloc[idx]
            yb = y.iloc[idx]
            est = clone(estimator)
            est.fit(Xb, yb)
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(Xb)
            abs_mean = np.abs(sv).mean(axis=0)
            for f, v in zip(feats, abs_mean):
                vals_collect[f].append(v)
        rows = []
        for f, arr in vals_collect.items():
            if not arr:
                continue
            a = np.array(arr)
            rows.append({
                'feature': f,
                'shap_mean': float(a.mean()),
                'shap_std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                'p05': float(np.percentile(a, 5)),
                'p95': float(np.percentile(a, 95)),
                'n_boot': len(a)
            })
        import pandas as pd
        out = pd.DataFrame(rows).sort_values('shap_mean', ascending=False)
        out.to_csv(ARTIFACT_DIR / 'shap_importance_bootstrap.csv', index=False, encoding='utf-8')
        return out
    except Exception as e:
        print(f"Kunne ikke beregne SHAP bootstrap intervaller: {e}")
        return None


def residual_reports(model, X: pd.DataFrame, y: pd.Series, prefix: str = 'residuals'):
    preds = model.predict(X)
    resid = y - preds
    df_res = pd.DataFrame({
        'påstigninger': y,
        'pred': preds,
        'residual': resid
    }, index=X.index)
    df_res.to_csv(ARTIFACT_DIR / f'{prefix}.csv', encoding='utf-8')
    # Aggregert statistikk
    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5
    mape = float(np.mean(np.abs(resid) / np.where(y != 0, y, np.nan))) * 100.0
    bias = float(resid.mean())
    stats = {
        'rmse': rmse,
        'mape_pct': mape,
        'bias': bias
    }
    (ARTIFACT_DIR / 'residual_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return df_res, stats


def compute_population_elasticity(model, X: pd.DataFrame, pop_feature: str = 'anall innbyggere'):
    """Beregner lokal (punktvis) befolkningselastisitet for modellen.
    Elastisitet per observasjon i: (dY/dPop) * (Pop / Y).
    dY/dPop estimeres med sentral differanse (+/- 1% av Pop, min 1).
    Returnerer (df_elasticity, summary_dict) eller (None, None) hvis feature mangler.
    """
    if pop_feature not in X.columns:
        return None, None
    try:
        preds = model.predict(X)
    except Exception as e:
        print(f"Kunne ikke beregne elastisitet (predict-feil): {e}")
        return None, None
    records = []
    for i, (pop, y_hat) in enumerate(zip(X[pop_feature].values, preds)):
        pop_val = float(pop)
        y_val = float(y_hat)
        if pop_val <= 0 or y_val <= 0:
            records.append({
                'index': X.index[i],
                pop_feature: pop_val,
                'pred': y_val,
                'elasticity': np.nan
            })
            continue
        delta = max(1.0, pop_val * 0.01)  # 1% av pop, minst 1
        # Lag plus/minus kopier
        row_plus = X.iloc[i].copy()
        row_minus = X.iloc[i].copy()
        row_plus[pop_feature] = pop_val + delta
        row_minus[pop_feature] = max(0.0, pop_val - delta)
        try:
            y_plus = model.predict(pd.DataFrame([row_plus], columns=X.columns))[0]
            y_minus = model.predict(pd.DataFrame([row_minus], columns=X.columns))[0]
            denom = (row_plus[pop_feature] - row_minus[pop_feature])
            if denom == 0:
                deriv = np.nan
            else:
                deriv = (y_plus - y_minus) / denom
            elasticity = deriv * (pop_val / y_val) if (deriv is not None and y_val != 0) else np.nan
        except Exception:
            y_plus = np.nan
            y_minus = np.nan
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
    el_df = pd.DataFrame(records)
    el_path = ARTIFACT_DIR / 'population_elasticity.csv'
    el_df.to_csv(el_path, index=False, encoding='utf-8')
    valid = el_df['elasticity'].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        summary = {
            'n_valid': 0,
            'mean': None,
            'median': None,
            'p25': None,
            'p75': None
        }
    else:
        summary = {
            'n_valid': int(valid.shape[0]),
            'mean': float(valid.mean()),
            'median': float(valid.median()),
            'p25': float(valid.quantile(0.25)),
            'p75': float(valid.quantile(0.75))
        }
    (ARTIFACT_DIR / 'population_elasticity_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return el_df, summary


def compute_population_elasticity_by_area(elasticity_df: pd.DataFrame, df_original: pd.DataFrame, area_col: str = 'delmarkedsområde'):
    """Aggreger elastisitet per delmarkedsområde ved å koble index tilbake til original df.
       Forutsetter at elasticity_df['index'] matcher X.index brukt tidligere.
    """
    if area_col not in df_original.columns:
        return None
    try:
        tmp = elasticity_df.merge(df_original[[area_col]].reset_index().rename(columns={'index':'orig_index'}), left_on='index', right_on='orig_index', how='left')
        grp = tmp.groupby(area_col, dropna=False)['elasticity']
        rows = []
        for area, series in grp:
            v = series.replace([np.inf, -np.inf], np.nan).dropna()
            if v.empty:
                rows.append({area_col: area, 'n_valid': 0, 'mean': None, 'median': None, 'p25': None, 'p75': None})
            else:
                rows.append({
                    area_col: area,
                    'n_valid': int(v.shape[0]),
                    'mean': float(v.mean()),
                    'median': float(v.median()),
                    'p25': float(v.quantile(0.25)),
                    'p75': float(v.quantile(0.75))
                })
        out = pd.DataFrame(rows)
        out_path = ARTIFACT_DIR / 'population_elasticity_by_area.csv'
        out.to_csv(out_path, index=False, encoding='utf-8')
        return out
    except Exception as e:
        print(f"Kunne ikke beregne per-område elastisitet: {e}")
        return None


def compute_scenario_elasticity(model, scenario_df: pd.DataFrame, feature_cols: list[str], pop_feature: str = 'anall innbyggere'):
    """Beregner elastisitet for scenariopunktene på samme måte, men bruker scenariorader.
       Forventer at 'pred' allerede finnes; hvis ikke predikerer vi først.
    """
    if pop_feature not in scenario_df.columns:
        return None
    scen = scenario_df.copy()
    if 'pred' not in scen.columns:
        scen['pred'] = model.predict(scen[feature_cols])
    records = []
    for i, row in scen.iterrows():
        pop_val = float(row[pop_feature])
        y_val = float(row['pred'])
        if pop_val <= 0 or y_val <= 0:
            records.append({'row_id': i, 'scenario': row.get('scenario'), pop_feature: pop_val, 'pred': y_val, 'elasticity': np.nan})
            continue
        delta = max(1.0, pop_val * 0.01)
        r_plus = row.copy()
        r_minus = row.copy()
        r_plus[pop_feature] = pop_val + delta
        r_minus[pop_feature] = max(0.0, pop_val - delta)
        try:
            y_plus = model.predict(pd.DataFrame([r_plus[feature_cols]], columns=feature_cols))[0]
            y_minus = model.predict(pd.DataFrame([r_minus[feature_cols]], columns=feature_cols))[0]
            denom = (r_plus[pop_feature] - r_minus[pop_feature])
            deriv = (y_plus - y_minus) / denom if denom != 0 else np.nan
            elast = deriv * (pop_val / y_val) if (deriv is not None and y_val != 0) else np.nan
        except Exception:
            y_plus = np.nan
            y_minus = np.nan
            deriv = np.nan
            elast = np.nan
        records.append({
            'row_id': i,
            'scenario': row.get('scenario'),
            pop_feature: pop_val,
            'pred': y_val,
            'delta_pop': delta,
            'pred_plus': y_plus,
            'pred_minus': y_minus,
            'derivative_dY_dPop': deriv,
            'elasticity': elast
        })
    out_df = pd.DataFrame(records)
    out_df.to_csv(ARTIFACT_DIR / 'population_elasticity_scenarios.csv', index=False, encoding='utf-8')
    return out_df


def write_elasticity_report(global_summary: dict | None, area_df: pd.DataFrame | None, scen_df: pd.DataFrame | None):
    lines = ["BEFOLKNINGSELASTISITET – METODER OG RESULTATER", ""]
    lines.append("Metode:")
    lines.append("  Elastisitet beregnes punktvis som (dY/dPop)*(Pop/Y) der dY/dPop estimeres via sentral differanse med ±1% av befolkningen (minst 1).")
    lines.append("  Scenarioversjonen bruker samme metode på framtidsscenariorader.")
    lines.append("  Per-område resultat aggregerer median og kvartiler av gyldige elastisiteter.")
    lines.append("")
    lines.append("Forklaring / tolkning:")
    lines.append("  - 'Globalt sammendrag' er IKKE en egen modell uten områder, men et aggregat (samlet for alle punktvise elastisiteter i datasettet).")
    lines.append("  - 'mean' kan trekkes opp av enkelte høye verdier; 'median' gir ofte et mer robust uttrykk for typisk respons.")
    lines.append("  - p25/p75 viser interkvartilspennet (variabilitet). Stor avstand indikerer heterogen effekt over tid/observasjoner.")
    lines.append("  - Negative elastisiteter kan oppstå når lokal modellstruktur eller støy gjør at marginalprediksjon synker ved økt befolkning.")
    lines.append("  - 'linear_coeff' er koeffisienten til befolkning fra lineær referansemodell (separat regressjon) og 'linear_mean_elasticity_at_means' er elastisitet beregnet ved å bruke denne lineære koeffisienten i datapunktet med gjennomsnittsverdier.")
    lines.append("  - Punktvis elastisitet er lokal (marginal) – gjelder små endringer (~1%). Ikke anta lineær skalering for store sjokk uten videre analyse.")
    lines.append("  - Per-område tall er beregnet etter at aggregert område (12) er filtrert bort dersom det fantes i data.")
    lines.append("  - Høy spredning mellom områder kan indikere strukturelle forskjeller eller datagrunnlag med ulik kvalitet.")
    lines.append("")
    lines.append("")
    if global_summary:
        lines.append("Globalt sammendrag:")
        for k in ['n_valid','mean','median','p25','p75','linear_coeff','linear_mean_elasticity_at_means']:
            if k in global_summary and global_summary[k] is not None:
                lines.append(f"  {k}: {global_summary[k]}")
        lines.append("")
    if area_df is not None and not area_df.empty:
        lines.append("Per-område (første 20 rader):")
        try:
            show = area_df.head(20)
            for _, r in show.iterrows():
                lines.append(f"  Område {r['delmarkedsområde']}: n={r['n_valid']} mean={r['mean']} median={r['median']} p25={r['p25']} p75={r['p75']}")
            lines.append("")
        except Exception as e:
            lines.append(f"  (Kunne ikke skrive per-område tabell: {e})")
    if scen_df is not None and not scen_df.empty:
        lines.append("Scenario-elastisitet (aggregert etter scenario):")
        try:
            # Robust iterasjon uten .apply -> groupby over serier
            clean = scen_df.copy()
            if 'elasticity' in clean.columns:
                clean['elasticity'] = clean['elasticity'].replace([np.inf, -np.inf], np.nan)
            for scen, grp in clean.groupby('scenario', dropna=False):
                ser = grp['elasticity'].dropna() if 'elasticity' in grp.columns else pd.Series(dtype=float)
                if ser.empty:
                    lines.append(f"  {scen}: ingen gyldige verdier")
                else:
                    lines.append(f"  {scen}: mean={ser.mean():.4f} median={ser.median():.4f} n={ser.shape[0]}")
        except Exception as e:
            lines.append(f"  (Kunne ikke aggregere scenario-elastisitet robust: {e})")
    (ARTIFACT_DIR / 'elasticity_report.txt').write_text("\n".join(lines), encoding='utf-8')


def organize_artifacts():
    """Flytt artefakter til underkataloger for bedre struktur.
    Idempotent: eksisterende filer i målmappe overskrives.
    """
    structure = {
        'models': [
            'modell_pastigninger.joblib', 'model_manifest.json'
        ],
        'models/versions': [
            # versjonsfiler matcher prefiks
        ],
        'scenarios': [
            'scenario_predictions.csv'
        ],
        'shap': [
            'shap_importance.csv', 'shap_importance_enriched.csv', 'shap_summary.png',
            'shap_per_area.csv', 'shap_importance_bootstrap.csv'
        ],
        'diagnostics': [
            'area_metrics.csv', 'drop_column_importance.csv', 'residuals.csv', 'residual_stats.json',
            'validation_report.txt', 'oot_metrics.json', 'pdp_values.csv'
        ],
        'elasticity': [
            'population_elasticity.csv', 'population_elasticity_summary.json', 'population_elasticity_by_area.csv',
            'population_elasticity_scenarios.csv', 'elasticity_report.txt'
        ],
        'reports': [
            'model_report.txt', 'linear_regression_equation.txt'
        ],
        'metadata': [
            'feature_cols.json', 'areas.json', 'linear_regression_coeffs.csv'
        ]
    }
    # Opprett mapper
    for folder in structure.keys():
        (ARTIFACT_DIR / folder).mkdir(parents=True, exist_ok=True)
    # Flytt filer
    import shutil
    for folder, files in structure.items():
        for fname in files:
            src = ARTIFACT_DIR / fname
            if src.exists():
                dest = ARTIFACT_DIR / folder / fname
                try:
                    shutil.move(str(src), str(dest))
                except Exception:
                    pass
    # Flytt versjonsfiler modell_pastigninger_*.joblib
    for f in ARTIFACT_DIR.glob('modell_pastigninger_*.joblib'):
        target = ARTIFACT_DIR / 'models' / 'versions' / f.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shutil
            shutil.move(str(f), str(target))
        except Exception:
            pass
    # Flytt scenario versjonsfiler
    for f in ARTIFACT_DIR.glob('scenario_predictions_*.csv'):
        target = ARTIFACT_DIR / 'scenarios' / f.name
        try:
            import shutil
            shutil.move(str(f), str(target))
        except Exception:
            pass
    # Indeksfil med kort oversikt
    index_lines = ["ARTIFACTS STRUKTUR:", ""]
    for folder in sorted(structure.keys()):
        index_lines.append(f"/{folder}")
        for fname in sorted((ARTIFACT_DIR / folder).iterdir()):
            if fname.is_file():
                index_lines.append(f"  - {fname.name}")
    (ARTIFACT_DIR / 'INDEX.txt').write_text("\n".join(index_lines), encoding='utf-8')
    print("Artifacts reorganisert (se artifacts/INDEX.txt)")



def load_custom_scenarios(path: Path, base_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Forventer CSV med kolonner: år, kvartall, anall innbyggere.
       Legger til nødvendige derived features for prediksjon."""
    if not path.exists():
        raise FileNotFoundError(f"Fant ikke scenario-fil: {path}")
    df = pd.read_csv(path)
    required = {'år', 'kvartall', 'anall innbyggere'}
    if not required.issubset(df.columns):
        raise ValueError(f"Scenario-fil må inneholde: {required}")
    df = df.copy()
    # Lag t_index fortsettende fra siste i base_df
    start_t = base_df['t_index'].max()
    df = df.sort_values(['år', 'kvartall'])
    df['t_index'] = range(start_t + 1, start_t + 1 + len(df))
    df['sin_q'] = np.sin(2 * np.pi * (df['kvartall'] / 4.0))
    df['cos_q'] = np.cos(2 * np.pi * (df['kvartall'] / 4.0))
    for q in [1,2,3,4]:
        df[f'Q_{q}'] = (df['kvartall'] == q).astype(int)
    # Sikre alle feature_cols
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df


def generate_version_id(model, params: dict, feature_cols: list[str]) -> str:
    """Lag en deterministisk kort versjons-ID basert på timestamp + hash av innhold."""
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    payload = json.dumps({
        'params': params,
        'features': feature_cols,
        'model_class': model.__class__.__name__
    }, sort_keys=True).encode('utf-8')
    h = hashlib.sha256(payload).hexdigest()[:8]
    return f"{ts}_{h}"


def load_manifest() -> list[dict]:
    m_path = ARTIFACT_DIR / 'model_manifest.json'
    if m_path.exists():
        try:
            return json.loads(m_path.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []


def save_manifest(entries: list[dict]):
    m_path = ARTIFACT_DIR / 'model_manifest.json'
    m_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    print("Laster og forbereder data...")
    raw = load_data(DATA_PATH)
    df = make_features(raw)
    print(f"Antall rader: {len(df)}")
    val_issues = validate_data(raw)

    print("\nKjører baseline modeller...")
    baseline_results, X, y, feature_cols = train_baselines(df)
    for name, r2, rmse, _ in baseline_results:
        print(f"  {name:18} R²={r2:.3f} RMSE={rmse:.0f}")

    # Lagre lineær regresjons koeffisienter og ligning for eksport
    try:
        lin_entry = next((t for t in baseline_results if t[0].startswith('Lineær')), None)
        if lin_entry is not None:
            _, lin_r2, lin_rmse, lin_model = lin_entry
            coeffs = lin_model.coef_
            intercept = lin_model.intercept_
            import pandas as _pd
            coef_df = _pd.DataFrame({'feature': X.columns, 'coefficient': coeffs})
            coef_df.to_csv(ARTIFACT_DIR / 'linear_regression_coeffs.csv', index=False, encoding='utf-8')
            # Bygg enkel ligning: y = a0 + a1*feat1 + ...
            terms = [f"({intercept:.6f})"]
            for f, c in zip(X.columns, coeffs):
                terms.append(f"({c:.6f})*{f}")
            equation = ' + '.join(terms)
            eq_lines = [
                'LINEÆR REGRESJON LIGNING',
                f'R2={lin_r2:.4f} RMSE={lin_rmse:.2f}',
                f'y = {equation}'
            ]
            (ARTIFACT_DIR / 'linear_regression_equation.txt').write_text("\n".join(eq_lines), encoding='utf-8')
            print("Lagret linear_regression_coeffs.csv og linear_regression_equation.txt")
    except Exception as e:
        print(f"Kunne ikke lagre lineær regresjonsligning: {e}")

    print("\nTuner modeller (kan ta litt tid)...")
    rf_search, gb_search = tune_models(X, y)
    final_name, final_model, final_score, final_params = select_final(rf_search, gb_search)
    print(f"Valgt modell: {final_name} (CV R²={final_score:.3f})")
    print(f"Parametre: {final_params}")

    # Out-of-time split & metrics (bruk beste estimator for referanse)
    print("\nBeregner out-of-time (OOT) metrics...")
    # Velg estimator fra search-objekt basert på final_name
    base_estimator = gb_search.best_estimator_ if final_name.startswith('Gradient') else rf_search.best_estimator_
    oot_split = compute_out_of_time(X, y, n_holdout_quarters=4, index_order=X.index)
    oot_metrics = evaluate_out_of_time(base_estimator, oot_split)
    print(f"OOT R²={oot_metrics['oot_r2']:.3f} RMSE={oot_metrics['oot_rmse']:.0f} (n={oot_metrics['n_holdout']})")

    print("\nTrener endelig modell på hele datasettet...")
    final_model.fit(X, y)

    # Beregn befolkningselastisitet for endelig modell
    print("\nBeregner befolkningselastisitet...")
    _, elasticity_summary = compute_population_elasticity(final_model, X, pop_feature='anall innbyggere')
    # Legg til lineær modell-informasjon dersom tilgjengelig
    try:
        lin_entry = next((t for t in baseline_results if t[0].startswith('Lineær')), None)
        if lin_entry is not None and elasticity_summary is not None:
            _, lin_r2_tmp, lin_rmse_tmp, lin_model_tmp = lin_entry
            try:
                coef_map = {f: c for f, c in zip(X.columns, lin_model_tmp.coef_)}
                pop_coef = coef_map.get('anall innbyggere')
                if pop_coef is not None:
                    mean_pop = float(X['anall innbyggere'].mean())
                    mean_pred_lin = float(lin_model_tmp.predict(X).mean())
                    if mean_pred_lin != 0:
                        lin_elast = pop_coef * (mean_pop / mean_pred_lin)
                    else:
                        lin_elast = None
                    elasticity_summary['linear_coeff'] = float(pop_coef)
                    elasticity_summary['linear_mean_elasticity_at_means'] = float(lin_elast) if lin_elast is not None else None
            except Exception:
                pass
    except Exception:
        pass

    # Versjonering
    version_id = generate_version_id(final_model, final_params, feature_cols)
    versioned_name = f"modell_pastigninger_{version_id}.joblib"  # unik fil
    versioned_path = ARTIFACT_DIR / versioned_name
    latest_path = ARTIFACT_DIR / 'modell_pastigninger.joblib'  # peker alltid til sist

    joblib.dump(final_model, versioned_path)
    # Oppdater 'latest'
    joblib.dump(final_model, latest_path)

    print(f"Modell lagret -> artifacts/{versioned_name} (og oppdatert modell_pastigninger.joblib)")

    # Oppdater manifest
    manifest = load_manifest()
    manifest_entry = {
        'version': version_id,
        'file': versioned_name,
        'created_utc': datetime.utcnow().isoformat() + 'Z',
        'model_type': final_name,
        'cv_r2': final_score,
        'oot_r2': oot_metrics.get('oot_r2'),
        'oot_rmse': oot_metrics.get('oot_rmse'),
        'params': final_params,
        'n_samples': int(len(X)),
        'features': feature_cols,
        'validation_issues': val_issues
    }
    manifest.append(manifest_entry)
    save_manifest(manifest)
    print("Manifest oppdatert -> artifacts/model_manifest.json")

    save_feature_metadata(feature_cols)
    print("Feature metadata lagret -> artifacts/feature_cols.json")

    print("\nLager scenarioprediksjoner...")
    fut = build_scenarios(df, feature_cols)
    fut['pred'] = final_model.predict(fut[feature_cols])

    # Bootstrap med samme model class som final (kun hvis RF/GB)
    if isinstance(final_model, (RandomForestRegressor, GradientBoostingRegressor)):
        model_class = final_model.__class__
        mean_, low, high = bootstrap_intervals(model_class, X, y, fut[feature_cols])
        fut['pred_mean_boot'] = mean_
        fut['pred_p05'] = low
        fut['pred_p95'] = high

    fut.to_csv(ARTIFACT_DIR / 'scenario_predictions.csv', index=False, encoding='utf-8')
    versioned_scen_name = f"scenario_predictions_{version_id}.csv"
    fut.to_csv(ARTIFACT_DIR / versioned_scen_name, index=False, encoding='utf-8')
    print(f"Scenario-filer lagret -> artifacts/scenario_predictions.csv og {versioned_scen_name}")

    # Scenario elastisitet
    print("Beregner scenario-elastisitet...")
    scen_el_df = compute_scenario_elasticity(final_model, fut.copy(), feature_cols, pop_feature='anall innbyggere')
    if scen_el_df is not None:
        print("Scenario-elastisitet lagret -> artifacts/population_elasticity_scenarios.csv")

    # SHAP (kun for tre-modeller)
    top_features_list = None
    if isinstance(final_model, (RandomForestRegressor, GradientBoostingRegressor)):
        print("Beregner SHAP verdier...")
        shap_values, imp = compute_and_save_shap(final_model, X)
        if imp is not None:
            imp_enriched = enrich_global_shap(imp.copy())
            top_features_list = list(zip(imp_enriched['feature'], imp_enriched['mean_abs_shap']))
        if shap_values is not None:
            per_area_shap(shap_values, X)
        # Partial dependence (PDP) for kjernefeatures
        core_feats = [c for c in ['anall innbyggere', 't_index', 't_index_area'] if c in X.columns]
        if core_feats:
            print("Genererer partial dependence (PDP)...")
            generate_pdp(final_model, X, core_feats)
            print("PDP lagret -> artifacts/pdp_values.csv")
        # SHAP bootstrap intervaller (C)
        print("Beregner SHAP bootstrap intervaller...")
        shap_bootstrap_intervals(final_model, X, y, n_boot=25, sample_frac=0.7)
        # Drop-column importance (A) – bruk OOT split
        print("Kjører drop-column bootstrap importance...")
        drop_column_bootstrap(final_model, X, y, oot_split, feature_cols, n_boot=20, max_features=15)

    # Per-område metrics (hvis delmarkedsområde finnes i df)
    if 'delmarkedsområde' in df.columns:
        print("Beregner per-område R² og RMSE...")
        preds_full = final_model.predict(X)
        import pandas as pd
        preds_full_series = pd.Series(preds_full, index=X.index)
        area_rows = []
        for area, grp in df.groupby('delmarkedsområde'):
            X_area = X.loc[grp.index]
            y_area = y.loc[grp.index]
            pred_area = preds_full_series.loc[grp.index]
            if len(y_area.unique()) > 1 and len(y_area) >= 3:
                r2_a = r2_score(y_area, pred_area)
            else:
                r2_a = float('nan')
            rmse_a = mean_squared_error(y_area, pred_area) ** 0.5
            area_rows.append({'delmarkedsområde': area, 'r2': r2_a, 'rmse': rmse_a, 'n': len(y_area)})
        area_df = pd.DataFrame(area_rows)
        area_df.to_csv(ARTIFACT_DIR / 'area_metrics.csv', index=False, encoding='utf-8')
        print("Per-område metrics lagret -> artifacts/area_metrics.csv")

    # Residualrapporter (D)
    print("Lager residualrapporter...")
    residual_reports(final_model, X, y)
    (ARTIFACT_DIR / 'oot_metrics.json').write_text(json.dumps(oot_metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    # Per-område elastisitet (hvis mulig)
    try:
        import pandas as _pd
        elast_path = ARTIFACT_DIR / 'population_elasticity.csv'
        if elast_path.exists():
            el_df_loaded = _pd.read_csv(elast_path)
            area_el = compute_population_elasticity_by_area(el_df_loaded, df)
            if area_el is not None:
                print("Per-område elastisitet lagret -> artifacts/population_elasticity_by_area.csv")
    except Exception as e:
        print(f"Kunne ikke beregne per-område elastisitet: {e}")

    # Elastisitetsrapport
    try:
        scen_el_df_small = None
        scen_path = ARTIFACT_DIR / 'population_elasticity_scenarios.csv'
        if scen_path.exists():
            import pandas as _pd
            scen_el_df_small = _pd.read_csv(scen_path)
        area_el_df = None
        area_path = ARTIFACT_DIR / 'population_elasticity_by_area.csv'
        if area_path.exists():
            import pandas as _pd
            area_el_df = _pd.read_csv(area_path)
        write_elasticity_report(elasticity_summary, area_el_df, scen_el_df_small)
        print("Elastisitetsrapport lagret -> artifacts/elasticity_report.txt")
    except Exception as e:
        print(f"Kunne ikke skrive elastisitetsrapport: {e}")

    print("\nSkriver rapport...")
    write_report(baseline_results, final_name, final_score, final_params, feature_cols, version_id, top_features_list, elasticity_summary)
    print("Rapport lagret -> artifacts/model_report.txt")

    # Organiser artifacts til underkataloger
    organize_artifacts()

    # Vise kort sammendrag av scenarioer
    print("\nScenario sammendrag:")
    show_cols = ['år', 'kvartall', 'scenario', 'anall innbyggere', 'pred']
    if 'pred_p05' in fut.columns:
        show_cols += ['pred_p05', 'pred_p95']
    print(fut[show_cols].head(12).to_string(index=False))
    print("\nFor å bruke egen scenario-fil: legg en CSV i prosjektet og kall load_custom_scenarios() i egen kode.")


if __name__ == '__main__':
    main()
