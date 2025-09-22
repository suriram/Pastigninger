import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import shap

ARTIFACT_DIR = Path('artifacts')

# Helper for å finne filer i underkatalog-struktur etter opprydding
def find_artifact(*names: str) -> Path | None:
    """Søk etter første fil som finnes blant oppgitte relative stier innen artifacts/.
    Eksempel: find_artifact('reports/model_report.txt','model_report.txt')
    Returnerer Path eller None.
    """
    # First try with ARTIFACT_DIR
    for rel in names:
        p = ARTIFACT_DIR / rel
        if p.exists():
            return p
    
    # Fallback to direct paths (for Streamlit Cloud)
    for rel in names:
        direct_path = Path('artifacts') / rel
        if direct_path.exists():
            return direct_path
    
    # Final fallback: brute force search
    base_names = {Path(n).name for n in names}
    artifacts_dir = Path('artifacts')
    if artifacts_dir.exists():
        for p in artifacts_dir.rglob('*'):
            if p.is_file() and p.name in base_names:
                return p
    
    return None

MODEL_PATH = find_artifact('models/modell_pastigninger.joblib', 'modell_pastigninger.joblib')
MANIFEST_PATH = find_artifact('models/model_manifest.json', 'model_manifest.json')
SHAP_ENRICHED_PATH = find_artifact('shap/shap_importance_enriched.csv', 'shap_importance_enriched.csv')
SHAP_PER_AREA_PATH = find_artifact('shap/shap_per_area.csv', 'shap_per_area.csv')
AREA_METRICS_PATH = find_artifact('diagnostics/area_metrics.csv', 'area_metrics.csv')
FEATURE_PATH = find_artifact('metadata/feature_cols.json', 'feature_cols.json')
SCENARIO_PATH = find_artifact('scenarios/scenario_predictions.csv', 'scenario_predictions.csv')
SHAP_IMPORTANCE_PATH = find_artifact('shap/shap_importance.csv', 'shap_importance.csv')
REPORT_PATH = find_artifact('reports/model_report.txt', 'model_report.txt')
VALIDATION_PATH = find_artifact('diagnostics/validation_report.txt', 'validation_report.txt')
PDP_PATH = find_artifact('diagnostics/pdp_values.csv', 'pdp_values.csv')
DROP_COL_PATH = find_artifact('diagnostics/drop_column_importance.csv', 'drop_column_importance.csv')
SHAP_BOOT_PATH = find_artifact('shap/shap_importance_bootstrap.csv', 'shap_importance_bootstrap.csv')
RESIDUALS_PATH = find_artifact('diagnostics/residuals.csv', 'residuals.csv')
RESIDUAL_STATS_PATH = find_artifact('diagnostics/residual_stats.json', 'residual_stats.json')
OOT_PATH = find_artifact('diagnostics/oot_metrics.json', 'oot_metrics.json')
ELASTICITY_GLOBAL_SUMMARY_PATH = find_artifact('elasticity/population_elasticity_summary.json', 'population_elasticity_summary.json')
ELASTICITY_GLOBAL_PATH = find_artifact('elasticity/population_elasticity.csv', 'population_elasticity.csv')
ELASTICITY_AREA_PATH = find_artifact('elasticity/population_elasticity_by_area.csv', 'population_elasticity_by_area.csv')
ELASTICITY_SCENARIO_PATH = find_artifact('elasticity/population_elasticity_scenarios.csv', 'population_elasticity_scenarios.csv')
ELASTICITY_METHOD_PATH = find_artifact('elasticity/elasticity_report.txt', 'elasticity_report.txt')
ELASTICITY_AREA_PATH = find_artifact('elasticity/population_elasticity_by_area.csv', 'population_elasticity_by_area.csv')
ELASTICITY_SCENARIO_PATH = find_artifact('elasticity/population_elasticity_scenarios.csv', 'population_elasticity_scenarios.csv')
ELASTICITY_METHOD_PATH = find_artifact('elasticity/elasticity_report.txt', 'elasticity_report.txt')
DATA_PATH = Path('reg.csv')

st.set_page_config(page_title="Påstigningsprognoser", layout="wide")
st.title("Påstigningsprognoser – modelldashboard")
st.markdown(
    """
**Formål:** Dette dashboardet lar deg utforske en maskinlæringsmodell som estimerer antall påstigninger.

Modellen er trent på historikk og bruker spesielt befolkningsnivå og sesongmønster (kvartal + trigonometriske funksjoner) for å forklare variasjon.

**Hoveddeler:**
1. Manuell prediksjon – test enkeltverdier
2. Scenario-prediksjoner – auto-genererte fremtidsbaner (basis, +2%, -2%)
3. SHAP – bidrag per feature for å forstå modellen

Tips: Kjør `main.py` på nytt etter at du legger til nye historiske data for å oppdatere alt.
"""
)

@st.cache_data
def load_feature_cols():
    """Load feature columns with multiple fallback strategies"""
    # Try via find_artifact first
    if FEATURE_PATH and FEATURE_PATH.exists():
        return json.loads(FEATURE_PATH.read_text(encoding='utf-8'))
    
    # Direct path fallbacks
    fallback_paths = [
        Path('artifacts/metadata/feature_cols.json'),
        Path('./artifacts/metadata/feature_cols.json')
    ]
    
    for path in fallback_paths:
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
    
    return []

def load_csv_with_fallback(primary_path, *fallback_relative_paths):
    """Generic CSV loader with fallback paths"""
    if primary_path and primary_path.exists():
        return pd.read_csv(primary_path)
    
    for rel_path in fallback_relative_paths:
        direct_path = Path('artifacts') / rel_path
        if direct_path.exists():
            return pd.read_csv(direct_path)
    
    return pd.DataFrame()

def load_json_with_fallback(primary_path, *fallback_relative_paths):
    """Generic JSON loader with fallback paths"""
    if primary_path and primary_path.exists():
        return json.loads(primary_path.read_text(encoding='utf-8'))
    
    for rel_path in fallback_relative_paths:
        direct_path = Path('artifacts') / rel_path
        if direct_path.exists():
            return json.loads(direct_path.read_text(encoding='utf-8'))
    
    return None

@st.cache_data
def load_manifest():
    if MANIFEST_PATH and MANIFEST_PATH.exists():
        try:
            return pd.read_json(MANIFEST_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

@st.cache_data
@st.cache_data
def load_scenarios():
    return load_csv_with_fallback(SCENARIO_PATH, 'scenarios/scenario_predictions.csv')

@st.cache_data
def load_shap_importance():
    return load_csv_with_fallback(SHAP_IMPORTANCE_PATH, 'shap/shap_importance.csv')

@st.cache_data
def load_shap_enriched():
    return load_csv_with_fallback(SHAP_ENRICHED_PATH, 'shap/shap_importance_enriched.csv')

@st.cache_data
def load_shap_per_area():
    df = load_csv_with_fallback(SHAP_PER_AREA_PATH, 'shap/shap_per_area.csv')
    if not df.empty and 'delmarkedsområde' in df.columns:
        df['delmarkedsområde'] = df['delmarkedsområde'].astype(str)
        df = df[df['delmarkedsområde'] != '12']
    return df

@st.cache_data
def load_area_metrics():
    return load_csv_with_fallback(AREA_METRICS_PATH, 'diagnostics/area_metrics.csv')

@st.cache_data
def load_validation():
    if VALIDATION_PATH and VALIDATION_PATH.exists():
        return VALIDATION_PATH.read_text(encoding='utf-8')
    return ""

@st.cache_data
def load_pdp():
    if PDP_PATH and PDP_PATH.exists():
        try:
            return pd.read_csv(PDP_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_drop_importance():
    return load_csv_with_fallback(DROP_COL_PATH, 'diagnostics/drop_column_importance.csv')

@st.cache_data
def load_shap_bootstrap():
    return load_csv_with_fallback(SHAP_BOOT_PATH, 'shap/shap_importance_bootstrap.csv')

@st.cache_data
def load_residuals():
    return load_csv_with_fallback(RESIDUALS_PATH, 'diagnostics/residuals.csv')

def load_residual_stats():
    return load_json_with_fallback(RESIDUAL_STATS_PATH, 'diagnostics/residual_stats.json')

def load_oot_metrics():
    return load_json_with_fallback(OOT_PATH, 'diagnostics/oot_metrics.json')
def load_elasticity_artifacts():
    summary = load_json_with_fallback(ELASTICITY_GLOBAL_SUMMARY_PATH, 'elasticity/population_elasticity_summary.json')
    df_global = load_csv_with_fallback(ELASTICITY_GLOBAL_PATH, 'elasticity/population_elasticity.csv')
    df_area = load_csv_with_fallback(ELASTICITY_AREA_PATH, 'elasticity/population_elasticity_by_area.csv')
    df_scen = load_csv_with_fallback(ELASTICITY_SCENARIO_PATH, 'elasticity/population_elasticity_scenarios.csv')
    
    # Load method text
    method_text = ""
    if ELASTICITY_METHOD_PATH and ELASTICITY_METHOD_PATH.exists():
        method_text = ELASTICITY_METHOD_PATH.read_text(encoding='utf-8')
    else:
        method_path = Path('artifacts/elasticity/elasticity_report.txt')
        if method_path.exists():
            method_text = method_path.read_text(encoding='utf-8')
    
    return summary, df_global, df_area, df_scen, method_text

@st.cache_data
def load_history():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    num_cols = ['år','kvartall','påstigninger','anall innbyggere']
    for col in num_cols:
        df[col] = (df[col].astype(str)
                            .str.replace('"','', regex=False)
                            .str.replace(',','', regex=False)
                            .str.strip())
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=num_cols)
    df = df.astype({'år':int,'kvartall':int,'påstigninger':int,'anall innbyggere':int})
    df = df.sort_values(['år','kvartall']).copy()
    df['t_index'] = range(1, len(df)+1)
    return df


def build_input_df(year, quarter, population, feature_cols, hist_df: pd.DataFrame, t_mode: str):
    """Build input DataFrame with robust feature name handling for encoding issues."""
    
    # Start med grunnleggende features 
    row = {
        'anall innbyggere': population,
        't_index': 9999,  # default fallback
        'sin_q': np.sin(2 * np.pi * (quarter / 4.0)),
        'cos_q': np.cos(2 * np.pi * (quarter / 4.0)),
    }
    
    # Håndter år-kolonnen robust (encoding-safe)
    # Finn den faktiske år-kolonnen i feature_cols 
    year_col = None
    for col in feature_cols:
        if 'år' in col or 'ar' in col or 'Ã¥r' in col:
            year_col = col
            break
    
    if year_col:
        row[year_col] = year
    else:
        # Fallback - prøv begge varianter
        row['år'] = year
        row['Ã¥r'] = year
        
    # Kvartall for debugging (ikke i model)
    row['kvartall'] = quarter
    
    if not hist_df.empty:
        # Finn eksisterende t_index hvis historisk punkt
        match = hist_df[(hist_df['år']==year) & (hist_df['kvartall']==quarter)]
        if not match.empty:
            row['t_index'] = int(match['t_index'].iloc[0])
        else:
            # Fortsett sekvens
            row['t_index'] = int(hist_df['t_index'].max() + 1)
        if t_mode == 'Uten t_index (sett til 0)':
            row['t_index'] = 0
    
    # Quarterly dummies
    for q in [1,2,3,4]:
        row[f'Q_{q}'] = 1 if quarter == q else 0
    
    # Ensure all feature_cols exist (default to 0)
    for f in feature_cols:
        if f not in row:
            row[f] = 0
    
    # Return DataFrame with exact column order
    df = pd.DataFrame([row])
    return df[feature_cols]

feature_cols = load_feature_cols()
manifest_df = load_manifest()
version_label = 'Siste (latest)'
model_choice = version_label
if MODEL_PATH is None:
    st.error("Fant ikke modellfil. Kjør main.py for å generere artifacts.")
    st.stop()

if not manifest_df.empty:
    # Sorter etter tid synkende
    manifest_df = manifest_df.sort_values('created_utc', ascending=False)
    version_options = [version_label] + [f"{r.version} | {r.model_type} (R²={r.cv_r2:.3f})" for r in manifest_df.itertuples()]
    with st.sidebar:
        model_choice = st.selectbox("Modellversjon", options=version_options, help="Velg hvilken modellfil som skal lastes")
    if model_choice != version_label:
        chosen_version = model_choice.split(' | ')[0]
        row = manifest_df[manifest_df['version'] == chosen_version].iloc[0]
        model_path = ARTIFACT_DIR / row.file
    else:
        model_path = MODEL_PATH
else:
    model_path = MODEL_PATH

model = load_model(model_path)

# Critical fallback if feature_cols is empty
if not feature_cols and model is not None:
    st.warning("⚠️ Using model feature names as fallback since JSON loading failed")
    feature_cols = list(model.feature_names_in_)

scenarios_df = load_scenarios()
shap_imp = load_shap_importance()
shap_enriched = load_shap_enriched()
shap_per_area_df = load_shap_per_area()
area_metrics_df = load_area_metrics()
hist_df = load_history()
validation_text = load_validation()
pdp_df = load_pdp()
drop_imp_df = load_drop_importance()
shap_boot_df = load_shap_bootstrap()
residuals_df = load_residuals()
resid_stats = load_residual_stats()
oot_metrics = load_oot_metrics()
elasticity_summary, elasticity_global_df, elasticity_area_df, elasticity_scen_df, elasticity_method_text = load_elasticity_artifacts()
# Ekskluder Area 12 fra elastisitets-områdedata hvis tilstede
if not elasticity_area_df.empty and 'delmarkedsområde' in elasticity_area_df.columns:
    elasticity_area_df['delmarkedsområde'] = elasticity_area_df['delmarkedsområde'].astype(str)
    elasticity_area_df = elasticity_area_df[elasticity_area_df['delmarkedsområde'] != '12']

# --- Sidebar modellinfo ---
with st.sidebar:
    st.header("Modellinformasjon")
    if REPORT_PATH and REPORT_PATH.exists():
        report_text = REPORT_PATH.read_text(encoding='utf-8')
        # Ekstrahér noen nøkler (enkelt parsing)
        lines = [l for l in report_text.splitlines() if l.strip()]
        chosen_line = next((l for l in lines if l.startswith('  Gradient') or l.startswith('  Random')), None)
        st.markdown("**Valgt modell**")
        if chosen_line:
            st.code(chosen_line.strip())
        cv_line = next((l for l in lines if '(CV R²=' in l), None)
        if cv_line:
            st.markdown(cv_line)
        st.markdown("**Features**")
        feat_section = False
        feats = []
        for l in lines:
            if l.startswith('FEATURES BRUKT:'):
                feat_section = True
                continue
            if feat_section:
                if l.startswith('  - '):
                    feats.append(l.replace('  - ','').strip())
        if feats:
            st.write(", ".join(feats))
            # Feature forklaringer
            feature_descriptions = {
                'anall innbyggere': 'Antall innbyggere i området – hoveddriver for volum.',
                'år': 'Kalenderår – fanger lineær/langsiktig trend.',
                't_index': 'Løpende sekvens (1..N) sortert etter år+kvartal. Alternativ representasjon av tid.',
                'sin_q': 'Sinus-transformasjon av kvartall (sesongsyklus).',
                'cos_q': 'Cosinus-transformasjon av kvartall – kombinert med sin_q gir glatt sesong.',
                'Q_1': 'Dummy: 1 hvis 1. kvartal, ellers 0 (diskret sesongeffekt).',
                'Q_2': 'Dummy: 1 hvis 2. kvartal, ellers 0.',
                'Q_3': 'Dummy: 1 hvis 3. kvartal, ellers 0.',
                'Q_4': 'Dummy: 1 hvis 4. kvartal, ellers 0.'
            }
            with st.expander("Forklaring av features", expanded=False):
                for f in feats:
                    desc = feature_descriptions.get(f, 'Ingen spesifikk beskrivelse (kan være modell-spesifikk).')
                    st.markdown(f"**{f}**: {desc}")
        st.divider()
        # OOT metrics
        if oot_metrics:
            st.markdown("**Out-of-time (OOT) metrics**")
            st.code(f"OOT R²={oot_metrics.get('oot_r2'):.3f} RMSE={oot_metrics.get('oot_rmse',0):.0f} (n={oot_metrics.get('n_holdout')})")
    else:
        st.info("Ingen rapport funnet – kjør main.py")

    st.markdown("**Begrensninger**")
    st.caption(
        """
        - Ingen eksplisitt variabel for delmarkedsområde (aggregerer mønstre)
        - t_index kan introdusere ekstrapolasjonsrisiko langt frem i tid
        - Befolkningsscenarier er enkle (+/-2%) – legg inn egne hvis nødvendig
        - Ekstreme outliers påvirker tre-modellen (sjekk seneste rader)
        """
    )
    st.markdown("**Anbefalt bruk**")
    st.caption(
        """
        1. Valider mot kjent historikk (bruk predict.py --vis-faktisk)
        2. Vurder per-område modeller hvis heterogenitet
        3. Oppdater data jevnlig og retren
        4. Følg med på avvik mot faktiske tall >5–10%
        """
    )
    st.markdown("**t_index modus**")
    t_mode = st.radio("Velg hvordan t_index skal settes", ["Historisk/fremtid sekvens", "Uten t_index (sett til 0)"], help="Velg om du vil bruke trendvariabelen eller teste modell uten.")
    # Områdefilter (for scenario / SHAP per område)
    area_options = []
    if 'delmarkedsområde' in hist_df.columns:
        area_options = sorted(hist_df['delmarkedsområde'].astype(str).unique())
    selected_area = None
    if area_options:
        selected_area = st.selectbox("Filtrer på delmarkedsområde (valgfritt)", options=["(ingen)"] + area_options)
        if selected_area == "(ingen)":
            selected_area = None
    # Valideringsstatus
    if validation_text:
        st.divider()
        st.markdown("**Datavalidering**")
        issues = [l for l in validation_text.splitlines() if l.startswith('- ')]
        if issues:
            st.warning(f"Fant {len(issues)} potensielle dataproblemer. Se detaljer nedenfor.")
            with st.expander("Se valideringsrapport"):
                st.code(validation_text)
        else:
            st.success("Ingen valideringsproblemer funnet")

if 't_mode' not in locals():
    t_mode = "Historisk/fremtid sekvens"

col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("År", min_value=2020, max_value=2100, value=2026)
with col2:
    quarter = st.selectbox("Kvartall", [1,2,3,4], index=0)
with col3:
    population = st.number_input("Antall innbyggere", min_value=0, value=270000, step=1000)

with st.expander("ℹ️ Hvordan tolke prediksjonen?", expanded=False):
    st.markdown(
        """
        Når du trykker *Prediker* bygger verktøyet en enkel feature-vektor av input verdiene dine og mater den inn i den lagrede modellen.

        Merk:
        - For produksjon kan du bruke reell sekvens / fortløpende indeks
        - Usikkerhet (intervaller) vises ikke her, men kan beregnes i `main.py` via bootstrap
        """
    )

if st.button("Prediker", help="Kjør modell på input over"):
    try:
        X_row = build_input_df(year, quarter, population, feature_cols, hist_df, t_mode)
        
        # Check feature name matching
        missing_features = set(model.feature_names_in_) - set(X_row.columns)
        
        if not missing_features:  # Only predict if no missing features
            pred = model.predict(X_row)[0]
            st.success(f"Estimert påstigninger: {pred:,.0f}")
            st.caption(f"(Tall avrundet) – t_index brukt: {int(X_row['t_index'].iloc[0])}")
        else:
            st.error("Kan ikke predikere på grunn av manglende features.")
            
    except Exception as e:
        st.error(f"Prediksjons-feil: {e}")
        st.exception(e)

st.markdown("---")
st.subheader("Scenario-prediksjoner")
st.markdown(
    """
Disse scenarioene er generert automatisk basert på en lineær trend i historiske befolkningstall og tre variasjoner:

- **basis**: Forventet trend
- **pluss2pct**: +2% justering i befolkning per kvartal
- **minus2pct**: -2% justering

Bootstrap-kolonner (p05 / p95) viser et enkelt usikkerhetsbånd.
"""
)
if scenarios_df.empty:
    st.info("Ingen scenario-fil funnet. Kjør main.py først.")
else:
    if selected_area and 'delmarkedsområde' in scenarios_df.columns:
        scen_show = scenarios_df[scenarios_df['delmarkedsområde'].astype(str) == selected_area]
        if scen_show.empty:
            st.info(f"Ingen scenarier for område {selected_area} (kan være at genereringen ikke inkluderte det). Viser alle.")
            scen_show = scenarios_df
    else:
        scen_show = scenarios_df
    st.dataframe(scen_show.head(40))

st.markdown("---")
st.subheader("Feature importance (SHAP)")
with st.expander("Hva er SHAP?", expanded=False):
    st.markdown(
        """
        **SHAP (SHapley Additive exPlanations)** gir et konsistent rammeverk for å fordele modellens prediksjon på input-features.

        Tolkning:
        - Høy *mean_abs_shap* => stort gjennomsnittlig bidrag (positivt eller negativt)
        - Features som dominerer her driver modellen mest
        - Sesong (Q_*) og trigonometriske (sin_q / cos_q) fanger sesongmønster
        - `t_index` kan fange trend over tid
        - `anall innbyggere` forklarer ofte mest

        Summary-plottet (hvis aktivert) viser fordeling av SHAP-verdier per rad og feature.
        """
    )
if shap_imp.empty:
    st.info("Ingen SHAP importance funnet.")
else:
    if not shap_enriched.empty:
        st.markdown("**Global SHAP (med % andel)**")
        show_cols = shap_enriched.copy()
        show_cols['pct'] = show_cols['pct'].map(lambda x: f"{x:.2f}%")
        st.dataframe(show_cols)
        if not shap_boot_df.empty:
            st.markdown("**SHAP bootstrap intervaller (absolutt verdier)**")
            st.dataframe(shap_boot_df[['feature','shap_mean','p05','p95','shap_std']].head(25))
    else:
        st.dataframe(shap_imp)

    if selected_area and not shap_per_area_df.empty:
        sub_area = shap_per_area_df[shap_per_area_df['area'] == selected_area]
        if not sub_area.empty:
            st.markdown(f"**SHAP per feature for område {selected_area}**")
            st.dataframe(sub_area.sort_values('mean_abs_shap', ascending=False).head(25))
        else:
            st.caption("Ingen per-område SHAP funnet for valgt område.")

    # Optional global SHAP summary (compute on demand)
    if st.checkbox("Vis SHAP summary-plot (kan ta litt tid)", help="Genererer et globalt oversiktsplot"):
        try:
            explainer = shap.TreeExplainer(model)
            # Bruk et lite utvalg fra scenario_df eller generer en syntetisk batch
            if not scenarios_df.empty:
                X_sample = scenarios_df[feature_cols].head(50)
            else:
                X_sample = build_input_df(year, quarter, population, feature_cols, hist_df, t_mode)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values, X_sample, show=False)
            import matplotlib.pyplot as plt
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Feil ved generering av SHAP plot: {e}")

st.markdown("---")
st.markdown("---")
st.subheader("Per-område metrikker")
if area_metrics_df.empty:
    st.caption("Ingen area_metrics.csv funnet – kjør main.py på nytt etter at delområder er inkludert.")
else:
    if selected_area:
        st.dataframe(area_metrics_df[area_metrics_df['delmarkedsområde'].astype(str) == selected_area])
    else:
        st.dataframe(area_metrics_df.sort_values('r2'))
    # Advarsler for lav R²
    low_r2 = area_metrics_df[area_metrics_df['r2'] < 0.2]
    if not low_r2.empty:
        st.warning(f"{len(low_r2)} område(r) har lav forklaringsgrad (R² < 0.2). Vurder segmentering eller ekstra features.")

st.markdown("---")
st.subheader("Drop-column importance (OOT delta R²)")
if drop_imp_df.empty:
    st.caption("Ingen drop_column_importance.csv – kjør main.py med nyeste kode.")
else:
    st.dataframe(drop_imp_df.head(30))

st.markdown("---")
st.subheader("Residualanalyse")
if not resid_stats:
    st.caption("Ingen residual_stats.json funnet.")
else:
    st.markdown(f"**RMSE:** {resid_stats.get('rmse',0):,.0f}  |  **MAPE:** {resid_stats.get('mape_pct',0):.2f}%  |  **Bias:** {resid_stats.get('bias',0):,.0f}")
if residuals_df.empty:
    st.caption("Ingen residuals.csv funnet.")
else:
    with st.expander("Se residualtabell (første 1000)"):
        st.dataframe(residuals_df.head(1000))
    # Enkle plot hvis plotly
    try:
        import plotly.express as px
        if 'pred' in residuals_df.columns:
            fig_r = px.scatter(residuals_df.head(3000), x='pred', y='residual', title='Residual vs prediksjon', opacity=0.6)
            st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        pass

st.markdown("---")
st.subheader("Partial Dependence (PDP)")
with st.expander("Hva er PDP?", expanded=False):
    st.markdown(
        """
        **Partial Dependence** viser hvordan den forventede prediksjonen endres når én feature varierer, mens andre holdes på et representativt nivå.

        Tolkning:
        - En bratt kurve indikerer at modellen er sensitiv for endringer i feature-nivået.
        - Relativt flat linje betyr lav marginal effekt.
        - PDP er et globalt gjennomsnitt og skjuler interaksjoner.
        """
    )
if pdp_df.empty:
    st.caption("Ingen PDP-data funnet. Kjør main.py for å generere pdp_values.csv.")
else:
    feat_options = sorted([f for f in pdp_df['feature'].dropna().unique()])
    sel_pdp_feat = st.selectbox("Velg feature for PDP", options=feat_options)
    sub_pdp = pdp_df[pdp_df['feature'] == sel_pdp_feat].dropna(subset=['value','pdp'])
    if sub_pdp.empty:
        st.info("Ingen gyldige PDP-punkter for valgt feature.")
    else:
        try:
            import plotly.express as px
            fig = px.line(sub_pdp.sort_values('value'), x='value', y='pdp', markers=True, title=f'PDP: {sel_pdp_feat}')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Plotly ikke tilgjengelig – viser tabell i stedet. (Installer plotly for graf: pip install plotly)")
        st.dataframe(sub_pdp.sort_values('value'))

st.markdown("---")
st.subheader("Befolkningselastisitet")
with st.expander("Metode og definisjon", expanded=False):
    if elasticity_method_text:
        st.code(elasticity_method_text)
    else:
        st.markdown("Elastisitet beregnes som (dY/dPop)*(Pop/Y) estimert med sentral differanse ±1% av nivået.")

col_e1, col_e2 = st.columns(2)
with col_e1:
    if elasticity_summary:
        st.markdown("**Globalt sammendrag**")
        st.json(elasticity_summary)
    else:
        st.caption("Ingen global elastisitet – kjør main.py")
with col_e2:
    if not elasticity_global_df.empty:
        try:
            import plotly.express as px
            sample_glob = elasticity_global_df.dropna(subset=['elasticity']).head(3000)
            if not sample_glob.empty:
                fig_el = px.histogram(sample_glob, x='elasticity', nbins=40, title='Fordeling global elastisitet')
                st.plotly_chart(fig_el, use_container_width=True)
        except Exception:
            pass
    else:
        st.caption("Ingen global elastisitetstabell.")

if not elasticity_area_df.empty:
    st.markdown("**Per-område elastisitet**")
    if selected_area:
        st.dataframe(elasticity_area_df[elasticity_area_df['delmarkedsområde'].astype(str) == str(selected_area)])
    else:
        st.dataframe(elasticity_area_df.sort_values('mean', ascending=False))
else:
    st.caption("Ingen population_elasticity_by_area.csv funnet.")

if not elasticity_scen_df.empty:
    st.markdown("**Scenario-elastisitet**")
    # Rens og grupper sikkert (unngå .empty på skalare verdier)
    scen_df_clean = (
        elasticity_scen_df[['scenario', 'elasticity']]
        .replace([np.inf, -np.inf], np.nan)
    )
    scen_groups = scen_df_clean.groupby('scenario', dropna=False)['elasticity']
    rows = []
    for scen, vals in scen_groups:
        vals_nonan = vals.dropna()
        if vals_nonan.shape[0] == 0:
            rows.append({'scenario': scen, 'mean': None, 'median': None, 'n': 0})
        else:
            rows.append({
                'scenario': scen,
                'mean': float(vals_nonan.mean()),
                'median': float(vals_nonan.median()),
                'n': int(vals_nonan.shape[0])
            })
    scen_table = pd.DataFrame(rows).sort_values('mean', ascending=False, na_position='last')
    st.dataframe(scen_table)
    # Plot boksdiagram dersom det finnes minst ett ikke-NaN datapunkt
    if scen_df_clean['elasticity'].notna().any():
        try:
            import plotly.express as px
            fig_scen = px.box(
                scen_df_clean.dropna(subset=['elasticity']),
                x='scenario', y='elasticity', title='Scenario elastisitet (fordeling)'
            )
            st.plotly_chart(fig_scen, use_container_width=True)
        except Exception:
            st.caption("Plotly plot feilet – viser kun tabell.")
else:
    st.caption("Ingen scenario-elastisitet funnet.")

st.caption("Bygget med Streamlit – oppdater modell og scenarier ved å kjøre main.py på nytt. For dypere analyser, se artefakter i artifacts/ mappen.")
