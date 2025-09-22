"""Generer en samlet Excel-pakke (modell_leveranse.xlsx) fra artifacts/.

Innhold / ark:
- Oversikt: nøkkelmetadata + siste versjon
- Rapport_raw: linjer fra model_report.txt
- Scenarier: scenario_predictions.csv
- Per_område_metrics: area_metrics.csv
- SHAP: shap_importance_enriched.csv (fallback: shap_importance.csv)
- PDP: pdp_values.csv
- Datavalidering: validation_report.txt
- Manifest: raw manifest (valgfritt)

Kjør:
    python export_excel_package.py

Filen blir lagt i prosjektroten: modell_leveranse.xlsx
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from typing import Iterable

ART = Path('artifacts')
OUT_FILE = Path('modell_leveranse.xlsx')


def find_file(*candidates: str) -> Path | None:
    """Returner første eksisterende fil gitt relative stier eller bare filnavn.
    Siste fallback: rekursivt søk etter filnavn i artifacts.
    """
    for rel in candidates:
        p = ART / rel
        if p.exists():
            return p
    names = {Path(c).name for c in candidates}
    for p in ART.rglob('*'):
        if p.is_file() and p.name in names:
            return p
    return None

def _safe_read_csv_path(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_overview(manifest: list[dict]) -> pd.DataFrame:
    if not manifest:
        return pd.DataFrame([{"Nøkkel":"Info","Verdi":"Ingen manifest funnet"}])
    latest = manifest[-1]
    rows = [
        {"Nøkkel": "Modellversjon", "Verdi": latest.get('version')},
        {"Nøkkel": "Modelltype", "Verdi": latest.get('model_type')},
        {"Nøkkel": "CV R²", "Verdi": latest.get('cv_r2')},
        {"Nøkkel": "Antall treningsrader", "Verdi": latest.get('n_samples')},
        {"Nøkkel": "Antall features", "Verdi": len(latest.get('features', []))},
    ]
    val_issues = latest.get('validation_issues')
    if val_issues:
        rows.append({"Nøkkel": "Valideringsfunn", "Verdi": ", ".join(val_issues.keys())})
    return pd.DataFrame(rows)


def main():  # noqa: C901 (enkelt skript)
    print("Genererer Excel-pakke...")
    manifest_path = find_file('models/model_manifest.json','model_manifest.json')
    manifest: list[dict] = []
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"Kunne ikke lese manifest: {e}")

    report_path = find_file('reports/model_report.txt','model_report.txt')
    report_lines = []
    if report_path and report_path.exists():
        report_lines = [l for l in report_path.read_text(encoding='utf-8').splitlines()]

    scen = _safe_read_csv_path(find_file('scenarios/scenario_predictions.csv','scenario_predictions.csv'))
    area = _safe_read_csv_path(find_file('diagnostics/area_metrics.csv','area_metrics.csv'))
    shap_enriched = _safe_read_csv_path(find_file('shap/shap_importance_enriched.csv','shap_importance_enriched.csv'))
    shap_basic = _safe_read_csv_path(find_file('shap/shap_importance.csv','shap_importance.csv'))
    shap_df = shap_enriched if not shap_enriched.empty else shap_basic
    shap_boot = _safe_read_csv_path(find_file('shap/shap_importance_bootstrap.csv','shap_importance_bootstrap.csv'))
    drop_imp = _safe_read_csv_path(find_file('diagnostics/drop_column_importance.csv','drop_column_importance.csv'))
    residuals = _safe_read_csv_path(find_file('diagnostics/residuals.csv','residuals.csv'))
    oot_metrics = {}
    oot_path = find_file('diagnostics/oot_metrics.json','oot_metrics.json')
    lin_coef = _safe_read_csv_path(find_file('metadata/linear_regression_coeffs.csv','linear_regression_coeffs.csv'))
    lin_eq_path = find_file('reports/linear_regression_equation.txt','linear_regression_equation.txt')
    lin_equation_lines = []
    if lin_eq_path and lin_eq_path.exists():
        try:
            lin_equation_lines = lin_eq_path.read_text(encoding='utf-8').splitlines()
        except Exception:
            lin_equation_lines = []
    if oot_path and oot_path.exists():
        try:
            oot_metrics = json.loads(oot_path.read_text(encoding='utf-8'))
        except Exception:
            oot_metrics = {}
    pdp = _safe_read_csv_path(find_file('diagnostics/pdp_values.csv','pdp_values.csv'))
    validation_path = find_file('diagnostics/validation_report.txt','validation_report.txt')
    val_lines = []
    if validation_path and validation_path.exists():
        val_lines = validation_path.read_text(encoding='utf-8').splitlines()

    # Elasticity artifacts
    elast_global = _safe_read_csv_path(find_file('elasticity/population_elasticity.csv','population_elasticity.csv'))
    elast_area = _safe_read_csv_path(find_file('elasticity/population_elasticity_by_area.csv','population_elasticity_by_area.csv'))
    elast_scen = _safe_read_csv_path(find_file('elasticity/population_elasticity_scenarios.csv','population_elasticity_scenarios.csv'))
    elast_summary_path = find_file('elasticity/population_elasticity_summary.json','population_elasticity_summary.json')
    elast_report_path = find_file('elasticity/elasticity_report.txt','elasticity_report.txt')
    elast_summary = {}
    if elast_summary_path and elast_summary_path.exists():
        try:
            elast_summary = json.loads(elast_summary_path.read_text(encoding='utf-8'))
        except Exception:
            elast_summary = {}
    elast_report_lines = []
    if elast_report_path and elast_report_path.exists():
        elast_report_lines = elast_report_path.read_text(encoding='utf-8').splitlines()

    overview_df = build_overview(manifest)

    # Filtrering: Fjern Area 12 (aggregat) fra alle relevante datarammer
    def _drop_area12(df: pd.DataFrame, col: str = 'delmarkedsområde') -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if col in df.columns:
            df = df.copy()
            df[col] = df[col].astype(str)
            df = df[df[col] != '12']
        return df

    scen = _drop_area12(scen)
    area = _drop_area12(area)
    elast_area = _drop_area12(elast_area)
    # scenario-elastisitet kan mangle kolonnen, så håndteres betinget
    elast_scen = _drop_area12(elast_scen)

    target = OUT_FILE
    # Hvis filen er låst (PermissionError), forsøk med timestamp i navn
    try:
        writer_context = pd.ExcelWriter(target, engine='xlsxwriter')
    except PermissionError:
        ts_name = f"modell_leveranse_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.xlsx"
        target = Path(ts_name)
        writer_context = pd.ExcelWriter(target, engine='xlsxwriter')

    with writer_context as xw:
        overview_df.to_excel(xw, sheet_name='Oversikt', index=False)
        if report_lines:
            pd.DataFrame({'Rapport': report_lines}).to_excel(xw, sheet_name='Rapport_raw', index=False)
        if not scen.empty:
            scen.to_excel(xw, sheet_name='Scenarier', index=False)
        if not area.empty:
            area.to_excel(xw, sheet_name='Per_område_metrics', index=False)
        if not shap_df.empty:
            shap_df.to_excel(xw, sheet_name='SHAP', index=False)
        if not shap_boot.empty:
            shap_boot.to_excel(xw, sheet_name='SHAP_bootstrap', index=False)
        if not drop_imp.empty:
            drop_imp.to_excel(xw, sheet_name='DropCol_import', index=False)
        if not residuals.empty:
            residuals.head(5000).to_excel(xw, sheet_name='Residuals', index=False)
        if oot_metrics:
            pd.DataFrame([oot_metrics]).to_excel(xw, sheet_name='OOT_metrics', index=False)
        if not lin_coef.empty:
            lin_coef.to_excel(xw, sheet_name='LinReg_coeffs', index=False)
        if lin_equation_lines:
            pd.DataFrame({'Equation': lin_equation_lines}).to_excel(xw, sheet_name='LinReg_equation', index=False)
        if not pdp.empty:
            pdp.to_excel(xw, sheet_name='PDP', index=False)
        if val_lines:
            pd.DataFrame({'Validering': val_lines}).to_excel(xw, sheet_name='Datavalidering', index=False)
        if manifest:
            pd.DataFrame(manifest).to_excel(xw, sheet_name='Manifest_raw', index=False)
        # Elasticity sheets
        if elast_summary:
            pd.DataFrame([elast_summary]).to_excel(xw, sheet_name='Elasticity_summary', index=False)
        if not elast_global.empty:
            elast_global.to_excel(xw, sheet_name='Elasticity_rows', index=False)
        if not elast_area.empty:
            elast_area.to_excel(xw, sheet_name='Elasticity_area', index=False)
        if not elast_scen.empty:
            elast_scen.to_excel(xw, sheet_name='Elasticity_scenarios', index=False)
        if elast_report_lines:
            pd.DataFrame({'Elasticity_report': elast_report_lines}).to_excel(xw, sheet_name='Elasticity_report', index=False)

    print(f"Skrev {target}")


if __name__ == '__main__':
    main()
