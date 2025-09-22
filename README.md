# 🚍 Påstigningsprognoser - ML Dashboard

En maskinlæringsbasert modell for å forutsi påstigninger i kollektivtransport basert på befolkning, tid og sesongvariasjoner.

## 📊 Funksjoner

- **Interaktiv Streamlit dashboard** med manuell prediksjon og scenariovisninger
- **Avansert ML-pipeline** (Linear Regression, Random Forest, Gradient Boosting)
- **SHAP-basert tolkbarhet** med feature importance og per-område analyser
- **Befolkningselastisitet** - punktvis beregning av marginaleffekter
- **Robust diagnostikk** (PDP, out-of-time validering, bootstrap intervaller)
- **Excel-eksport** for leveranse til stakeholdere

## 🚀 Kom i gang

### Lokalt oppsett

```bash
# Klon repository
git clone <repo-url>
cd mariaReg

# Opprett virtuelt miljø
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installer avhengigheter
pip install -r requirements_minimal.txt

# Kjør Streamlit appen
streamlit run streamlit_app.py
```

### Streamlit Cloud deployment

1. **Push til GitHub** og gjør repository public (eller privat med Streamlit Cloud Pro)
2. **Gå til [share.streamlit.io](https://share.streamlit.io)**
3. **Deploy**: Velg repository, branch (main) og main file (`streamlit_app.py`)
4. **Advanced settings**: 
   - Python version: 3.9+
   - Requirements file: `requirements_minimal.txt`

Appen starter automatisk på første besøk og redeployer ved push til main branch.

## 📁 Prosjektstruktur

```
mariaReg/
├── streamlit_app.py          # Hoveddashboard
├── main.py                   # ML pipeline og trening
├── rf_elasticity.py          # Random Forest elastisitet
├── export_excel_package.py   # Excel-leveranse generator
├── reg.csv                   # Treningsdata (ikke inkludert i repo)
├── requirements_minimal.txt  # Streamlit Cloud avhengigheter
├── requirements.txt          # Komplett freeze (lokal utvikling)
└── artifacts/                # Genererte filer (ikke i repo)
 - `artifacts/oot_metrics.json` (out-of-time holdout metrics)
 - `streamlit_app.py` – interaktivt dashboard
 - `artifacts/` – lagrer modell, rapport, scenarioprediksjoner, shap-data
    ├── models/               # Lagrede modeller (.joblib)
    ├── reports/              # Tekst-rapporter og sammendrag
    ├── scenarios/            # Framtidsprediksjoner
    ├── shap/                 # Tolkbarhetsanalyser
    ├── diagnostics/          # Validering og metrikker
    ├── elasticity/           # Befolkningselastisitet
    └── distribution/         # Modellpakker for deling
```

## 🔄 Arbeidsflyt

### 1. Trene modell
```bash
python main.py
```
Genererer artifacts (modell, rapporter, SHAP, elastisitet, scenarier).

### 2. Utforske resultater
```bash
streamlit run streamlit_app.py
```
Interaktiv dashboard med prediksjoner, feature importance, og elastisitet.

### 3. Eksportere leveranse
```bash
python export_excel_package.py
```
Lager `modell_leveranse_<timestamp>.xlsx` med alle hovedresultater.

## 🧠 Modelldetaljer

- **Algoritmer**: Linear Regression, Random Forest, Gradient Boosting
- **Features**: Befolkning, år, kvartalsesong (sin/cos + dummies), tidsindeks, område-dummies
- **Validering**: Cross-validation + out-of-time holdout
- **Tolkbarhet**: SHAP global/lokal, Partial Dependence Plots
- **Elastisitet**: Punktvis (dY/dPop)*(Pop/Y) via sentral differanse ±1%

## 📈 Tolkningseksempler

**Elastisitet = 0.8**: 1% økning i befolkning → ~0.8% økning i påstigninger  
**Negativ elastisitet**: Modellen fanger opp ikke-lineære effekter eller lokale mønstre  
**Høy spredning mellom områder**: Indikerer strukturelle forskjeller i responsivitet

## ⚠️ Begrensninger

- **Ikke kausalt**: Modellen beskriver mønstre, ikke årsak-virkning
- **Punktvis elastisitet**: Gjelder marginale endringer (~1%), ikke store sjokk
- **Dataavhengig**: Kvalitet avhenger av representativitet i treningsdata
- **Temporal scope**: Ekstrapolering utenfor treningsperiode kan være upresis

## 🛠️ Avanserte funksjoner

### Custom scenarios
Legg til egen CSV med kolonner: `år`, `kvartall`, `anall innbyggere`, (evt. `delmarkedsområde`)

### Random Forest elastisitet
```bash
python rf_elasticity.py
```
Sammenlign elastisiteter på tvers av algoritmer.

### Modellpakker
RF-script genererer zip-pakker i `artifacts/distribution/` for deling med andre team.

## 📞 Support

For spørsmål om modellspesifikasjoner, se `artifacts/reports/model_report.txt` og `artifacts/elasticity/elasticity_report.txt` etter kjøring av `main.py`.

---

*Bygget med Streamlit, scikit-learn, SHAP og kjærlighet til maskinlæring* ❤️

## Drop-column bootstrap importance
Filen `drop_column_importance.csv` viser gjennomsnittlig og percentil (5/95) reduksjon i OOT R² når hver feature fjernes (topp ~15). Dette fungerer som en “prediktiv signifikans”-indikator.

## SHAP bootstrap intervaller
`shap_importance_bootstrap.csv` gir mean, std, p05, p95 for gj.snitt absolutt SHAP via bootstrap-resampling av rader + re-trening. Bruk det til å vurdere robusthet for toppfeatures.

## Residualanalyse
- `residuals.csv` radvise residualer (faktisk - pred)
- `residual_stats.json` oppsummerer RMSE, MAPE (%) og bias.
Bruk dette for å identifisere systematiske skjevheter eller økende feil over tid.

## Eksport av leveranse (Excel)
For å lage en samlet fil til interessenter uten Python:
```bash
python export_excel_package.py
```
Dette genererer `modell_leveranse.xlsx` med ark:
- Oversikt (nøkkelmetadata)
- Rapport_raw (model_report.txt linjer)
- Scenarier (scenario_predictions.csv)
- Per_område_metrics (area_metrics.csv)
- SHAP (enriched eller fallback basic)
- PDP (pdp_values.csv)
- Datavalidering (validation_report.txt)
- Manifest_raw (hele model_manifest.json)

Filen kan deles direkte via e‑post/Teams.

## Avhengigheter
Se `requirements.txt`.

## Videre forbedringer

- Lese inn ny historikk og inkrementell retrening
- Valg av modellversjon direkte i Streamlit (drop-down)
- Automatisk arkivering av gamle scenarioprediksjoner
- Mer avansert hierarkisk modellering (global + område-spesifikke residualer)
- Evaluering av modell per delmarkedsområde (separate metrics)
- Ekstra interaksjons-PDP (to-features) / ICE curves
- Automatisk epost/Slack-varsling ved datavalideringsavvik
- Modellstabilitetsrapport (drift over versjoner)
- Lagre konfidensintervaller i scenariofil per område
 - Rolling window OOT (f.eks. sliding origin) for mer robust tidsvalidasjon
 - Feature stability tracking (endring i drop-column delta over versjoner)
 - Automatisk generering av PDF-rapport
# Pastigninger
