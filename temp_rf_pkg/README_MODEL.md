# Random Forest påstigningsmodell

Denne pakken inneholder en trent RandomForestRegressor for å estimere påstigninger basert på befolkning og tids-/sesongfeatures.

## Innhold
- rf_model.joblib (siste versjon)
- rf_feature_cols.json (feature-liste og metadata)
- rf_inference.py (enkelt prediksjonsskript)
- requirements_min.txt (minimal avhengigheter)
- rf_model_manifest.json (historikk)
- *rf_model_...*.joblib (versjonert modell)
- *.sha256 (integritetshash)

## Bruk
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements_min.txt
python rf_inference.py
```

## Elastisitet (RF punktvis)
Mean: 0.5786529950503113  Median: 0.5054411759573714  (n=132)
Tolkning: 1% økning i befolkning ~ 0.5786529950503113% endring i påstigninger (gj.snitt av lokale punkter).

## Forutsetninger
- Ikke-kausal modell; beskriver mønstre i historikk.
- Punktvis elastisitet kan variere betydelig mellom perioder.

## Reproduserbarhet
- sklearn 1.7.2
- hash modell: ebaf2371696797f52e2b5726078d310ad900690d3bff39b6b7d5c79da1523f58

