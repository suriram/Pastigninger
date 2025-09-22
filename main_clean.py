"""
Dette skriptet bruker regresjonsmodeller for å estimere antall påstigninger basert på kvartall og antall innbyggere.

Venstreside variabel (avhengig):
	påstigninger

Høyreside variabler (uavhengige):
	kvartall, antall innbyggere

Modeller som testes:
- Lineær regresjon
- Random Forest 
- Gradient Boosting

Resultatene vises med R² og RMSE for hver modell.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Les inn data med riktige kolonnenavn og fjerning av mellomrom/tegn
data = pd.read_csv('reg.csv')
data.columns = [col.strip() for col in data.columns]

# Rens tallkolonner for mellomrom og komma
for col in ['år', 'kvartall', 'påstigninger', 'anall innbyggere']:
    data[col] = data[col].astype(str).str.replace('"', '').str.replace(',', '').str.strip()
    # Konverter til int, sett ugyldige til NaN, og dropp rader med NaN
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=['år', 'kvartall', 'påstigninger', 'anall innbyggere'])
data = data.astype({'år': int, 'kvartall': int, 'påstigninger': int, 'anall innbyggere': int})

# Uavhengige variabler (uten 'år')
X = data[['kvartall', 'anall innbyggere']]
y = data['påstigninger']

# Del opp i trenings- og testdata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regresjonsmodeller
models = {
    'Lineær regresjon': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

print("\n--- Resultater for regresjonsmodeller ---")
for navn, modell in models.items():
    modell.fit(X_train, y_train)
    y_pred = modell.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"{navn}: R² = {r2:.3f}, RMSE = {rmse:.0f}")

print("\nR² (forklaringsgrad) viser hvor mye av variasjonen i påstigninger modellen forklarer. RMSE (root mean squared error) viser gjennomsnittlig avvik mellom predikert og faktisk verdi.")

# Estimerte funksjoner og parametre
print("\n--- Estimerte funksjoner og parametre ---")
for navn, modell in models.items():
    if navn == 'Lineær regresjon':
        print("Lineær regresjon:")
        print(f"  y = {modell.intercept_:.2f} + {modell.coef_[0]:.2f} * kvartall + {modell.coef_[1]:.2f} * anall innbyggere")
    else:
        print(f"{navn} feature importance:")
        for feat, imp in zip(X.columns, modell.feature_importances_):
            print(f"  {feat}: {imp:.3f}")