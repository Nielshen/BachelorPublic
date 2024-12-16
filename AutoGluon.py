import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def prepare_trading_data(df, traded_assets, lookback_period=30):
    """Bereitet die Daten für das Trading vor"""
    
    # Berechne Returns für die gehandelten Assets
    for asset in traded_assets:
        df[f'{asset}_returns'] = df[asset].pct_change()
        
        # Füge Lag-Features hinzu
        for lag in range(1, lookback_period + 1):
            df[f'{asset}_return_lag_{lag}'] = df[f'{asset}_returns'].shift(lag)
    
    # Erstelle Labels (nächster Tag Return)
    for asset in traded_assets:
        df[f'{asset}_next_return'] = df[f'{asset}_returns'].shift(-1)
    
    # Entferne Zeilen mit NaN-Werten
    df = df.dropna()
    
    return df

def calculate_portfolio_value(predictions, test_data, traded_assets, initial_capital=10000):
    """Berechnet den Portfoliowert mit variablen Positionsgrößen"""
    
    portfolio_value = initial_capital
    portfolio_values = [portfolio_value]
    cash = initial_capital  # Verfügbares Bargeld
    
    # Tracking für jedes Asset
    positions = {
        asset: {
            'amount': 0,        # Anzahl der gehaltenen Einheiten
            'value': 0,         # Aktueller Wert der Position
            'entry_price': 0    # Durchschnittlicher Einstandspreis
        } for asset in traded_assets
    }
    
    # Trading Parameter
    MAX_POSITION_SIZE = 0.3    # Maximale Position pro Asset (30% des Portfolios)
    MIN_POSITION_SIZE = 0.1    # Minimale Position pro Asset (10% des Portfolios)
    
    for i in range(len(test_data)):
        for asset in traded_assets:
            prediction = predictions[asset][i]
            current_price = test_data[asset].iloc[i]
            
            # Berechne verfügbares Kapital für Trading
            total_portfolio = cash + sum(pos['value'] for pos in positions.values())
            available_cash = cash
            
            # Position Size basierend auf Vorhersage-Stärke
            confidence = abs(prediction)  # Wie sicher ist die Vorhersage?
            position_size = min(
                MAX_POSITION_SIZE * confidence * total_portfolio,
                available_cash
            )
            position_size = max(position_size, MIN_POSITION_SIZE * total_portfolio)
            
            # Trading Logik
            if prediction > 0.001:  # KAUFEN
                if positions[asset]['amount'] == 0:  # Neue Position
                    units_to_buy = position_size / current_price
                    cost = units_to_buy * current_price
                    
                    if cost <= available_cash:
                        positions[asset]['amount'] = units_to_buy
                        positions[asset]['entry_price'] = current_price
                        positions[asset]['value'] = cost
                        cash -= cost
                        print(f"KAUF: {asset}: {units_to_buy:.2f} Units @ {current_price:.2f}€ "
                              f"(Gesamt: {cost:.2f}€)")
                
            elif prediction < -0.001:  # VERKAUFEN
                if positions[asset]['amount'] > 0:  # Bestehende Position
                    units_to_sell = positions[asset]['amount']
                    revenue = units_to_sell * current_price
                    
                    # Berechne Gewinn/Verlust
                    profit = revenue - (units_to_sell * positions[asset]['entry_price'])
                    
                    positions[asset]['amount'] = 0
                    positions[asset]['value'] = 0
                    positions[asset]['entry_price'] = 0
                    cash += revenue
                    
                    print(f"VERKAUF: {asset}: {units_to_sell:.2f} Units @ {current_price:.2f}€ "
                          f"(Gewinn/Verlust: {profit:.2f}€)")
            
            # Update Position Wert
            if positions[asset]['amount'] > 0:
                positions[asset]['value'] = positions[asset]['amount'] * current_price
        
        # Berechne Gesamtwert am Ende des Tages
        portfolio_value = cash + sum(pos['value'] for pos in positions.values())
        portfolio_values.append(portfolio_value)
        
        # Periodischer Status-Report
        if i % 100 == 0:
            print(f"\n=== Tag {i} Status ===")
            print(f"Portfolio Wert: {portfolio_value:.2f}€")
            print(f"Verfügbares Cash: {cash:.2f}€")
            for asset, pos in positions.items():
                if pos['amount'] > 0:
                    print(f"{asset}: {pos['amount']:.2f} Units, "
                          f"Wert: {pos['value']:.2f}€")
    
    return portfolio_values, positions, cash

# Daten laden
df = pd.read_pickle('multi_data_indicators.pkl')

# Zeitraum filtern
start_date = '2017-09-01'
end_date = df.index[-1]
df = df[start_date:end_date]

btc_columns = [col for col in df.columns if 'Crypto_BTCUSDT' in col]
df = df[btc_columns]

# Definiere die zu handelnden Assets
traded_assets = ['Crypto_BTCUSDT'] #['Crypto_BTCUSDT', 'Stock_AAPL', 'Commodity_Crude_Oil_WTI']

# Daten vorbereiten
df = prepare_trading_data(df, traded_assets)

# Train-Test-Split (80-20)
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

# Separate Modelle für jedes Asset trainieren
models = {}
feature_importance = {}
model_performance = {}

for asset in traded_assets:
    print(f"\nTraining model for {asset}...")
    
    # Zielvariable
    target = f'{asset}_next_return'
    
    # Features (alle Spalten außer die *_next_return Spalten)
    features = [col for col in df.columns if not col.endswith('_next_return')]
    
    # Erstelle Trainingsdatensatz und behandle Inf-Werte
    train_data_clean = train_data[features + [target]].copy()
    
    # Ersetze Inf-Werte durch NaN
    train_data_clean = train_data_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fülle NaN-Werte mit Methode 'ffill' (forward fill)
    train_data_clean = train_data_clean.fillna(method='ffill')
    
    # Fülle verbleibende NaN-Werte mit 0
    train_data_clean = train_data_clean.fillna(0)
    
    # Überprüfe nochmal auf Inf/NaN
    assert not np.any(np.isnan(train_data_clean)) and not np.any(np.isinf(train_data_clean)), "Noch immer Inf/NaN Werte vorhanden!"
    
    # AutoGluon Training
    predictor = TabularPredictor(
        label=target,
        eval_metric='root_mean_squared_error',
        path=f'../models/autogluon_{asset}'
    ).fit(
        train_data_clean,
        time_limit=1200,  # 1h  3600
        presets='high_quality'
    )
    
    # Modell speichern
    models[asset] = predictor
    
    # Feature Importance
    #feature_importance[asset] = predictor.feature_importance(train_data[features + [target]])
    
    # Modell Performance
    model_performance[asset] = predictor.evaluate(test_data[features + [target]])

# Vorhersagen für Testdaten
predictions = {}
for asset in traded_assets:
    predictions[asset] = models[asset].predict(test_data)

# Nach dem Training
print("\n=== Modell Details ===")
leaderboard = predictor.leaderboard(silent=True)
print("\nTop Modelle:")
print(leaderboard.head())

print("\nEnsemble Informationen:")
if hasattr(predictor, '_trainer') and hasattr(predictor._trainer, '_get_model_weights'):
    weights = predictor._trainer._get_model_weights('WeightedEnsemble_L2')
    if weights is not None:
        for model, weight in weights.items():
            print(f"{model}: {weight:.3f}")

# Portfolio Performance berechnen
print("Starte Teste!")
portfolio_values = calculate_portfolio_value(predictions, test_data, traded_assets)

# Ergebnisse ausgeben
print("\n=== Training Results ===")
'''
for asset in traded_assets:
    print(f"\nModel Performance for {asset}:")
    print(f"RMSE: {model_performance[asset]}")
    
    print("\nBerechne Feature Importance...")
    try:
        # Verbesserte Datenvorbereitung für Feature Importance
        fi_data = train_data[features + [target]].copy()
        
        # Ersetze Inf-Werte durch sehr große/kleine Zahlen
        fi_data = fi_data.replace([np.inf], np.finfo(np.float64).max / 2)
        fi_data = fi_data.replace([-np.inf], np.finfo(np.float64).min / 2)
        
        # Entferne extreme Ausreißer (Optional)
        for col in fi_data.columns:
            if fi_data[col].dtype in [np.float64, np.float32]:
                q1 = fi_data[col].quantile(0.01)
                q3 = fi_data[col].quantile(0.99)
                iqr = q3 - q1
                fi_data[col] = fi_data[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        
        # Fülle verbleibende NaN-Werte
        fi_data = fi_data.fillna(method='ffill').fillna(0)
        
        # Überprüfe nochmal auf Inf/NaN
        assert not np.any(np.isnan(fi_data)) and not np.any(np.isinf(fi_data)), "Noch immer Inf/NaN Werte vorhanden!"
        
        feature_importance[asset] = predictor.feature_importance(fi_data)
        print("\nTop 10 Important Features:")
        print(feature_importance[asset].head(10))
    except Exception as e:
        print(f"\nFehler bei Feature Importance Berechnung: {str(e)}")
        print("Fahre mit Training fort...") '''

print("\n=== Portfolio Performance ===")
initial_value = 10000
final_value = portfolio_values[-1]
total_return = (final_value - initial_value) / initial_value * 100

print(f"Initial Portfolio Value: €{initial_value:,.2f}")
print(f"Final Portfolio Value: €{final_value:,.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Return: {(total_return / (len(test_data)/252)):.2f}%")

# Plot der Portfolio-Entwicklung
plt.figure(figsize=(12, 6))
# Konvertiere zu numpy array wenn nötig
if isinstance(portfolio_values, (list, tuple)):
    portfolio_values = np.array(portfolio_values)
elif hasattr(portfolio_values, 'values'):  # Falls es ein pandas Series/DataFrame ist
    portfolio_values = portfolio_values.values

plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value (€)')
plt.grid(True)
plt.savefig('portfolio_performance.png')
plt.close()

# Speichere detaillierte Ergebnisse
results = {
    'model_performance': model_performance,
    'feature_importance': feature_importance,
    'portfolio_values': portfolio_values,
    'final_portfolio_value': final_value,
    'total_return': total_return
}

# Als Pickle speichern
import pickle
with open('trading_results.pkl', 'wb') as f:
    pickle.dump(results, f)