import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import sys
import joblib
import time
from tqdm import tqdm
from tensorflow.keras.models import load_model

sys.path.append('/Users/niels/Htwg/Bachelorarbeit/BachelorarbeitLocal')

from standardStrategy import standard_bitcoin_strategy
from randomStrategy import random_bitcoin_strategy

from supervisedLearning import lopezDePrado
from supervisedLearning import lopezDePradoImproved
from supervisedLearning import lopezDePradoLstm
from supervisedLearning import lstm
from supervisedLearning.lstm import LSTMModel, load_lstm_model
from reinforcmentLearning.PpoLstm import PPOAgent, load_and_preprocess_data
from reinforcmentLearning.PpoLstmImp import PPOAgent as PPOAgentImproved, load_and_preprocess_data as load_and_preprocess_data_improved
from reinforcmentLearning.PpoLstmImp2 import PPOAgent, load_and_preprocess_data



# Überprüfen , ob MPS verfügbar ist
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")
# Lade Reinforcement Learning Modul hier
#from reinforcmentLearning.rlbtc import load_trained_model, prepare_rl_features, DQNAgent

def load_data(file_path, test_size=1):
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['Open time'])
    data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)
    test_data = data.iloc[-int(len(data) * test_size):]
    return test_data

def prepare_models():
    models = []

    # Random Forest Modell
    try:
        rf_model = joblib.load("../models/random_forest_model.joblib")
        models.append({
            'name': 'Random Forest',
            'model': rf_model,
            'prepare_features': lopezDePrado.prepare_features
        })
        print("Random Forest Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Random Forest Modells: {e}")

    # Random Forest Improved Modell
    try:
        rfi_model = joblib.load("../models/random_forest_model_improved.joblib")
        models.append({
            'name': 'Random Forest Improved',
            'model': rfi_model,
            'prepare_features': lopezDePradoImproved.prepare_features
        })
        print("Random Forest Improved Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Random Forest Improved Modells: {e}")

    # LSTM Modell
    try:
        lstm_model, lstm_scaler = load_lstm_model()
        models.append({
            'name': 'LSTM',
            'model': lstm_model,
            'scaler': lstm_scaler,
            'prepare_features': lstm.prepare_lstm_features_live
        })
        print("LSTM Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des LSTM Modells: {e}")

    # López de Prado LSTM Modell
    try:
        lstm_model = load_model("../models/lopez_deprado_lstm_model.h5")
        scaler = joblib.load("../models/lopez_deprado_lstm_scaler.joblib")
        models.append({
            'name': 'López de Prado LSTM',
            'model': lstm_model,
            'scaler': scaler,
            'prepare_features': lopezDePradoLstm.prepare_features
        })
        print("López de Prado LSTM Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des López de Prado LSTM Modells: {e}")

    # PPO LSTM Reinforcement Learning
    try:
        #input_filename = "../data/files/btc_usdt_1h_candlestick.csv"
        #_, scaler = load_and_preprocess_data(input_filename)
        scaler = joblib.load("../models/ppo_lstm_scaler.joblib")
        
        # Überprüfe den Typ des Scalers
        print(f"Type of scaler after loading: {type(scaler)}")
        
        # Überprüfe die Attribute des Scalers
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            print("Scaler has mean_ and scale_ attributes.")
            print(f"Number of features in scaler: {len(scaler.mean_)}")
            print(f"First few means: {scaler.mean_[:5]}")
            print(f"First few scales: {scaler.scale_[:5]}")
        else:
            print("Warning: Scaler does not have expected attributes.")
        
        # Überprüfe, ob der Scaler mit den erwarteten Features trainiert wurde
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'SMA_20', 'SMA_50', 'RSI']
        if hasattr(scaler, 'feature_names_in_'):
            missing_features = set(expected_features) - set(scaler.feature_names_in_)
            if missing_features:
                print(f"Warning: Scaler is missing these expected features: {missing_features}")
            else:
                print("Scaler has all expected features.")
        else:
            print("Warning: Scaler does not have feature_names_in_ attribute.")
        
        ppo_lstm_model = PPOAgent(input_dim=9, hidden_dim=64, output_dim=3, lr_actor=0.0003, lr_critic=0.0003, 
                                gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
        ppo_lstm_model.load_model("../models/ppo_lstm_btc_model.pth")
        models.append({
            'name': 'PPO-LSTM',
            'model': ppo_lstm_model,
            'scaler': scaler,
            'prepare_features': prepare_ppo_lstm_features
        })
        
        print("PPO-LSTM Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-LSTM Modells: {e}")

    # PPO LSTM Improved2
    try:
        #input_filename = "../data/files/btc_usdt_1h_candlestick.csv"
        #_, scaler = load_and_preprocess_data(input_filename)
        scaler_imp2 = joblib.load("../models/ppo_lstm_scaler_imp2.joblib")
        
        # Überprüfe den Typ des Scalers
        print(f"Type of scaler after loading: {type(scaler)}")
        
        # Überprüfe die Attribute des Scalers
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            print("Scaler has mean_ and scale_ attributes.")
            print(f"Number of features in scaler: {len(scaler.mean_)}")
            print(f"First few means: {scaler.mean_[:5]}")
            print(f"First few scales: {scaler.scale_[:5]}")
        else:
            print("Warning: Scaler does not have expected attributes.")
        
        # Überprüfe, ob der Scaler mit den erwarteten Features trainiert wurde
        expected_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'SMA_20', 'SMA_50', 'RSI']
        if hasattr(scaler, 'feature_names_in_'):
            missing_features = set(expected_features) - set(scaler.feature_names_in_)
            if missing_features:
                print(f"Warning: Scaler is missing these expected features: {missing_features}")
            else:
                print("Scaler has all expected features.")
        else:
            print("Warning: Scaler does not have feature_names_in_ attribute.")
        
        ppo_lstm_model_imp2 = PPOAgent(input_dim=9, hidden_dim=64, output_dim=3, lr_actor=0.0003, lr_critic=0.0003, 
                                gamma=0.99, epsilon=0.2, value_coef=0.5, entropy_coef=0.01)
        ppo_lstm_model.load_model("../models/ppo_lstm_btc_model_imp2.pth")
        models.append({
            'name': 'PPO-LSTM Imp2',
            'model': ppo_lstm_model_imp2,
            'scaler': scaler_imp2,
            'prepare_features': prepare_ppo_lstm_features
        })
        
        print("PPO-LSTM Imp2 Modell geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des PPO-LSTM Imp2 Modells: {e}")
   
    # PPO LSTM Improved
    try:
        print("Versuche, PPO-LSTM Improved Modell zu laden...")
        
        # Laden des Scalers
        scaler_improved = joblib.load("../models/ppo_lstm_scaler_imp.joblib")
        print("Scaler für PPO-LSTM Improved erfolgreich geladen.")
        
        # Initialisierung des Agenten mit output_dim=1
        ppo_lstm_model_improved = PPOAgentImproved(
            input_dim=9, 
            hidden_dim=64, 
            output_dim=3, 
            lr_actor=0.0003, 
            lr_critic=0.0003, 
            gamma=0.99, 
            epsilon=0.2, 
            value_coef=0.5, 
            entropy_coef=0.01
        )
        ppo_lstm_model_improved.load_model("../models/ppo_lstm_btc_model_imp.pth")
        models.append({
            'name': 'PPO-LSTM Improved',
            'model': ppo_lstm_model_improved,
            'scaler': scaler_improved,
            'prepare_features': prepare_ppo_lstm_features
        })
        print("PPO-LSTM Improved Modell erfolgreich geladen und zur Liste hinzugefügt.")
        
    except Exception as e:
        print(f"Fehler beim Laden des PPO-LSTM Improved Modells: {e}")

    return models
    
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_ppo_lstm_features(data_window, scaler):
    #print(f"Originale data_window Länge: {len(data_window)}")
    
    # Berechne technische Indikatoren
    data_window['returns'] = data_window['Close'].pct_change()
    data_window['SMA_20'] = data_window['Close'].rolling(window=20).mean()
    data_window['SMA_50'] = data_window['Close'].rolling(window=50).mean()
    data_window['RSI'] = calculate_rsi(data_window['Close'])
    
    # Fülle NaN-Werte
    data_window = data_window.ffill().bfill()
    
    # Verwende nur die Features, die beim Training verwendet wurden
    features = scaler.feature_names_in_
    
    # Entferne zusätzliche Features und stelle die Reihenfolge sicher
    data_window = data_window.reindex(columns=features)
    
    # Prüfe auf fehlende Features
    missing_features = set(features) - set(data_window.columns)
    if missing_features:
        print(f"Warnung: Fehlende Features: {missing_features}")
        # Optional: Fehlende Features mit Nullen auffüllen
        for feature in missing_features:
            data_window[feature] = 0
    
    # Normalisiere die Daten
    normalized_data = scaler.transform(data_window)
    
    return normalized_data

def simulate_realtime_trading(data, models, strategies, initial_capital=10000, interval_minutes=60):
    # Füge 'Buy and Hold' zur Liste der Strategienamen hinzu
    strategy_names = [m['name'] for m in models] + [s['name'] for s in strategies] + ['Buy and Hold']
    
    results = {name: pd.DataFrame(columns=['Timestamp', 'Price', 'Signal', 'Position', 'PnL']) 
               for name in strategy_names}
    
    current_positions = {name: 0 for name in results.keys()}

    # Initialisiere Buy-and-Hold-Strategie
    buy_and_hold_initial_price = data['Close'].iloc[0]
    buy_and_hold_position = initial_capital / buy_and_hold_initial_price
    
    # Fenstergröße für die Feature-Berechnung (z.B. 50 Datenpunkte)
    window_size = 100
    
    for i in tqdm(range(len(data))):
        # Verwende nur die Daten innerhalb des Fensters
        start_idx = max(i - window_size + 1, 0)
        data_window = data.iloc[start_idx:i+1].copy()
        current_row = data.iloc[i]
        current_price = current_row['Close']
        timestamp = current_row.name

        # Buy-and-Hold-Strategie
        buy_and_hold_pnl = buy_and_hold_position * (current_price - buy_and_hold_initial_price)
        new_row = pd.DataFrame({
            'Timestamp': [timestamp],
            'Price': [current_price],
            'Signal': [1],  # Immer 1, da wir immer halten
            'Position': [buy_and_hold_position],
            'PnL': [buy_and_hold_pnl]
        })
        results['Buy and Hold'] = pd.concat([results['Buy and Hold'], new_row], ignore_index=True)



        for model in models:
            name = model['name']
            prepare_features = model['prepare_features']
            
            if name == 'PPO-LSTM':
                features = prepare_features(data_window, model['scaler'])
                if len(features) > 0:
                    # Verwende nur das letzte Feature
                    input_feature = torch.FloatTensor(features[-1]).unsqueeze(0).unsqueeze(0).to(device)
                    action_probs, _ = model['model'].actor(input_feature, None)
                    signal = torch.argmax(action_probs).item() - 1  # -1 für Short, 0 für Halten, 1 für Long
                else:
                    signal = 0
            elif name == 'PPO-LSTM Imp2':
                features = prepare_features(data_window, model['scaler'])
                if len(features) > 0:
                    # Verwende nur das letzte Feature
                    input_feature = torch.FloatTensor(features[-1]).unsqueeze(0).unsqueeze(0).to(device)
                    action_probs, _ = model['model'].actor(input_feature, None)
                    signal = torch.argmax(action_probs).item() - 1  # -1 für Short, 0 für Halten, 1 für Long
                else:
                    signal = 0
            elif name == 'PPO-LSTM Improved':
                features = prepare_features(data_window, model['scaler'])
                if len(features) > 0:
                    input_feature = torch.FloatTensor(features[-1]).unsqueeze(0).unsqueeze(0).to(device)
                    mean, std, hidden = model['model'].actor(input_feature, None)
                    action = torch.normal(mean, std)
                    
                    # Direkte Verwendung der drei Aktionswerte
                    action_values = action.squeeze().tolist()
                    
                    # Bestimmen des Signals basierend auf dem höchsten Aktionswert
                    max_action_index = action_values.index(max(action_values))
                    if max_action_index == 0:
                        signal = -1  # Short
                    elif max_action_index == 2:
                        signal = 1   # Long
                    else:
                        signal = 0   # Halten
                else:
                    signal = 0
            elif name == 'LSTM':
                features = prepare_features(data_window, model['scaler'])
                if len(features) > 0:
                    # Verwende nur das letzte Feature
                    input_feature = features[-1].unsqueeze(0)
                    predictions = lstm_predict(model['model'], input_feature)
                    signal = lstm.generate_lstm_signals(predictions, [current_price])[-1]
                else:
                    signal = 0
            elif name == 'López de Prado LSTM':
                features = prepare_features(data_window, data_window['Close'])
                if len(features) > 0:
                    scaled_features = model['scaler'].transform(features)
                    # Erstelle Sequenzen für das LSTM-Modell
                    sequence = np.array([scaled_features[-10:]])  # Nehme die letzten 10 Zeitschritte
                    predictions = model['model'].predict(sequence)
                    signal = np.argmax(predictions[0]) - 1  # -1 für Short, 0 für Halten, 1 für Long
                else:
                    signal = 0
            else:
                features = prepare_features(data_window, data_window['Close'])
                required_features = model['model'].feature_names_in_
                # Prüfen ob alle erforderlichen Features vorhanden sind
                if not features.empty and set(required_features).issubset(features.columns):
                    input_feature = features[required_features].iloc[[-1]]
                    signal = model['model'].predict(input_feature)[0]
                else:
                    signal = 0

            position = update_position(current_positions[name], signal)
            pnl = calculate_pnl(position, current_price, results[name])
            
            new_row = pd.DataFrame({
                'Timestamp': [timestamp],
                'Price': [current_price],
                'Signal': [signal],
                'Position': [position],
                'PnL': [pnl]
            })
            results[name] = pd.concat([results[name], new_row], ignore_index=True)
            
            current_positions[name] = position

        for strategy in strategies:
            name = strategy['name']
            # Stelle sicher, dass der Index i innerhalb der Grenzen der Signale liegt
            if i < len(strategy['signals']):
                signal = strategy['signals'].iloc[i]
            else:
                signal = 0  # Oder eine andere geeignete Standardaktion
            
            position = update_position(current_positions[name], signal)
            pnl = calculate_pnl(position, current_price, results[name])
            
            new_row = pd.DataFrame({
                'Timestamp': [timestamp],
                'Price': [current_price],
                'Signal': [signal],
                'Position': [position],
                'PnL': [pnl]
            })
            results[name] = pd.concat([results[name], new_row], ignore_index=True)
            
            current_positions[name] = position

        # Entfernen oder reduziere die Zeitverzögerung für eine effizientere Ausführung
        time.sleep(0.1)  # Kann entfernt oder angepasst werden
    
    return results

def update_position(current_position, signal):
    if signal > 0 and current_position <= 0:
        return 1
    elif signal < 0 and current_position >= 0:
        return -1
    else:
        return current_position

def lstm_predict(model, features):
    model.eval()
    with torch.no_grad():
        predictions = model(features).cpu().numpy()
    return predictions.flatten()

def calculate_pnl(position, current_price, previous_data):
    if previous_data.empty:
        return 0
    previous_price = previous_data['Price'].iloc[-1]
    previous_position = previous_data['Position'].iloc[-1]
    return position * (current_price - previous_price) + previous_data['PnL'].iloc[-1]


def plot_realtime_results(results):
    plt.figure(figsize=(12, 6))
    for name, df in results.items():
        plt.plot(df['Timestamp'], df['PnL'], label=name)
    plt.title('Echtzeit-Strategie-Vergleich')
    plt.xlabel('Datum')
    plt.ylabel('Gewinn und Verlust (PnL)')
    plt.legend()
    plt.tight_layout()
    
    # Speichern des Plots
    plt.savefig('realtime_strategy_comparison.png')
    print("Plot gespeichert unter 'realtime_strategy_comparison.png'")
    
    plt.show()

def compute_metrics(results):
    metrics_summary = {}
    for name, df in results.items():
        metrics = {}
        print(f"\nStrategie: {name}")
        
        # Anzahl der Trades (Änderungen in der Position)
        df['Trade'] = (df['Position'].diff() != 0)
        df['Trade'] = df['Trade'].fillna(False)
        total_trades = df['Trade'].sum()
        print(f"Gesamte Anzahl der Trades: {total_trades}")
        
        # Anzahl der Kauf-, Verkaufs- und Halteaktionen
        num_buy = (df['Signal'] > 0).sum()
        num_sell = (df['Signal'] < 0).sum()
        num_hold = (df['Signal'] == 0).sum()
        print(f"Anzahl der Kaufaktionen: {num_buy}")
        print(f"Anzahl der Verkaufsaktionen: {num_sell}")
        print(f"Anzahl der Halteaktionen: {num_hold}")
        print(f"Anzahl der Aktionen Gesamt: {num_buy + num_sell + num_hold}")
        
        # Gesamtertrag
        total_return = df['PnL'].iloc[-1]
        print(f"Gesamtertrag: {total_return:.2f}")
        
        # Renditen für Sharpe Ratio berechnen
        # Renditen für Sharpe Ratio berechnen
        returns = df['PnL'].diff()
        returns = returns.where(pd.notnull(returns), 0.0)  # Ersetze NaN durch 0.0
        average_return = returns.mean()
        return_std = returns.std()
        if return_std != 0:
            # Da die Daten stündlich sind, gibt es ca. 8760 Stunden pro Jahr
            sharpe_ratio = (average_return / return_std) * np.sqrt(8760)
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        else:
            print("Sharpe Ratio: Nicht berechenbar (Standardabweichung der Renditen ist 0)")
        
        # Maximaler Drawdown berechnen
        cumulative_returns = df['PnL']
        running_max = cumulative_returns.cummax()
        drawdown = (running_max - cumulative_returns)
        max_drawdown = drawdown.max()
        print(f"Maximaler Drawdown: {max_drawdown:.2f}")

        # Berechne den tatsächlichen Gewinn (PnL - Anfangskapital)
        initial_capital = 10000
        total_return = df['PnL'].iloc[-1]
        actual_profit = total_return - initial_capital
        print(f"Anfangskapital: {initial_capital:.2f}")
        print(f"Endkapital: {total_return:.2f}")
        print(f"Tatsächlicher Gewinn/Verlust: {actual_profit:.2f}")
        
        # Speichere die Metriken
        metrics['total_trades'] = total_trades
        metrics['num_buy'] = num_buy
        metrics['num_sell'] = num_sell
        metrics['num_hold'] = num_hold
        metrics['total_return'] = total_return
        metrics['actual_profit'] = actual_profit
        metrics['sharpe_ratio'] = sharpe_ratio if return_std != 0 else 0
        metrics['max_drawdown'] = max_drawdown
        
        metrics_summary[name] = metrics
    
    return metrics_summary

def plot_model_performance(results, data):
    for name, df in results.items():
        plt.figure(figsize=(12, 6))
        
        # Erstelle zwei y-Achsen
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Plotte den BTC/USDT Preisverlauf
        ax1.plot(data.index, data['Close'], label='BTC/USDT Preis', color='blue')
        ax1.set_xlabel('Datum')
        ax1.set_ylabel('BTC/USDT Preis', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Plotte den BTC/USDT Bestand des Modells
        ax2.plot(df['Timestamp'], df['Position'].cumsum(), label='BTC/USDT Bestand', color='red')
        ax2.set_ylabel('BTC/USDT Bestand', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title(f'{name} - Preisverlauf und Bestand')
        
        # Kombinieredie Legenden
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Speichern des Plots
        plt.savefig(f'{name.replace(" ", "_").lower()}_performance.png')
        print(f"Plot für {name} gespeichert unter '{name.replace(' ', '_').lower()}_performance.png'")
        
        plt.close()

def run_multiple_simulations(num_runs=10):
    all_metrics = []
    
    for run in range(num_runs):
        print(f"\nSimulation {run + 1}/{num_runs}")
        data = load_data("../data/files/btc_usdt_1h_candlestick.csv")
        test_size = int(len(data) * 0.20)
        data = data.iloc[-test_size:]
        
        models = prepare_models()
        standard_data = standard_bitcoin_strategy(data.copy())
        random_data = random_bitcoin_strategy(data.copy())
        
        strategies = [
            {'name': 'Standard Strategie', 'signals': standard_data['Signal']},
            {'name': 'Zufallsstrategie', 'signals': random_data['Signal']}
        ]
        
        results = simulate_realtime_trading(data, models, strategies)
        metrics = compute_metrics(results)
        all_metrics.append(metrics)
    
    # Berechne Durchschnittswerte
    print("\n=== Durchschnittliche Ergebnisse nach", num_runs, "Simulationen ===")
    avg_metrics = {}
    
    for strategy in all_metrics[0].keys():
        avg_metrics[strategy] = {
            'avg_profit': np.mean([m[strategy]['actual_profit'] for m in all_metrics]),
            'avg_sharpe': np.mean([m[strategy]['sharpe_ratio'] for m in all_metrics]),
            'avg_max_drawdown': np.mean([m[strategy]['max_drawdown'] for m in all_metrics])
        }
        
        print(f"\n{strategy}:")
        print(f"Durchschnittlicher Gewinn/Verlust: {avg_metrics[strategy]['avg_profit']:.2f}")
        print(f"Durchschnittliche Sharpe Ratio: {avg_metrics[strategy]['avg_sharpe']:.2f}")
        print(f"Durchschnittlicher Max Drawdown: {avg_metrics[strategy]['avg_max_drawdown']:.2f}")


def main():
    '''
    data = load_data("../data/files/btc_usdt_1h_candlestick.csv") # test_size=0.2
    test_size = int(len(data) * 0.20) # Lade nur die letzten 20 des Datensatztes
    data = data.iloc[-test_size:]
    
    print(f"Geladene {len(data)} Datenpunkte für den Live-Test")
    close_prices = data['Close']

    # Plot erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(close_prices, label='Schlusskurs', color='blue')
    plt.title('BTC/USDT Schlusskurse')
    plt.xlabel('Zeit')
    plt.ylabel('Preis (USDT)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    models = prepare_models()
    for model in models:
        print(model)

    standard_data = standard_bitcoin_strategy(data.copy())
    random_data = random_bitcoin_strategy(data.copy())

    print(f"Länge der Standardsignale: {len(standard_data)}")
    print(f"Länge der Zufallssignale: {len(random_data)}")
    print(f"Länge der Testdaten: {len(data)}")

    strategies = [
        {'name': 'Standard Strategie', 'signals': standard_data['Signal']},
        {'name': 'Zufallsstrategie', 'signals': random_data['Signal']}
    ]

    results = simulate_realtime_trading(data, models, strategies)

    print("Verfügbare Modelle in den Ergebnissen:")
    for name in results.keys():
        print(name)

    plot_realtime_results(results)

    compute_metrics(results)

    plot_model_performance(results, data)

    # Ausgabe des Gesamtertrags für Buy-and-Hold-Strategie
    buy_and_hold_return = results['Buy and Hold']['PnL'].iloc[-1]
    print(f"\nBuy-and-Hold Gesamtertrag: {buy_and_hold_return:.2f}") '''
    run_multiple_simulations(num_runs=10)

    

if __name__ == "__main__":
    main()