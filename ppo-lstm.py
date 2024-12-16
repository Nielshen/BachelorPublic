import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.model_selection import train_test_split
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import optuna 

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Berechne die Feature-Dimensionen
        self.n_features = observation_space.shape[0] // 10  # Teile durch lookback_window
        self.hidden_size = 64
        self.num_layers = 2
        
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.linear = nn.Linear(self.hidden_size, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape für LSTM: (batch_size, seq_length, features)
        batch_size = observations.shape[0]
        seq_length = 10  # lookback_window
        n_features = observations.shape[1] // seq_length
        
        x = observations.view(batch_size, seq_length, n_features)
        
        # LSTM und Linear Layer
        lstm_out, (hidden, _) = self.lstm(x)
        features = self.linear(hidden[-1])
        
        return features
    


# Überprüfen, ob MPS (Metal Performance Shaders) verfügbar ist
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Verwendetes Gerät: {device}")

# Laden der Daten
data = pd.read_pickle('../data/files/multi_data_indicators.pkl')

# Definition der gehandelten Assets
traded_assets = ['Crypto_BTCUSDT']

# Laden der Scaler
scalers = {}
for asset_type in ['Stock', 'Crypto', 'Commodity', 'Bond', 'RE', 'Forex', 'ETF']:
    scaler_path = f'../data/files/{asset_type}_scaler.pkl'
    if os.path.exists(scaler_path):
        scalers[asset_type] = joblib.load(scaler_path)

# Funktion zum Denormalisieren der Preise
def denormalize_price(normalized_price, asset):
    '''denormalized_price = np.expm1(normalized_price)
    print(f"Asset: {asset}")
    print(f"Normalized Price: {normalized_price:.4f}")
    print(f"Denormalized Price: {denormalized_price:.4f}")
    return denormalized_price'''
    return normalized_price

# Denormalisieren der Preise für den Plot
denormalized_data = pd.DataFrame(index=data.index)
for asset in traded_assets:
    denormalized_data[asset] = data[asset].apply(lambda x: denormalize_price(x, asset))

# Aufteilen in Trainings- und Testdaten
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Analyse der Zeiträume
print("\nAnalyse der Datenzeiträume:")
print(f"\nTrainingsdaten:")
print(f"Start: {train_data.index[0]}")
print(f"Ende: {train_data.index[-1]}")
print(f"Länge: {len(train_data)} Datenpunkte/Tage)")

print(f"\nTestdaten:")
print(f"Start: {test_data.index[0]}")
print(f"Ende: {test_data.index[-1]}")
print(f"Länge: {len(test_data)} Datenpunkte/Tage)")

# Plotten der Kurse mit denormalisierten Preisen
plt.figure(figsize=(12, 6))
for asset in traded_assets:
    plt.plot(denormalized_data.index, denormalized_data[asset], label=asset)
plt.axvline(x=train_data.index[-1], color='r', linestyle='--', label='Train/Test Split')
plt.legend()
plt.title('Asset Prices (Denormalized)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.yscale('log')  # Verwende eine logarithmische Skala für die y-Achse
plt.show()

def get_asset_start_dates(data, assets):
    start_dates = {}
    for asset in assets:
        non_zero_values = data[asset][data[asset] != 0]
        if not non_zero_values.empty:
            start_dates[asset] = non_zero_values.index[0]
        else:
            start_dates[asset] = pd.Timestamp.max
    return start_dates

def get_available_assets(current_date, asset_start_dates):
    return [asset for asset, start_date in asset_start_dates.items() if current_date >= start_date]

class TradingEnv(gym.Env):
    def __init__(self, data, traded_assets, lookback_window=10, is_training=True): #TODO: change lookback window: 50 lange trends, 100 sehr lang
        super(TradingEnv, self).__init__()

        relevant_columns = []
        for asset in traded_assets:
            asset_columns = [col for col in data if asset in col]
            relevant_columns.extend(asset_columns)
            print(f"Gefundene Spalten für {asset}:")
            print("\n".join(f"- {col}" for col in asset_columns))
            print()
        
        # DataFrame auf relevante Spalten reduzieren
        self.data = data[relevant_columns]

        n_features = len(data.columns) + len(traded_assets) * 2 + 1
        print(f"Reduzierte Feature-Dimension: {n_features} Spalten")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features * lookback_window,),  # Flache Form
            dtype=np.float32
        )
        
        self.lookback_window = lookback_window
        self.n_features = n_features

        self.data = data
        self.traded_assets = traded_assets
        self.is_training = is_training
        self.current_step = 0
        self.initial_balance = 10000

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance 
        self.max_net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.holding_periods = {asset: 0 for asset in traded_assets}
        self.holdings = {asset: 0 for asset in traded_assets}
        self.short_prices = {asset: 0 for asset in traded_assets}
        # Berechne die Startdaten für alle Assets
        self.asset_start_dates = get_asset_start_dates(data, traded_assets)
        self.current_traded_assets = get_available_assets(data.index[0], self.asset_start_dates)
        
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([2, len(traded_assets) - 1, 1]),
            dtype=np.float32
        )

        self.action_counts = {'hold': 0, 'long': 0, 'short': 0}
        self.action_counts_asset = {asset: {'hold': 0, 'long': 0, 'short': 0} for asset in traded_assets}
        # Trading History
        self.returns_history = []
        self.episode_rewards = []  # Liste für Rewards der aktuellen Episode
        self.all_episode_rewards = []
        # Dictionaries für Entry-Preise
        self.long_positions = {
            asset: {
                'total_amount': 0,
                'avg_price': 0
            } for asset in traded_assets
        }
        
        self.short_positions = {
            asset: {
                'total_amount': 0,
                'avg_price': 0
            } for asset in traded_assets
        }

    def reset(self):
        if self.is_training:
            print("Umgebung wird zurückgesetzt!")  # Debugging-Ausgabe
            self.current_step = 0
            self.balance = self.initial_balance
            self.net_worth = self.initial_balance
            self.holdings = {asset: 0 for asset in self.traded_assets}
            self.short_prices = {asset: 0 for asset in self.traded_assets}
            self.action_counts = {'hold': 0, 'long': 0, 'short': 0}
            if len(self.episode_rewards) > 0:
                self.all_episode_rewards.append(self.episode_rewards)
            self.episode_rewards = [] 
        else:
            print("Test-Modus: Umgebung wird nicht zurückgesetzt.")
            # Im Testmodus setzen wir nur den current_step zurück
            self.current_step = 0

        return self._next_observation()

    def _next_observation(self):
        observations = []
        for i in range(self.lookback_window):
            idx = max(0, self.current_step - (self.lookback_window - 1 - i))
            
            # Marktdaten aus self.data
            obs = self.data.iloc[idx].values.astype(np.float32)
            
            # Holdings, Balance und Tradable
            holdings = np.array([self.holdings[asset] for asset in self.traded_assets], dtype=np.float32)
            balance = np.array([self.balance], dtype=np.float32)
            tradable = np.array([1 if asset in self.current_traded_assets else 0 
                               for asset in self.traded_assets], dtype=np.float32)
            
            # Kombiniere alle Features
            step_obs = np.concatenate([obs, holdings, balance, tradable])
            observations.append(step_obs)
        
        # Konvertiere zu numpy array und flatten
        observations = np.array(observations, dtype=np.float32)
        flattened_obs = observations.flatten()
        
        return flattened_obs

    @staticmethod
    def denormalize_price(normalized_price, asset):
        '''denormalized_price = np.expm1(normalized_price)
        print(f"Asset: {asset}")
        print(f"Normalized Price: {normalized_price:.4f}")
        print(f"Denormalized Price: {denormalized_price:.4f}")
        return denormalized_price'''
        return normalized_price
    
    def update_average_price(self, position_dict, new_amount, new_price, asset):
        """Aktualisiert den durchschnittlichen Einstandspreis für ein spezifisches Asset"""
        if position_dict[asset]['total_amount'] + new_amount == 0:
            return 0
        
        total_value = (position_dict[asset]['total_amount'] * position_dict[asset]['avg_price']) + (new_amount * new_price)
        new_total_amount = position_dict[asset]['total_amount'] + new_amount
        return total_value / new_total_amount
    
    def step(self, action):
        previous_net_worth = self.net_worth
        reward = 0

        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if self.current_step >= len(self.data):
            return self._next_observation(), 0, True, {}
        
        current_date = self.data.index[self.current_step]
        self.current_traded_assets = get_available_assets(current_date, self.asset_start_dates)
        
        action_type, asset_index, amount_fraction = action
        action_type = int(action_type)  # Runde auf den nächsten Integer
        asset_index = int(asset_index)  # Runde auf den nächsten Integer
        asset = self.traded_assets[asset_index]
        
        if asset not in self.current_traded_assets:
            return self._next_observation(), -1, False, {"invalid_action": "Asset not available"}

        normalized_price = self.data[asset].iloc[self.current_step]
        current_price = self.denormalize_price(normalized_price, asset)
        
        # Berechne den Betrag basierend auf dem Bruchteil des verfügbaren Vermögens
        max_amount = self.balance / current_price
        amount = max_amount * amount_fraction
        
        transaction_cost = 0.001
                
        if action_type == 0:  # Halten
                self.action_counts['hold'] += 1
                #print("HOLD - Keine Änderungen")
                
        elif action_type == 1:  # Kaufen/Long
            if self.holdings[asset] < 0:  # Short-Position schließen
                buy_amount = min(-self.holdings[asset], amount)

                old_balance = self.balance
                old_holdings = self.holdings[asset]
                self.holdings[asset] += buy_amount
                avg_entry_price = self.short_positions[asset]['avg_price']
                original_margin = avg_entry_price * buy_amount
                short_profit = (avg_entry_price - current_price) * buy_amount
                self.balance += original_margin + short_profit

                self.short_positions[asset]['total_amount'] -= buy_amount
                if self.short_positions[asset]['total_amount'] == 0:
                    self.short_positions[asset]['avg_price'] = 0
                '''
                print(f"Schließe Short-Position für {asset}:")
                print(f"Current Asset Price/Close Price: {current_price}")
                print(f"Close Amount: {buy_amount}")
                print(f"Balance: {old_balance} -> {self.balance}")
                print(f"Holdings: {old_holdings} -> {self.holdings[asset]}")
                print(f"Avg Short Entry Price: {avg_entry_price}")
                print(f"Short Profit: {short_profit}")'''
                
            else:  # Long-Position eröffnen oder erweitern
                cost = current_price * amount * (1 + transaction_cost)
                if self.balance >= cost:
                    old_balance = self.balance
                    old_holdings = self.holdings[asset]
                    self.balance -= cost
                    self.holdings[asset] += amount
                    
                    self.long_positions[asset]['avg_price'] = self.update_average_price(
                        self.long_positions, amount, current_price, asset
                    )
                    self.long_positions[asset]['total_amount'] += amount
                    '''
                    print(f"Eröffne/Erweitere Long-Position für {asset}:")
                    print(f"Current Asset Price: {current_price}")
                    print(f"Buy Amount: {amount}")
                    print(f"Cost with Fee: {cost}")
                    print(f"Balance: {old_balance} -> {self.balance}")
                    print(f"Holdings: {old_holdings} -> {self.holdings[asset]}")
                    print(f"New Avg Long Entry Price: {self.long_positions[asset]['avg_price']}")'''
                    self.action_counts['long'] += 1
                        
        elif action_type == 2:  # Verkaufen/Short
            if self.holdings[asset] > 0:  # Long-Position schließen
                sell_amount = min(self.holdings[asset], amount)
                
                old_balance = self.balance
                old_holdings = self.holdings[asset]
                
                avg_entry_price = self.long_positions[asset]['avg_price']
                original_margin = avg_entry_price * sell_amount
                long_profit = (current_price - avg_entry_price) * sell_amount
                
                self.balance += original_margin + long_profit
                self.holdings[asset] -= sell_amount
                
                self.long_positions[asset]['total_amount'] -= sell_amount
                if self.long_positions[asset]['total_amount'] == 0:
                    self.long_positions[asset]['avg_price'] = 0
                '''
                print(f"Schließe Long-Position für {asset}:")
                print(f"Current Asset Price/Close Price: {current_price}")
                print(f"Close Amount: {sell_amount}")
                print(f"Balance: {old_balance} -> {self.balance}")
                print(f"Holdings: {old_holdings} -> {self.holdings[asset]}")
                print(f"Avg Long Entry Price: {avg_entry_price}")
                print(f"Long Profit: {long_profit}")'''
                
            else:  # Short-Position eröffnen oder erweitern
                cost = current_price * amount * (1 + transaction_cost)
                if self.balance >= cost:
                    old_balance = self.balance
                    old_holdings = self.holdings[asset]
                    self.balance -= cost
                    self.holdings[asset] -= amount
                    
                    self.short_positions[asset]['avg_price'] = self.update_average_price(
                        self.short_positions, amount, current_price, asset
                    )
                    self.short_positions[asset]['total_amount'] += amount
                    '''
                    print(f"Eröffne/Erweitere Short-Position für {asset}:")
                    print(f"Current Asset Price: {current_price}")
                    print(f"Buy Amount to Short: {amount}")
                    print(f"Cost with Fee: {cost}")
                    print(f"Balance: {old_balance} -> {self.balance}")
                    print(f"Holdings: {old_holdings} -> {self.holdings[asset]}")
                    print(f"New Avg Short Entry Price: {self.short_positions[asset]['avg_price']}")'''
                    self.action_counts['short'] += 1

        # Berechnung des Nettowerts
        position_value = sum(
            self.holdings[asset] * self.denormalize_price(self.data[asset].iloc[self.current_step], asset)
            for asset in self.traded_assets
        )
        self.net_worth = self.balance + position_value
                
        # Absoluter Reward (Gewinn/Verlust in Dollar)
        reward = self.net_worth
        
        # Handelsgebühr (z.B. $1 pro Trade)
        #action_type = int(action[0])
        #if action_type != 0:  # Wenn nicht "Halten"
            #reward -= 1
        
        #print(f"Reward = {reward:.2f}")
        self.episode_rewards.append(reward)     

        done = self.current_step >= len(self.data) - 1

        if done:
            print(f"AI Model: Episode ended. Final Net Worth = {self.net_worth:.2f}")
            #print(f"Episode Reward = {reward:.2f}")
            print(f"Action counts: Hold = {self.action_counts['hold']}, Long = {self.action_counts['long']}, Short = {self.action_counts['short']}")

        return self._next_observation(), reward, done, {}


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_net_worths = []

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            current_reward = self.locals.get("rewards")[0]
            self.episode_rewards.append(current_reward)
            
            # Drucke nur die Belohnung, da der Nettowert bereits in der Umgebung gedruckt wurde
            print(f"Episode {len(self.episode_rewards)}: Reward = {current_reward:.2f}")            
            # Setze die Umgebung zurück
            self.training_env.envs[0].reset()
        
        return True

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func

# Funktion zum Trainieren und Speichern des Modells
def train_and_save_model(train_data, traded_assets, model_path, num_episodes):
    train_env = DummyVecEnv([lambda: TradingEnv(train_data, traded_assets, is_training=True)])

    # Berechne die korrekten Timesteps
    steps_per_episode = len(train_data)
    total_timesteps = num_episodes * steps_per_episode

    print(f"\nTrainings-Konfiguration:")
    print(f"Datenpunkte pro Episode: {steps_per_episode}")
    print(f"Anzahl Episoden: {num_episodes}")
    print(f"Gesamte Trainingsschritte: {total_timesteps}")

    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[128, 64],  # Reduzierte Architektur
            vf=[128, 64]
        )
    )

    model = PPO(
        "MlpPolicy", 
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-4,  # Höhere Learning Rate
        n_steps=2048,        # Kürzere Sequenzen
        batch_size=64,       # Kleinere Batch Size
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  #Exploration
        verbose=1,
        device=device
    )
    #{'features_dim': 512, 'pi_1': 168, 'pi_2': 256, 'pi_3': 247, 'vf_1': 378, 'vf_2': 272, 'vf_3': 249, 
    # 'learning_rate': 1.0866071598177003e-05, 'n_steps': 7168, 'batch_size': 1344, 'n_epochs': 13, 
    # 'gamma': 0.9793693354339195, 'gae_lambda': 0.9907401554284695, 
    # 'clip_range': 0.374363512968238, 'ent_coef': 0.00842357955550196}

    class CombinedCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(CombinedCallback, self).__init__(verbose)
            self.episode_count = 0
            self.total_episodes = num_episodes
            self.episode_rewards = []
            self.episode_net_worths = []

        def _on_step(self) -> bool:
            if self.locals.get("dones")[0]:
                self.episode_count += 1
                current_reward = self.locals.get("rewards")[0]
                self.episode_rewards.append(current_reward)
                
                print(f"\nEpisode {self.episode_count}/{self.total_episodes}")
                print(f"Episode Reward = {current_reward:.2f}")
                
                # Setze die Umgebung zurück
                self.training_env.envs[0].reset()
                
                # Training beenden wenn alle Episoden durchlaufen wurden
                if self.episode_count >= self.total_episodes:
                    return False
            return True

    callback = CombinedCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Speichern des Modells
    model.save(model_path)
    print(f"\nModell gespeichert unter: {model_path}")

    # Feature Importance Analyse
    feature_importance = model.policy.features_extractor.lstm.weight_ih_l0.abs().sum(dim=0)
    feature_names = list(train_data.columns) + traded_assets + ['balance'] + ['tradable_' + asset for asset in traded_assets]
    
    # Verschiebe den Tensor auf die CPU und wandel ihn in ein NumPy-Array um
    feature_importance_np = feature_importance.detach().cpu().numpy()
    
    # Stelle sicher, dass feature_names und feature_importance_np die gleiche Länge haben
    if len(feature_names) != len(feature_importance_np):
        print(f"Warnung: Länge von feature_names ({len(feature_names)}) stimmt nicht mit der Länge von feature_importance_np ({len(feature_importance_np)}) überein.")
        min_length = min(len(feature_names), len(feature_importance_np))
        feature_names = feature_names[:min_length]
        feature_importance_np = feature_importance_np[:min_length]

    # Plotten der Feature Importance
    plot_feature_importance(feature_names, feature_importance_np, save_path='feature_importance.html', top_n=1000)
    plot_training_results(train_env.envs[0])

    return model


def plot_feature_importance(feature_names, feature_importance, save_path='feature_importance.html', top_n=None):
    # Stelle sicher, dass feature_names und feature_importance die gleiche Länge haben
    if len(feature_names) != len(feature_importance):
        print(f"Warnung: Länge von feature_names ({len(feature_names)}) stimmt nicht mit der Länge von feature_importance ({len(feature_importance)}) überein.")
        # Kürze die längere Liste auf die Länge der kürzeren
        min_length = min(len(feature_names), len(feature_importance))
        feature_names = feature_names[:min_length]
        feature_importance = feature_importance[:min_length]

    # Sortieren der Features nach Wichtigkeit (absteigend)
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    # Wenn top_n nicht angegeben ist, verwende alle Features
    if top_n is None or top_n > len(feature_names):
        top_n = len(feature_names)
    
    # Wähle die Top N Features
    top_features = sorted_indices[:top_n]
    
    # Erstellen des Balkendiagramms
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[feature_names[i] for i in top_features],
        x=feature_importance[top_features],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=max(600, top_n * 20),  # Dynamische Höhe basierend auf der Anzahl der Features
        width=800
    )
    
    # Speichern als interaktive HTML-Datei
    fig.write_html(save_path)
    print(f"Interaktiver Feature Importance Plot gespeichert unter: {save_path}")
    
    # Optional: Anzeigen des Plots im Browser
    fig.show()

def plot_training_results(env):
    """Plottet die Reward-Entwicklung über alle Episoden"""
    if not env.all_episode_rewards:
        print("Keine Trainingsdaten zum Plotten verfügbar")
        return

    plt.figure(figsize=(15, 10))
    
    # Plot für durchschnittliche Rewards pro Episode
    avg_rewards = [np.mean(episode) for episode in env.all_episode_rewards]
    plt.subplot(2, 1, 1)
    plt.plot(avg_rewards, label='Durchschnittlicher Reward pro Episode')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Durchschnittliche Rewards pro Episode')
    plt.xlabel('Episode')
    plt.ylabel('Durchschnittlicher Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot für kumulative Rewards pro Episode
    cumulative_rewards = [sum(episode) for episode in env.all_episode_rewards]
    plt.subplot(2, 1, 2)
    plt.plot(cumulative_rewards, label='Kumulativer Reward pro Episode', color='orange')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Kumulative Rewards pro Episode')
    plt.xlabel('Episode')
    plt.ylabel('Kumulativer Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Statistiken ausgeben
    print("\nTrainingsstatistiken:")
    print(f"Durchschnittlicher Reward über alle Episoden: {np.mean(avg_rewards):.2f}")
    print(f"Bester durchschnittlicher Reward: {max(avg_rewards):.2f}")
    print(f"Schlechtester durchschnittlicher Reward: {min(avg_rewards):.2f}")
    print(f"Durchschnittlicher kumulativer Reward: {np.mean(cumulative_rewards):.2f}")
    
# Funktion zum Laden und Testen des Modells
def load_and_test_model(test_data, traded_assets, model_path, silent=False):
    model = PPO.load(model_path, device=device)
    if not silent:
        print(f"Modell geladen von: {model_path}")

    test_env = DummyVecEnv([lambda: TradingEnv(test_data, traded_assets, is_training=False)])

    obs = test_env.reset()
    ai_net_worth_history = []
    buy_hold_net_worth_history = []
    cash_balance_history = []
    asset_performance_history = {asset: [] for asset in traded_assets}

    scalers = {}
    for asset_type in ['Stock', 'Crypto', 'Commodity', 'Bond', 'RE', 'Forex', 'ETF']:
        scaler_path = f'../data/files/{asset_type}_scaler.pkl'
        if os.path.exists(scaler_path):
            scalers[asset_type] = joblib.load(scaler_path)

    def denormalize_price(normalized_price, asset):
        '''denormalized_price = np.expm1(normalized_price)
        print(f"Asset: {asset}")
        print(f"Normalized Price: {normalized_price:.4f}")
        print(f"Denormalized Price: {denormalized_price:.4f}")
        return denormalized_price'''
        return normalized_price

    initial_investment = 10000 / len(traded_assets)
    buy_hold_holdings = {asset: initial_investment / denormalize_price(test_data[asset].iloc[0], asset) 
                         for asset in traded_assets}
    
        # Initialisierung für Random-Strategie
    random_net_worth_history = []
    random_balance = 10000
    random_holdings = {asset: 0 for asset in traded_assets}
    random_short_prices = {asset: 0 for asset in traded_assets}
    
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = test_env.step(action)
        done = dones[0]  # Da wir DummyVecEnv verwenden, nehmen wir das erste Element
        # Random-Strategie Logik
        step = len(ai_net_worth_history)
        random_net_worth = random_balance
        
        # Zufällige Handelsentscheidung für jedes Asset
        for asset in traded_assets:
            current_price = denormalize_price(test_data[asset].iloc[step], asset)
            
            # Zufällige Aktion: 0 = Halten, 1 = Kaufen, 2 = Verkaufen
            random_action = np.random.randint(0, 3)
            
            if random_action == 1 and random_balance > 1000:  # Kaufen
                amount = (random_balance * 0.2) / current_price  # 20% des verfügbaren Kapitals
                cost = current_price * amount * 1.001  # Mit Transaktionskosten
                if random_balance >= cost:
                    random_balance -= cost
                    random_holdings[asset] += amount
            
            elif random_action == 2 and random_holdings[asset] > 0:  # Verkaufen
                amount = random_holdings[asset]
                random_balance += current_price * amount * 0.999  # Mit Transaktionskosten
                random_holdings[asset] = 0
            
            # Aktuellen Wert der Position zum Nettowert hinzufügen
            random_net_worth += random_holdings[asset] * current_price
        
        random_net_worth_history.append(random_net_worth)


        ai_net_worth_history.append(test_env.envs[0].net_worth)
        cash_balance_history.append(test_env.envs[0].balance)

        # Debugging-Ausgaben
        step = len(ai_net_worth_history) - 1
        if step % 100 == 0 or done:
            print(f"Schritt {step}:")
            current_date = test_data.index[step]
            print(f"Datum: {current_date.strftime('%Y-%m-%d')}")
            print(f"  Nettowert: {test_env.envs[0].net_worth:.2f}")
            print(f"  Bargeldbestand: {test_env.envs[0].balance:.2f}")
            for asset in traded_assets:
                print(f"  {asset} Bestand: {test_env.envs[0].holdings[asset]:.2f}")
        
        # Berechnung der Asset-Performance unter Berücksichtigung von Long- und Short-Positionen
        for asset in traded_assets:
            current_price = denormalize_price(test_data[asset].iloc[step], asset)
            holdings = test_env.envs[0].holdings[asset]
            short_price = test_env.envs[0].short_prices[asset]
            
            if holdings > 0:  # Long-Position
                performance = holdings * current_price
            elif holdings < 0:  # Short-Position
                performance = -holdings * (short_price - current_price)
            else:  # Keine Position
                performance = 0
            
            asset_performance_history[asset].append(performance)
        
        buy_hold_net_worth = sum(
            buy_hold_holdings[asset] * denormalize_price(test_data[asset].iloc[step], asset)
            for asset in traded_assets
        )
        buy_hold_net_worth_history.append(buy_hold_net_worth)

    # Plotten der Ergebnisse
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    if not silent:
        # Net Worth Plot
        ax1.plot(ai_net_worth_history, label='AI Model')
        ax1.plot(buy_hold_net_worth_history, label='Buy and Hold')
        ax1.set_ylabel('Net Worth ($)')
        ax1.set_title('AI Model vs Buy and Hold Strategy')
        ax1.legend()

        # Asset Performance Plot
        for asset in traded_assets:
            ax2.plot(asset_performance_history[asset], label=asset)
        ax2.set_ylabel('Asset Performance ($)')
        ax2.set_title('Asset Performance Over Time')
        ax2.legend()

        # Cash Balance Plot
        ax3.plot(cash_balance_history, label='Cash Balance')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Cash Balance ($)')
        ax3.set_title('Cash Balance Over Time')
        ax3.legend()

        # Net Worth Plot
        ax1.plot(ai_net_worth_history, label='AI Model')
        ax1.plot(buy_hold_net_worth_history, label='Buy and Hold')
        ax1.plot(random_net_worth_history, label='Random Strategy')
        ax1.set_ylabel('Net Worth ($)')
        ax1.set_title('Strategy Comparison')
        ax1.legend()
        

        plt.tight_layout()
        plt.show()

    # Berechnen der Gesamtrendite
    ai_total_return = (ai_net_worth_history[-1] - 10000) / 10000 * 100
    buy_hold_total_return = (buy_hold_net_worth_history[-1] - 10000) / 10000 * 100

    # Berechnen der Gesamtrendite für Random-Strategie
    random_total_return = (random_net_worth_history[-1] - 10000) / 10000 * 100
    
    print(f"AI Model Total Return on test data: {ai_total_return:.2f}%")
    print(f"Buy and Hold Total Return on test data: {buy_hold_total_return:.2f}%")
    print(f"Random Strategy Total Return on test data: {random_total_return:.2f}%")
    # Am Ende des Tests die Anzahl der Aktionen pro Asset ausgeben
    print("\nAnzahl der Aktionen pro Asset:")
    for asset in traded_assets:
        print(f"{asset}:")
        print(f"  Hold: {test_env.envs[0].action_counts_asset[asset]['hold']}")
        print(f"  Long: {test_env.envs[0].action_counts_asset[asset]['long']}")
        print(f"  Short: {test_env.envs[0].action_counts_asset[asset]['short']}")
    
    #return ai_total_return, buy_hold_total_return, random_total_return
    return ai_total_return, buy_hold_total_return, random_total_return, test_env.envs[0].action_counts_asset

def optimize_model(train_data, traded_assets, n_trials=50):
    def objective(trial):
        # Parameter-Raum definieren
        policy_kwargs = dict(
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=trial.suggest_int('features_dim', 128, 1024, step=128)
            ),
            net_arch=dict(
                pi=[
                    trial.suggest_int('pi_1', 128, 512),
                    trial.suggest_int('pi_2', 128, 512),
                    trial.suggest_int('pi_3', 64, 256)
                ],
                vf=[
                    trial.suggest_int('vf_1', 128, 512),
                    trial.suggest_int('vf_2', 128, 512),
                    trial.suggest_int('vf_3', 64, 256)
                ]
            ),
            activation_fn=nn.ReLU
        )

        # Trainingsumgebung erstellen
        train_env = DummyVecEnv([lambda: TradingEnv(train_data, traded_assets, is_training=True)])

        # PPO-Modell mit Trial-Parametern
        model = PPO(
            "MlpPolicy", 
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            n_steps=trial.suggest_int('n_steps', 1024, 8192, step=1024),
            batch_size=trial.suggest_int('batch_size', 64, 2048, step=64),
            n_epochs=trial.suggest_int('n_epochs', 1, 30),
            gamma=trial.suggest_float('gamma', 0.9, 0.9999),
            gae_lambda=trial.suggest_float('gae_lambda', 0.9, 0.999),
            clip_range=trial.suggest_float('clip_range', 0.1, 0.4),
            ent_coef=trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
            verbose=0,
            device=device
        )

        try:
            # Kurzes Training für Evaluation
            model.learn(total_timesteps=50000)
            
            # Evaluation durchführen
            test_returns = []
            for _ in range(3):  # 3 Evaluierungsdurchläufe für Stabilität
                ai_return, _, _, _ = load_and_test_model(
                    test_data, 
                    traded_assets, 
                    None,  # Kein Modell-Pfad nötig
                    silent=True,
                    model=model  # Direktes Modell-Objekt übergeben
                )
                test_returns.append(ai_return)
            
            mean_return = np.mean(test_returns)
            
            # Speichere zusätzliche Metriken
            trial.set_user_attr('std_return', np.std(test_returns))
            trial.set_user_attr('max_return', max(test_returns))
            
            return mean_return
            
        except Exception as e:
            print(f"Trial fehlgeschlagen: {e}")
            return float('-inf')

    # Optimierungsstudie erstellen
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimierung durchführen
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Beste Parameter ausgeben
    print("\nBeste gefundene Parameter:")
    print(study.best_params)
    print(f"\nBeste Performance: {study.best_value:.2f}%")

    # Visualisierungen erstellen
    try:
        import plotly
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.show()
    except Exception as e:
        print(f"Visualisierung fehlgeschlagen: {e}")

    return study.best_params

# Hauptausführungsteil
if __name__ == "__main__":
    # Optuna Hyperparamter Optimierung durchführen
    #best_params = optimize_model(train_data, traded_assets, n_trials=100) #50

    model_path = "ppo_trading_model11.zip"
    # train model
    num_episodes = 400 #200
    model = train_and_save_model(train_data, traded_assets, model_path, num_episodes)
    
    # Mehrfache Tests durchführen
    num_tests = 100 #100
    ai_returns = []
    bh_returns = []
    random_returns = []
    
    total_action_counts = {asset: {'hold': 0, 'long': 0, 'short': 0} for asset in traded_assets}
    
    print(f"Führe {num_tests} Tests durch...")
    for i in range(num_tests):
        if i % 10 == 0:
            print(f"Test {i+1}/{num_tests}")
        ai_return, bh_return, random_return, episode_action_counts = load_and_test_model(
            test_data, traded_assets, model_path, silent=True)
        
        # Summiere die Aktionen des aktuellen Tests
        for asset in traded_assets:
            for action_type in ['hold', 'long', 'short']:
                total_action_counts[asset][action_type] += episode_action_counts[asset][action_type]
        
        ai_returns.append(ai_return)
        bh_returns.append(bh_return)
        random_returns.append(random_return)

    # Statistiken berechnen
    ai_mean = np.mean(ai_returns)
    ai_std = np.std(ai_returns)
    bh_mean = np.mean(bh_returns)
    bh_std = np.std(bh_returns)
    random_mean = np.mean(random_returns)
    random_std = np.std(random_returns)

    print("\nTeststatistiken nach", num_tests, "Durchläufen:")
    print(f"AI Model:")
    print(f"  Durchschnittliche Rendite: {ai_mean:.2f}%")
    print(f"  Standardabweichung: {ai_std:.2f}%")
    print(f"  Min: {min(ai_returns):.2f}%")
    print(f"  Max: {max(ai_returns):.2f}%")
    print(f"\nBuy and Hold:")
    print(f"  Durchschnittliche Rendite: {bh_mean:.2f}%")
    print(f"  Standardabweichung: {bh_std:.2f}%")
    print(f"  Min: {min(bh_returns):.2f}%")
    print(f"  Max: {max(bh_returns):.2f}%")
    print(f"\nRandom Strategy:")
    print(f"  Durchschnittliche Rendite: {random_mean:.2f}%")
    print(f"  Standardabweichung: {random_std:.2f}%")
    print(f"  Min: {min(random_returns):.2f}%")
    print(f"  Max: {max(random_returns):.2f}%")

    # Ausgabe der Gesamtaktionen über alle Tests
    print("\nGesamtanzahl der Aktionen über alle Tests:")
    for asset in traded_assets:
        print(f"\n{asset}:")
        total_actions = sum(total_action_counts[asset].values())
        if total_actions > 0:
            print(f"  Hold: {total_action_counts[asset]['hold']} ({total_action_counts[asset]['hold']/total_actions*100:.1f}%)")
            print(f"  Long: {total_action_counts[asset]['long']} ({total_action_counts[asset]['long']/total_actions*100:.1f}%)")
            print(f"  Short: {total_action_counts[asset]['short']} ({total_action_counts[asset]['short']/total_actions*100:.1f}%)")
        else:
            print("  Keine Aktionen ausgeführt")
        print(f"  Gesamt: {total_actions}")