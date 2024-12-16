import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Einige Funktionen aus dem Buch "Advances in Financial Machine Learning"

def getDailyVol(close, span0=100):
    # Berechnet die tägliche Volatilität, reindiziert auf Close-Preise
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # tägliche Renditen
    df0 = df0.ewm(span=span0).std()
    return df0

def getVerticalBarriers(close, tEvents, numDays):
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])  # NaNs am Ende
    return t1

def applyPtSlOnT1(close, events, ptSl, molecule):
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # Preisverlauf
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # Renditeverlauf
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # frühester Stop Loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # frühestes Take Profit
    return out

def getEvents(close, tEvents, ptSl, trgt, minRet, t1=False, side=None):
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    if side is None:
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = applyPtSlOnT1(close, events, ptSl_, events.index)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    events['pt'] = ptSl[0]
    events['sl'] = ptSl[1]
    return events

def barrier_touched(out_df, events):
    store = []
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']
        pt_level_reached = ret > target * events.loc[date_time, 'pt']
        sl_level_reached = ret < -target * events.loc[date_time, 'sl']
        if ret > 0.0 and pt_level_reached:
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            store.append(-1)
        else:
            store.append(0)
    out_df['bin'] = store
    return out_df

def getBins(events, close):
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # Meta-Labeling
    out['trgt'] = events_['trgt']
    out = barrier_touched(out, events)
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        out['side'] = events['side']
    return out

def dropLabels(events, minPct=0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print(f"Gelöschtes Label {df0.idxmin()} mit Anteil {df0.min():.2%}")
        events = events[events['bin'] != df0.idxmin()]
    return events

def determine_trade_direction(close):
    # Technische Analyse MA Crossover Strategie
    slow_ma = close.rolling(100).mean()
    fast_ma = close.rolling(10).mean()
    side = pd.Series(0, index=close.index)
    side[fast_ma >= slow_ma] = 1  # Long
    side[fast_ma < slow_ma] = -1  # Short
    return side

def prepare_features(df, close):
    # Vorbereitung der Features für das Random Forest Modell
    features = pd.DataFrame(index=df.index)
    # Berechnung der logarithmischen Renditen
    features['log_ret'] = np.log(close).diff()
    # Volatilitätsmerkmale
    features['vol5'] = features['log_ret'].rolling(5).std()
    features['vol10'] = features['log_ret'].rolling(10).std()
    features['vol15'] = features['log_ret'].rolling(15).std()
    # Serielle Korrelationen
    features['serialcorr20-1'] = features['log_ret'].rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    features['serialcorr20-2'] = features['log_ret'].rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=2), raw=False)
    features['serialcorr20-3'] = features['log_ret'].rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=3), raw=False)

    features['side'] = determine_trade_direction(close)
    # Entfernen von NaN-Werten
    features = features.dropna()
    return features

def train_random_forest(features, labels):
    # Aufteilen der Daten in Trainings- und Testsets
    train_size = int(len(features) * 0.8)
    X_train, X_test = features[:train_size], features[train_size:]
    y_train, y_test = labels[:train_size], labels[train_size:]
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Speichern des Modells
    joblib.dump(rf_model, 'random_forest_model.joblib')
    print("Modell wurde gespeichert unter 'random_forest_model.joblib'")
    
    # Modellbewertung
    y_pred = rf_model.predict(X_test)
    print("\nBewertung des Random Forest Modells:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModellgenauigkeit: {accuracy:.4f}")
    return rf_model, X_test

def main():
    # Laden der Daten
    print("Lade Daten...")
    data = pd.read_pickle('multi_data_indicators.pkl')

    # Zeitraum filtern
    start_date = '2017-09-01'
    end_date = '2022-11-23'  # Anpassung des Enddatums
    data = data[start_date:end_date]

    # Train-Test-Split (angepasst)
    train_end_date = '2021-11-10'  # Ende des Trainingszeitraums
    train_data = data[:train_end_date]
    test_data = data[train_end_date:end_date]

    # Nur BTC-relevante Spalten auswählen
    relevant_columns = [col for col in data.columns if 'Crypto_BTCUSDT' in col or 'ETF_S&P_500' in col]
    dataEtf = data[relevant_columns]

    btc_columns = [col for col in data.columns if 'Crypto_BTCUSDT' in col]
    data = data[btc_columns]

    # Train-Test-Split für ETF-Daten
    train_dataEtf = dataEtf[:train_end_date]
    test_dataEtf = dataEtf[train_end_date:end_date]

    print("\nAnalyse der Datenzeiträume:")
    print(f"\nTrainingsdaten:")
    print(f"Start: {train_data.index[0]}")
    print(f"Ende: {train_data.index[-1]}")
    print(f"Länge: {len(train_data)} Datenpunkte")

    print(f"\nTestdaten:")
    print(f"Start: {test_data.index[0]}")
    print(f"Ende: {test_data.index[-1]}")
    print(f"Länge: {len(test_data)} Datenpunkte")

    # Triple Barrier Labeling auf Trainingsdaten
    close_train = train_data['Crypto_BTCUSDT']
    
    print("Berechne Volatilität...")
    volatility = getDailyVol(close_train)
    
    print("Sampling Ereignisse...")
    tEvents = close_train.index[close_train.diff() != 0]
    
    print("Bestimmen der vertikalen Barrieren...")
    numDays = 5  # Anpassbar: Zeitfenster für vertikale Barrieren
    t1 = getVerticalBarriers(close_train, tEvents, numDays)
    
    print("Bestimmen der Handelsrichtung...")
    side = determine_trade_direction(close_train)
    
    # Nur Ereignisse mit klarer Richtung
    tEvents = tEvents[side[tEvents] != 0]
    
    print("Bestimmen der Ereignisse...")
    events = getEvents(close_train, tEvents=tEvents, ptSl=[1, 1], t1=t1, trgt=volatility, minRet=0.05, side=side)
    
    print("Bestimmen der Bins...")
    bins = getBins(events, close_train)
    bins['side'] = events['side']
    final_events = bins
    
    print("Entferne Labels mit unzureichenden Beispielen...")
    final_events = dropLabels(bins, minPct=0.05)
    
    # Features und Labels für Training
    aligned_data = pd.concat([train_data, final_events['bin']], axis=1).dropna()
    X_train = aligned_data.drop('bin', axis=1)
    y_train = aligned_data['bin']

    # Training des Random Forest
    print("Trainiere Random Forest Modell...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Anpassbar: Anzahl der Bäume
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Testen mit 10000 Startkapital
    initial_balance = 10000
    current_balance = initial_balance
    current_btc = 0
    
    # Vorhersagen für Testdaten
    X_test = test_data
    test_predictions = rf_model.predict(X_test)
    
    # Portfolio Performance Tracking
    portfolio_values = []
    for i in range(len(test_predictions)):
        current_price = test_data['Crypto_BTCUSDT'].iloc[i]
        prediction = test_predictions[i]
        
        # Berechne aktuellen Portfolio-Wert
        portfolio_value = current_balance + (current_btc * current_price)
        portfolio_values.append(portfolio_value)
        
        # Trading Logik basierend auf Vorhersagen
        if prediction == 1 and current_balance > 0:  # Kaufsignal
            btc_to_buy = current_balance / current_price * 0.99  # 1% Transaktionskosten
            current_btc += btc_to_buy
            current_balance = 0
        elif prediction == -1 and current_btc > 0:  # Verkaufssignal
            current_balance = current_btc * current_price * 0.99  # 1% Transaktionskosten
            current_btc = 0

 # Performance Metriken für die ursprüngliche Strategie
    final_portfolio_value = portfolio_values[-1]
    total_return = ((final_portfolio_value - initial_balance) / initial_balance) * 100

    print("\nPortfolio Performance (ML-Strategie):")
    print(f"Startkapital: ${initial_balance:,.2f}")
    print(f"Endkapital: ${final_portfolio_value:,.2f}")
    print(f"Gesamtrendite: {total_return:.2f}%")

    # DCA-Strategie für BTC
    dca_balance = initial_balance
    dca_btc = 0
    monthly_investment = initial_balance / len(test_data) * 30  # Monatliche Investition

    for i in range(len(test_data)):
        current_price = test_data['Crypto_BTCUSDT'].iloc[i]
        
        # Investiere monatlich (ca. alle 30 Tage)
        if i % 30 == 0 and dca_balance >= monthly_investment:
            btc_to_buy = monthly_investment / current_price * 0.99  # 1% Transaktionskosten
            dca_btc += btc_to_buy
            dca_balance -= monthly_investment

    final_dca_value = dca_balance + (dca_btc * test_data['Crypto_BTCUSDT'].iloc[-1])
    dca_return = ((final_dca_value - initial_balance) / initial_balance) * 100

    print("\nDCA-Strategie Performance (BTC):")
    print(f"Startkapital: ${initial_balance:,.2f}")
    print(f"Endkapital: ${final_dca_value:,.2f}")
    print(f"Gesamtrendite: {dca_return:.2f}%")

    # Zufallsstrategie
    np.random.seed(42)
    random_balance = initial_balance
    random_btc = 0

    for i in range(len(test_data)):
        current_price = test_data['Crypto_BTCUSDT'].iloc[i]
        action = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4])
        
        if action == 'buy' and random_balance > 0:
            btc_to_buy = random_balance / current_price * 0.99 * np.random.random()  # Zufälliger Anteil des Vermögens
            random_btc += btc_to_buy
            random_balance -= btc_to_buy * current_price
        elif action == 'sell' and random_btc > 0:
            btc_to_sell = random_btc * np.random.random()  # Verkaufe zufälligen Anteil
            random_balance += btc_to_sell * current_price * 0.99
            random_btc -= btc_to_sell

    final_random_value = random_balance + (random_btc * test_data['Crypto_BTCUSDT'].iloc[-1])
    random_return = ((final_random_value - initial_balance) / initial_balance) * 100

    print("\nZufallsstrategie Performance (BTC):")
    print(f"Startkapital: ${initial_balance:,.2f}")
    print(f"Endkapital: ${final_random_value:,.2f}")
    print(f"Gesamtrendite: {random_return:.2f}%")

    # MSCI World Buy & Hold Strategie (S&P 500 als Proxy)
    msci_price_start = test_dataEtf['ETF_S&P_500'].iloc[0]  # S&P 500 als Proxy für MSCI World
    msci_price_end = test_dataEtf['ETF_S&P_500'].iloc[-1]
    msci_shares = (initial_balance / msci_price_start) * 0.99  # 1% Transaktionskosten
    final_msci_value = msci_shares * msci_price_end
    msci_return = ((final_msci_value - initial_balance) / initial_balance) * 100

    print("\nMSCI World Buy & Hold Performance:")
    print(f"Startkapital: ${initial_balance:,.2f}")
    print(f"Endkapital: ${final_msci_value:,.2f}")
    print(f"Gesamtrendite: {msci_return:.2f}%")

    # Vergleichende Zusammenfassung
    print("\nZusammenfassung aller Strategien:")
    print(f"ML-Strategie: {total_return:.2f}%")
    print(f"BTC DCA: {dca_return:.2f}%")
    print(f"Zufallsstrategie: {random_return:.2f}%")
    print(f"MSCI World B&H: {msci_return:.2f}%")

    # Buy & Hold Vergleich
    btc_start_price = test_data['Crypto_BTCUSDT'].iloc[0]
    btc_end_price = test_data['Crypto_BTCUSDT'].iloc[-1]
    bh_return = ((btc_end_price - btc_start_price) / btc_start_price) * 100
    print(f"\nBTC Buy & Hold Rendite: {bh_return:.2f}%")

if __name__ == "__main__":
    main()