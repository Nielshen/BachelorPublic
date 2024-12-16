import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import talib
import yfinance as yf
from fredapi import Fred
from binance.client import Client
from pycoingecko import CoinGeckoAPI
import quandl
from datetime import datetime, timedelta
import time
import logging
import requests
import joblib

import sys
import os
sys.path.append('/Users/niels/Htwg/Bachelorarbeit/BachelorarbeitLocal')
from data.sentiment.sentimentNYT import get_sentiment_data
#from data.sentiment.gdelt import fetch_gdelt_articles, aggregate_sentiment  # Neuer Import

# Logging einrichten
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: API-Schlüssel ersetzen
FRED_API_KEY = 'IHR_FRED_API_KEY'

# API-Clients initialisieren
fred = Fred(api_key=FRED_API_KEY)
cg = CoinGeckoAPI()
#quandl.ApiConfig.api_key = QUANDL_API_KEY

def get_stock_data(symbols, start_date, end_date):
    try:
        data = yf.download(symbols, start=start_date, end=end_date)['Close']
        data.index = data.index.tz_localize(None)  # Entfernt die Zeitzone
        return data
    except Exception as e:
        logging.error(f"Fehler beim Abrufen von Aktiendaten: {e}")
        return pd.DataFrame()

    
def get_economic_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        logging.error(f"Fehler beim Abrufen von Wirtschaftsdaten für {symbol}: {e}")
        return pd.Series()

binance_client = Client()

# Funktion zum Abrufen von Kryptodaten von Binance
def get_crypto_data_binance(symbol, start_date, end_date):
    try:
        klines = binance_client.get_historical_klines(
            symbol, Client.KLINE_INTERVAL_1DAY, 
            start_date.strftime("%d %b %Y %H:%M:%S"), 
            end_date.strftime("%d %b %Y %H:%M:%S")
        )
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df['close'].astype(float)
    except Exception as e:
        logging.error(f"Fehler beim Abrufen von Krypto-Daten für {symbol}: {e}")
        return pd.Series()

def get_commodity_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)['Close']
        return data
    except Exception as e:
        logging.error(f"Fehler beim Abrufen von Rohstoffdaten für {symbol}: {e}")
        return pd.Series()

def get_global_market_cap():
    try:
        url = "https://api.worldbank.org/v2/indicator/CM.MKT.LCAP.CD?format=json"
        response = requests.get(url)
        data = response.json()[1]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%Y')
        df['value'] = pd.to_numeric(df['value'])
        df.set_index('date', inplace=True)
        return df['value']
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der globalen Marktkapitalisierung: {e}")
        return pd.Series()

def get_global_crypto_market_cap(start_date, end_date):
    url = "https://api.coingecko.com/api/v3/global"
    data = []
    current_date = start_date
    max_retries = 5
    retry_delay = 2.5

    while current_date <= end_date:
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    market_data = response.json()['data']
                    data.append({
                        'date': current_date,
                        'total_market_cap': market_data['total_market_cap']['usd']
                    })
                    break
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponentielles Backoff
                    else:
                        logging.warning(f"Max Versuche erreicht für {current_date}")
                else:
                    logging.warning(f"Fehler beim Abrufen der Daten für {current_date}: {response.status_code}")
                    break
            except RequestException as e:
                logging.error(f"Netzwerkfehler: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logging.warning(f"Max Versuche erreicht für {current_date}")
        
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df['total_market_cap']
    
# Funktion zum Abrufen von Weltbank-Daten mit täglicher Interpolation
def get_world_bank_data(indicator, country=None, start_date=None, end_date=None):
    try:
        base_url = "https://api.worldbank.org/v2/country/"
        country_code = country if country else "all"
        url = f"{base_url}{country_code}/indicator/{indicator}?format=json&per_page=5000"
        
        if start_date and end_date:
            url += f"&date={start_date.year}:{end_date.year}"
        
        response = requests.get(url)
        data = response.json()
        
        if len(data) < 2 or not data[1]:
            logging.warning(f"Keine Daten verfügbar für Indikator {indicator}")
            return pd.Series(dtype=float)
        
        df = pd.DataFrame(data[1])
        logging.info(f"Erhaltene Spalten für {indicator}: {df.columns}")
        
        if 'date' not in df.columns or 'value' not in df.columns:
            logging.warning(f"Unerwartetes Datenformat für Indikator {indicator}. Verfügbare Spalten: {df.columns}")
            return pd.Series(dtype=float)
        
        df['date'] = pd.to_datetime(df['date'], format='%Y')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Gruppieren nach Datum und den Mittelwert nehmen, um doppelte Indizes zu vermeiden
        df = df.groupby('date')['value'].mean().reset_index()
        
        df.set_index('date', inplace=True)
        df = df.sort_index()
        
        logging.info(f"Datenbereich für {indicator}: {df.index.min()} bis {df.index.max()}")
        
        # Überprüfen, ob es sich um jährliche oder tägliche Daten handelt
        if df.index.freqstr == 'A-DEC' or (df.index[-1] - df.index[0]).days > 365:
            # Jährliche Daten auf tägliche Daten interpolieren
            daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
            daily_series = df['value'].reindex(daily_index)
            daily_series = daily_series.interpolate(method='time', limit_direction='both')
        else:
            daily_series = df['value']
        
        return daily_series
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Weltbank-Daten für {indicator}: {e}")
        return pd.Series(dtype=float)

def collect_data(start_date, end_date):
    data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='D'))
    
    #asset_columns = [col for col in data.columns if any(prefix in col for prefix in ['Stock_', 'ETF_', 'Crypto_', 'Bond_', 'RE_', 'Commodity_', 'Forex_'])]

    # Sentiment-Daten abrufen
    logging.info("Rufe Sentiment-Daten ab...")
    search_terms = ["stock market", "bond market", "commodity market", "real estate market", "crypto market", "bitcoin"]
    sentiment_data = get_sentiment_data(start_date, end_date, search_terms)
    data = pd.concat([data, sentiment_data], axis=1)
    


    # Globale Indikatoren für Anlageklassen
    logging.info("Sammle globale Indikatoren für Anlageklassen...")
    
    # Aktien
    print("Get StockMarket Cap")
    data['Global_Stock_Market_Cap'] = get_world_bank_data('CM.MKT.LCAP.CD', start_date=start_date, end_date=end_date)
    #data['Global_Stock_Market_Index'] = get_stock_data('URTH', start_date, end_date)  # MSCI World Index ETF
    data['Global_Stock_Trading_Value'] = get_world_bank_data('CM.MKT.TRAD.GD.ZS', start_date=start_date, end_date=end_date)
    data['Global_Stock_Turnover_Ratio'] = get_world_bank_data('CM.MKT.TRNR', start_date=start_date, end_date=end_date)

    # Anleihen
    #print("Get Bond Market Cap")
   #data['Global_Bond_Market_Index'] = get_stock_data('AGG', start_date, end_date)  # iShares Core U.S. Aggregate Bond ETF
    
    # Kryptowährungen global market cap TODO remove for test
    #print("Get Krypto Market Cap")
    #data['Global_Crypto_Market_Cap'] = get_global_crypto_market_cap(start_date, end_date)
    
    # Immobilien
    #print("Get Real Estate Data")
    #data['Global_Real_Estate_Index'] = get_stock_data('RWO', start_date, end_date)  # SPDR Dow Jones Global Real Estate ETF
    
    # Rohstoffe
    print("Get raw materials Data")
    data['Global_Commodity_Index'] = get_stock_data('DBC', start_date, end_date)  # Invesco DB Commodity Index Tracking Fund
    data['Global_Commodity_Agricultural_Index'] = get_stock_data('DBA', start_date, end_date)  # Invesco DB Agriculture Fund
    data['Global_Commodity_Energy_Index'] = get_stock_data('DBE', start_date, end_date)  # Invesco DB Energy Fund
    data['Global_Commodity_Metals_Index'] = get_stock_data('DBB', start_date, end_date)  # Invesco DB Base Metals Fund
    
    # Forex
    print("Get Forex Data")
    data['Dollar_Index'] = get_stock_data('UUP', start_date, end_date)  # Invesco DB US Dollar Index Bullish Fund

    #USA M2 Geldmenge
    print("Get M2 Money Data")
    data['US_M2_Money_Supply'] = get_world_bank_data('FM.LBL.BMNY.GD.ZS', country='US', start_date=start_date, end_date=end_date)

    # Volatilitätsindizes
    data['VSTOXX'] = get_stock_data('^STOXX', start_date, end_date)

    # Yield-Kurven
    data['US_10Y_Treasury_Yield'] = get_economic_data('^TNX', start_date, end_date)
    #data['DE_10Y_Bund_Yield'] = get_economic_data('GTDEM10Y:GOV', start_date, end_date)

    # VIX Index, Angst Index - erwartete Volatilität
    data['VIX'] = get_stock_data('^VIX', start_date, end_date)

    # Sektorspezifische Indizes
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Commodities': 'DBC'
    }
    for sector, symbol in sector_etfs.items():
        data[f'{sector}_Sector'] = get_stock_data(symbol, start_date, end_date)

    # Makroökonomische Indikatoren
    #data['US_Manufacturing_PMI'] = get_economic_data('MPMUGDMA', start_date, end_date)
    #data['US_Capacity_Utilization'] = get_economic_data('TCU', start_date, end_date)

    # Sentiment-Indikatoren
    #data['US_Economic_Policy_Uncertainty'] = get_economic_data('USEPUINDXD', start_date, end_date)
    
    # Globale Indikatoren
    global_indicators = {
        'Global_GDP': 'NY.GDP.MKTP.CD',
        'Global_GDP_Growth': 'NY.GDP.MKTP.KD.ZG',
        'Global_Inflation': 'FP.CPI.TOTL.ZG',
        'Global_Trade': 'NE.TRD.GNFS.ZS',
        'Global_FDI': 'BX.KLT.DINV.WD.GD.ZS',
        'Global_Population': 'SP.POP.TOTL',
        'Global_Unemployment': 'SL.UEM.TOTL.ZS',
        'Global_Manufacturing': 'NV.IND.MANF.ZS',
        'Global_High_Tech_Exports': 'TX.VAL.TECH.MF.ZS',
        'Global_Research_Development': 'GB.XPD.RSDV.GD.ZS'
    }

    logging.info("Sammle globale makroökonomische Indikatoren...")
    for name, indicator in global_indicators.items():
        data[name] = get_world_bank_data(indicator, start_date=start_date, end_date=end_date)

    # Länderspezifische Indikatoren
    countries = ['US', 'CN', 'JP', 'DE', 'GB', 'IN', 'FR', 'IT', 'BR', 'CA']
    country_indicators = {
        'GDP_Growth': 'NY.GDP.MKTP.KD.ZG',
        'Inflation_Rate': 'FP.CPI.TOTL.ZG',
        'Unemployment_Rate': 'SL.UEM.TOTL.ZS',
        'Government_Debt_To_GDP': 'GC.DOD.TOTL.GD.ZS',
        'Current_Account_Balance': 'BN.CAB.XOKA.GD.ZS',
        'Interest_Rate': 'FR.INR.RINR',
        'Industrial_Production': 'NV.IND.TOTL.ZS',
        'Retail_Sales': 'NE.CON.PRVT.ZS',
        'Consumer_Confidence': 'BI.CON.CONF.ZS'
    }

    logging.info("Sammle länderspezifische Indikatoren...")
    for country in countries:
        for indicator_name, indicator_code in country_indicators.items():
            column_name = f"{country}_{indicator_name}"
            data[column_name] = get_world_bank_data(indicator_code, country=country, start_date=start_date, end_date=end_date)
 

    #Ab hier Potenzeielle invests!
    # Einzelne Aktien
    stocks = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',  # Tech (US)
        'JPM', 'BAC', 'V', 'MA', 'PYPL',  # Finanzen (US)
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',  # Gesundheit (US)
        'XOM', 'CVX', 'SHEL', 'BP', 'TTE',  # Energie (Global)
        'WMT', 'PG', 'KO', 'PEP', 'COST',  # Konsumgüter (US)
        'TSM', 'BABA', 'TCEHY', '005930.KS', '7203.T',  # Asien, TSMC!
        'ASML', 'SAP', 'SIEGY', 'NSRGY', 'NVO'  # Europa
    ]
    # Einzelne Aktien
    logging.info("Sammle Daten für einzelne Aktien...")
    stocks_data = get_stock_data(stocks, start_date, end_date)
    
    # Zeitzonenkonflikt beheben
    stocks_data.index = stocks_data.index.tz_localize(None)
    
    # Präfix 'Stock_' hinzufügen
    stocks_data.columns = ['Stock_' + col for col in stocks_data.columns]
    
    data = pd.concat([data, stocks_data], axis=1)

     # Anleihen/Bonds-ETFs
    logging.info("Sammle Daten für Bond/Anleihen-ETFs...")
    bond_etfs = {
        'iShares_Core_US_Aggregate': 'AGG',  # iShares Core U.S. Aggregate Bond ETF
        'Vanguard_Total_Bond_Market': 'BND',  # Vanguard Total Bond Market ETF
        'iShares_iBoxx_Investment_Grade': 'LQD',  # iShares iBoxx $ Investment Grade Corporate Bond ETF
        'Vanguard_Short_Term_Bond': 'BSV',  # Vanguard Short-Term Bond ETF
        'iShares_TIPS_Bond': 'TIP',  # iShares TIPS Bond ETF
        'iShares_Core_EUR_Govt_Bond': 'EUNH.DE'  # Größter deutscher Anleihen-ETF
    }
    for name, symbol in bond_etfs.items():
        data[f'Bond_{name}'] = get_stock_data(symbol, start_date, end_date)

    # Rohstoffe F bedeutet Futures
    commodities = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Platinum': 'PL=F',
        'Palladium': 'PA=F',
        'Copper': 'HG=F',  # Kupfer
        'Zinc': 'ZN=F',    # Zink
        'Uranium': 'URA',  # Global X Uranium ETF als Proxy

        'Crude_Oil_WTI': 'CL=F', # Öl
        'Natural_Gas': 'NG=F',

        'Wheat': 'ZW=F',  # Weizen
        'Coffee': 'KC=F',  # Kaffee-
        'Sugar': 'SB=F',   # Zucker
        'Cocoa': 'CC=F' 
    }
    logging.info("Sammle Rohstoffdaten...")
    for name, symbol in commodities.items():
        data['Commodity_' + name] = get_commodity_data(symbol, start_date, end_date)


    # Forex-Paare
    forex_pairs = ['EURUSD=X', 'JPYUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'CADUSD=X', 'CHFUSD=X', 'NZDUSD=X', 'CNYUSD=X', 'HKDUSD=X', 'SGDUSD=X']
    logging.info("Sammle Forex-Daten...")
    forex_data = get_stock_data(forex_pairs, start_date, end_date)
    forex_data.index = forex_data.index.tz_localize(None)  # Entfernt die Zeitzone
    forex_data.columns = ['Forex_' + col for col in forex_data.columns]
    data = pd.concat([data, forex_data], axis=1)

    cryptos = {
        'BTCUSDT': '2009-01-03',  # Bitcoin Genesis Block
        'ETHUSDT': '2015-07-30',  # Ethereum Launch
        'ADAUSDT': '2017-09-29',  # Cardano Launch
        'SOLUSDT': '2020-04-10',  # Solana Launch
        'DOTUSDT': '2020-08-18',  # Polkadot Launch
        'BNBUSDT': '2017-07-14',  # Binance Coin Launch
        'XRPUSDT': '2013-01-01',  # Ripple Launch (approximation)
        'TRXUSDT': '2017-09-13',  # TRON Launch
        'AVAXUSDT': '2020-09-21', # Avalanche Launch
        'LINKUSDT': '2017-09-19', # Chainlink Launch
        'NEARUSDT': '2020-04-22', # NEAR Protocol Launch
        'RNDRUSDT': '2021-02-17', # Render Token Launch on Binance
        'TAOUSDT': '2023-03-23',  # TAO Launch (approximation)
        'VETUSDT': '2018-07-13',  # VeChain Launch
        'HBARUSDT': '2019-09-17', # Hedera Launch
        'IMXUSDT': '2021-08-11',  # ImmutableX Launch
        'MATICUSDT': '2019-04-29' # Polygon MATIC Launch
    }
    
    logging.info("Sammle Kryptowährungsdaten von Binance...")
    for crypto, launch_date in cryptos.items():
        launch_datetime = pd.to_datetime(launch_date)
        data_start_date = max(start_date, launch_datetime)
        
        crypto_data = get_crypto_data_binance(crypto, data_start_date, end_date)
        
        # Initialisiere die Spalte mit 0
        data[f'Crypto_{crypto}'] = 0
        
        # Fülle die Daten ab dem Einführungsdatum ein
        data.loc[data.index >= launch_datetime, f'Crypto_{crypto}'] = crypto_data
        
        time.sleep(1)  # Verzögerung, um API-Limits zu respektieren

    top_etfs = {
        'SPY': 'SPY',     # SPDR S&P 500 ETF Trust
        'VOO': 'VOO',     # Vanguard S&P 500 ETF
        'VTI': 'VTI',     # Vanguard Total Stock Market ETF
        'QQQ': 'QQQ',     # Invesco QQQ Trust (NASDAQ-100 Index)
        'IVV': 'IVV',     # iShares Core S&P 500 ETF
        'VEA': 'VEA',     # Vanguard FTSE Developed Markets ETF
        'IEFA': 'IEFA',   # iShares Core MSCI EAFE ETF
        'AGG': 'AGG',     # iShares Core U.S. Aggregate Bond ETF
        'VWO': 'VWO',     # Vanguard FTSE Emerging Markets ETF
        'BND': 'BND',     # Vanguard Total Bond Market ETF
        'DAX': '^GDAXI',  # Deutscher Aktienindex
        'S&P_500': '^GSPC', # S&P 500 Index
        'Dow_Jones': '^DJI', # Dow Jones Industrial Average
        'Russell_2000': '^RUT', # Russell 2000 Index
        'NASDAQ': '^IXIC', # NASDAQ Composite Index
        'Hang_Seng': '^HSI' # Hang Seng Index
    }
    logging.info("Sammle Daten für Top ETFs und Indizes...")
    for name, symbol in top_etfs.items():
        data[f'ETF_{name}'] = get_stock_data(symbol, start_date, end_date)

    logging.info("Sammle Daten für Immobilien-ETFs...")
    real_estate_etfs = {
        'iShares_European_Property': 'IPRP.L',  # iShares European Property Yield UCITS ETF
        'Vanguard_Real_Estate': 'VNQ',  # Vanguard Real Estate ETF
        'iShares_Global_REIT': 'REET',  # iShares Global REIT ETF
        'SPDR_Dow_Jones_Global_Real_Estate': 'RWO',  # SPDR Dow Jones Global Real Estate ETF
        'Schwab_US_REIT': 'SCHH',  # Schwab U.S. REIT ETF
    }
    for name, symbol in real_estate_etfs.items():
        data[f'RE_{name}'] = get_stock_data(symbol, start_date, end_date)

    # Daten bereinigen und interpolieren
    data = data.fillna(method='ffill').fillna(method='bfill')
    data = data.interpolate(method='time')

    # Zeitzonenkonflikt für alle Spalten beheben
    for column in data.columns:
        if isinstance(data[column].dtype, pd.DatetimeTZDtype):
            data[column] = data[column].dt.tz_localize(None)
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = pd.to_datetime(data[column]).dt.tz_localize(None)

    asset_columns = [col for col in data.columns if any(prefix in col for prefix in ['Stock_', 'ETF_', 'Crypto_', 'Bond_', 'RE_', 'Commodity_', 'Forex_'])]
    
    return data, asset_columns

def prepare_data_for_rl(data, asset_columns):
    print(f"Ursprüngliche Daten: Zeilen = {len(data)}, Spalten = {len(data.columns)}")

    # Entferne Spalten mit zu vielen fehlenden Werten
    threshold = len(data) * 0.5  # 50% der Daten müssen vorhanden sein
    data = data.dropna(axis=1, thresh=threshold)
    
    print(f"Nach Entfernen von Spalten mit vielen fehlenden Werten: Zeilen = {len(data)}, Spalten = {len(data.columns)}")
    
    # Fülle verbleibende NaN-Werte
    data = data.fillna(method='ffill').fillna(method='bfill')
    

    # Überprüfe, ob noch NaN-Werte vorhanden sind
    nan_count = data.isna().sum().sum()
    print(f"Verbleibende NaN-Werte: {nan_count}")

    # Aktualisiere asset_columns, falls einige entfernt wurden
    asset_columns = [col for col in asset_columns if col in data.columns]


    # Erstelle separate DataFrames für verschiedene Kategorien
    assets_df = data[asset_columns]
    tech_indicators_df = pd.DataFrame()
    macro_indicators_df = data[[col for col in data.columns if col not in asset_columns and not col.endswith('_sentiment_hf_financial')]]
    sentiment_df = data[[col for col in data.columns if col.endswith('_sentiment_hf_financial')]]

    print(f"Anzahl der Zeilen vor der Verarbeitung: {len(data)}")
    print(f"Anzahl der Spalten vor der Verarbeitung: {len(data.columns)}")

    # Normalisiere die Daten mit MinMaxScaler
    # Erstelle separate Scaler für jede Kategorie
    # Separate Skalierung für jede Anlageklasse

    #def log_transform(data):
        #return np.log1p(data)

    #for asset_type in ['Stock', 'ETF', 'Crypto', 'Commodity', 'Bond', 'RE', 'Forex']: #['Stock_', 'ETF_', 'Crypto_', 'Bond_', 'RE_', 'Commodity_', 'Forex_']
        #cols = [col for col in asset_columns if col.startswith(asset_type)]
        #assets_df[cols] = assets_df[cols].apply(log_transform)

    # Normalisiere die anderen Daten
    macro_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    tech_scaler = MinMaxScaler()

    macro_indicators_df = pd.DataFrame(macro_scaler.fit_transform(macro_indicators_df), columns=macro_indicators_df.columns, index=macro_indicators_df.index)
    sentiment_df = pd.DataFrame(sentiment_scaler.fit_transform(sentiment_df), columns=sentiment_df.columns, index=sentiment_df.index)

    # Berechne technische Indikatoren für Assets
    for column in asset_columns:
        tech_indicators_df[f'{column}_MA_30'] = talib.SMA(assets_df[column], timeperiod=30)
        tech_indicators_df[f'{column}_MA_60'] = talib.SMA(assets_df[column], timeperiod=60)
        tech_indicators_df[f'{column}_RSI'] = talib.RSI(assets_df[column], timeperiod=14)
        macd, signal, _ = talib.MACD(assets_df[column], fastperiod=12, slowperiod=26, signalperiod=9)
        tech_indicators_df[f'{column}_MACD'] = macd
        tech_indicators_df[f'{column}_MACD_Signal'] = signal

    tech_indicators_df = pd.DataFrame(tech_scaler.fit_transform(tech_indicators_df), columns=tech_indicators_df.columns, index=tech_indicators_df.index)

    # Erstelle das scalers Dictionary
    scalers = {
        'macro': macro_scaler,
        'sentiment': sentiment_scaler,
        'tech': tech_scaler
    }

    # Speicher die Scaler
    for scaler_name, scaler in scalers.items():
        joblib.dump(scaler, f'../files/{scaler_name}_scaler.pkl')

    time_features_df = pd.DataFrame({
        'day_of_week': assets_df.index.dayofweek,
        'month': assets_df.index.month,
        'year': assets_df.index.year,
        'is_quarter_end': assets_df.index.is_quarter_end.astype(int),
        'is_month_end': assets_df.index.is_month_end.astype(int),
        'is_year_end': assets_df.index.is_year_end.astype(int)
    }, index=assets_df.index)

    # Erstelle Lag-Features für Assets
    lag_features = {}
    for column in asset_columns:
        for lag in [1, 3, 7]:
            lag_features[f'{column}_lag_{lag}'] = assets_df[column].shift(lag)
    lag_features_df = pd.DataFrame(lag_features)

    # Kombiniere alle DataFrames
    combined_df = pd.concat([assets_df, tech_indicators_df, macro_indicators_df, time_features_df, lag_features_df, sentiment_df], axis=1) #, sentiment_df
    
    print(f"Anzahl der Zeilen vor dem Entfernen von NaN-Werten: {len(combined_df)}")
    print(f"Anzahl der Spalten vor dem Entfernen von NaN-Werten: {len(combined_df.columns)}")
    print(f"Anzahl der NaN-Werte: {combined_df.isna().sum().sum()}")

    # Entferne die ersten 60 Zeilen, da NaN-Werte für einige Indikatoren enthalten können
    # Entferne die ersten 60 Zeilen, da NaN-Werte für einige Indikatoren enthalten können
    combined_df = combined_df.iloc[60:]
    
    # Entferne Zeilen mit NaN-Werten
    combined_df = combined_df.dropna()
    
    print(f"Anzahl der Zeilen nach dem Entfernen von NaN-Werten: {len(combined_df)}")   
    print(f"Anzahl der Spalten nach dem Entfernen von NaN-Werten: {len(combined_df.columns)}")
    
    return combined_df, {'macro': macro_scaler, 'sentiment': sentiment_scaler, 'tech': tech_scaler}




if __name__ == "__main__":
    start_date = datetime(2000, 1, 1) #datetime(2000, 1, 1)
    end_date = datetime.now()

    logging.info("Starte Datensammlung...")
    financial_data, asset_columns = collect_data(start_date, end_date)

    print(f"Gesammelte Daten: Zeilen = {len(financial_data)}, Spalten = {len(financial_data.columns)}")
    print(f"Asset Spalten: {asset_columns}")

    logging.info("Bereite Daten für Reinforcement Learning vor...")
    rl_data, scalers = prepare_data_for_rl(financial_data, asset_columns)
    
    logging.info("Speichere vorbereitete Daten...")
    rl_data.to_csv('../files/multi_data_indicators.csv')
    rl_data.to_pickle('../files/multi_data_indicators.pkl')

    logging.info("Datenverarbeitung abgeschlossen. Daten wurden in 'multi_data_indicators.csv' und 'multi_data_indicators.pkl' gespeichert.")
    print(f"Die vorbereitete Tabelle hat {rl_data.shape[1]} Spalten und {rl_data.shape[0]} Zeilen.")