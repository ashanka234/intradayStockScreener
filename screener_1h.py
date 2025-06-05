import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from tqdm import tqdm
from datetime import datetime,date
import os
import json
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from pytz import timezone
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe

IST = timezone('Asia/Kolkata')

# --- Load tickers ---
tickers_df = pd.read_csv("/home/ashank234/intradayStockOptionsScreener/FnO_stocks_list.csv")  # replace with your file
tickers = tickers_df['Symbol'].tolist()
tickers = [t + ".NS" for t in tickers]  # or ".BO" for BSE

# --- Screener Params ---
atr_period = 14
atr_multiplier = 3
zlema_period = 70
lag = (zlema_period - 1) // 2
batch_size = 50

results = []

# --- Helper Function: ATR Stop ---
def compute_atr_stop(df):
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
    df['nLoss'] = df['ATR'] * atr_multiplier

    stop = [np.nan] * len(df)
    for i in range(1, len(df)):
        prev = stop[i - 1]
        if np.isnan(prev):
            stop[i] = df['Close'].iloc[i] - df['nLoss'].iloc[i]
        elif df['Close'].iloc[i - 1] > prev and df['Close'].iloc[i] > prev:
            stop[i] = max(prev, df['Close'].iloc[i] - df['nLoss'].iloc[i])
        elif df['Close'].iloc[i - 1] < prev and df['Close'].iloc[i] < prev:
            stop[i] = min(prev, df['Close'].iloc[i] + df['nLoss'].iloc[i])
        else:
            stop[i] = df['Close'].iloc[i] - df['nLoss'].iloc[i] if df['Close'].iloc[i] > prev else df['Close'].iloc[i] + df['nLoss'].iloc[i]
    df['ATR_Stop'] = stop
    return df

def calculate_intraday_vwap(df):
    """
    VWAP with:
    - Source = HLC3
    - Anchor = Session (i.e., VWAP resets every trading day)
    """
    df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['DateOnly'] = df.index.date  # separate session by date

    # Initialize empty series for VWAP
    vwap_series = []

    # Calculate VWAP for each session (i.e., each trading day)
    for date, group in df.groupby('DateOnly'):
        pv = (group['HLC3'] * group['Volume']).cumsum()
        v = group['Volume'].cumsum()
        vwap_day = pv / v
        vwap_series.append(vwap_day)

    df['VWAP'] = pd.concat(vwap_series)
    df.drop(columns=['DateOnly'], inplace=True)
    return df

for i in tqdm(range(0, len(tickers), batch_size)):
    batch = tickers[i:i + batch_size]
    try:
        data = yf.download(batch, period="30d", interval="1h", group_by='ticker', multi_level_index=False)
    except Exception as e:
        print(f"Error downloading batch: {batch} | {e}")
        continue

    for ticker in batch:
        try:
            df = data[ticker].dropna()
            if len(df) < zlema_period + 5:
                continue

            # Compute ATR Stop and ZLEMA, and session VWAP
            df = compute_atr_stop(df)
            price = df['Close']
            adjusted_price = 2 * price - price.shift(lag)
            df['ZLEMA'] = adjusted_price.ewm(span=zlema_period, adjust=False).mean()

            df = calculate_intraday_vwap(df)
            df.dropna(inplace=True)

            # Get last two candles
            prev, curr = df.iloc[-2], df.iloc[-1]

            # Bullish condition
            if (
                curr['Open'] <= curr['VWAP'] and
                curr['Close'] > curr['VWAP'] and
                curr['Close'] > curr['ZLEMA'] and
                curr['Close'] > curr['ATR_Stop'] and
                curr['Close'] > curr['Open']
            ):
                results.append({
                    'Ticker': ticker,
                    'Date': df.index[-1].strftime("%Y-%m-%d %H:%M"),
                    'Signal': 'Bullish',
                    'Close': curr['Close'],
                    'VWAP': curr['VWAP'],
                    'ATR_Stop': curr['ATR_Stop'],
                    'ZLEMA': curr['ZLEMA']
                })

            # Bearish condition
            elif (
                curr['Open'] >= curr['VWAP'] and
                curr['Close'] < curr['VWAP'] and
                curr['Close'] < curr['ZLEMA'] and
                curr['Close'] < curr['ATR_Stop'] and
                curr['Close'] < curr['Open']
            ):
                results.append({
                    'Ticker': ticker,
                    'Date': df.index[-1].strftime("%Y-%m-%d %H:%M"),
                    'Signal': 'Bearish',
                    'Close': curr['Close'],
                    'VWAP': curr['VWAP'],
                    'ATR_Stop': curr['ATR_Stop'],
                    'ZLEMA': curr['ZLEMA']
                })

        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

df = pd.DataFrame(results)
df['Symbol'] = df['Ticker'].str.replace('.NS', '', regex=False)


print(f"Screened {len(df)} stocks")
print(df)

###### Write to google sheets ######
STATE_FILE = 'last_run_date.txt'                # Used to track last cleared date
# Load Google credentials JSON from environment variable
credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not credentials_json:
    raise Exception("Google credentials not found in env variable")

credentials_dict = json.loads(credentials_json)
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
gc = gspread.authorize(credentials)
sheet = gc.open(SHEET_NAME).sheet1

# Check if sheet should be cleared today
def is_new_day():
    today = datetime.now(IST).date().isoformat()
    if not os.path.exists(STATE_FILE) or open(STATE_FILE).read() != today:
        with open(STATE_FILE, 'w') as f:
            f.write(today)
        return True
    return False

# Main logic to write/append the DataFrame
def append_results(df: pd.DataFrame):
    IST = timezone('Asia/Kolkata')
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    df.insert(0, "Timestamp", now)

    existing = get_as_dataframe(sheet, evaluate_formulas=True, header=0)

    if existing.empty or is_new_day():
        # First run of the day – clear and write fresh
        sheet.clear()
        set_with_dataframe(sheet, df)
    else:
        # Append without header
        start_row = len(existing) + 2  # Existing rows + 1 empty + 1 for header
        set_with_dataframe(sheet, df, row=start_row, include_column_header=False)

append_results(df)
print("Results written to sheets successfully.")
