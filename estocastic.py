import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

# ===== CONFIGURACIÓN MT5 =====
MT5_SERVER   = "ForexClubBY-MT5 Demo Server"
MT5_LOGIN    = 520002796
MT5_PASSWORD = "PclNMY2*"
PATH_TO_TERMINAL = None
# ==============================

# ===== CONFIGURACIÓN TELEGRAM =====
TELEGRAM_BOT_TOKEN = "7324855560:AAF3WHWC7cqwGFEcIa6rUKDjvJcyO9uvj3I"
TELEGRAM_CHAT_ID = "-4979821691"
# ==================================

SYMBOL = "USDJPY"
TIMEFRAME_SIGNAL = mt5.TIMEFRAME_H1
TIMEFRAME_ATR = mt5.TIMEFRAME_H1
STO_K = 14
STO_D = 3
STO_SMOOTH = 3
ATR_LEN = 14

# ===== FUNCIONES BASE =====
def send_telegram(message: str):
    """Envía una alerta al chat de Telegram configurado."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"⚠️ Error enviando mensaje: {response.text}")
        else:
            print("📨 Alerta enviada a Telegram")
    except Exception as e:
        print(f"❌ Error enviando a Telegram: {e}")

def init_mt5():
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init error: {mt5.last_error()}")
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")
    print("✅ Conectado a MetaTrader 5")

def copy_rates(symbol, timeframe, n=500):
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if data is None or len(data) == 0:
        raise RuntimeError("No se pudieron obtener datos")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# ===== INDICADORES =====
def stochastic(df, k=14, d=3, smooth=3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    sto_k = 100 * (df["close"] - low_min) / (high_max - low_min)
    sto_k = sto_k.rolling(smooth).mean()
    sto_d = sto_k.rolling(d).mean()
    df["sto_k"], df["sto_d"] = sto_k, sto_d
    return df

def atr(df, length=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ===== LÓGICA DE SEÑAL =====
def find_stochastic_signal(df):
    # Condiciones:
    # - %K < 20 y %D < 20
    # - %K cruza %D al alza
    # - Cierre fuera de 20 (confirmación)
    sig = []
    for i in range(2, len(df)):
        k1, d1 = df["sto_k"].iloc[i-1], df["sto_d"].iloc[i-1]
        k2, d2 = df["sto_k"].iloc[i], df["sto_d"].iloc[i]
        if k1 < d1 and k2 > d2 and k2 > 20 and k1 < 20:
            sig.append(i)
    return sig

def get_trade_setup(df, atr_val):
    sig_idx = find_stochastic_signal(df)
    if not sig_idx:
        print("ℹ️ No hay señales válidas aún.")
        return None

    i = sig_idx[-1]  # última señal
    entry_price = df["high"].iloc[i]  # ruptura del máximo
    stop_price = min(df["low"].iloc[i-1], df["low"].iloc[i] - atr_val)
    risk = entry_price - stop_price
    take_profit = entry_price + 2 * risk  # 2x riesgo

    return {
        "signal_time": df["time"].iloc[i],
        "entry": float(entry_price),
        "stop": float(stop_price),
        "tp": float(take_profit)
    }

# ===== MAIN =====
def main():
    init_mt5()

    print(f"📊 Analizando {SYMBOL} en H1...")
    df = copy_rates(SYMBOL, TIMEFRAME_SIGNAL, 300)
    df = stochastic(df, STO_K, STO_D, STO_SMOOTH)
    df_atr = copy_rates(SYMBOL, TIMEFRAME_ATR, 200)
    atr_val = atr(df_atr, ATR_LEN).iloc[-1]

    setup = get_trade_setup(df, atr_val)
    if setup:
        mensaje = (
            f"📊 Señal USDJPY detectada\n"
            f"Hora señal: {setup['signal_time']}\n"
            f"Entrada: {setup['entry']:.3f}\n"
            f"Stop: {setup['stop']:.3f}\n"
            f"Take Profit: {setup['tp']:.3f}\n"
            f"ATR(14): {atr_val:.3f}\n\n"
            f"➡️ Condición: Esperar ruptura del máximo de la vela señal."
        )
        print(mensaje)
        send_telegram(mensaje)
    else:
        print("❌ Ninguna señal activa por ahora.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
