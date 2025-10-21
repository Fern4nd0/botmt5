import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import json
import os
import time
import argparse
from typing import Optional
from zoneinfo import ZoneInfo

# ===========================
#      CREDENCIALES (TU SCRIPT)
# ===========================
MT5_SERVER         = "ForexClubBY-MT5 Demo Server"
MT5_LOGIN          = 520002796
MT5_PASSWORD       = "PclNMY2*"
PATH_TO_TERMINAL   = None  # pon la ruta si MT5 no inicia sin ella

TELEGRAM_BOT_TOKEN = "7324855560:AAF3WHWC7cqwGFEcIa6rUKDjvJcyO9uvj3I"
TELEGRAM_CHAT_ID   = "-4979821691"

# ===========================
#      CONFIGURACIÓN
# ===========================
SYMBOL = "USDJPY"

# Timeframes
TIMEFRAME_SIGNAL = mt5.TIMEFRAME_H1
TIMEFRAME_ATR    = mt5.TIMEFRAME_H1

# Parámetros indicadores
ATR_LEN = 14

# Parámetros RSI
RSI_LEN = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# A) Señal reciente: últimas N velas cerradas
MAX_CLOSED_BARS_AGE = 2

# B) Archivo de estado
STATE_FILE = "last_signal.json"

# C) Zona horaria
TZ = ZoneInfo("Europe/Madrid")

# Verbosidad
DEBUG = True

# ===========================
#      UTILIDADES
# ===========================
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

def send_telegram(message: str):
    """Envía un único mensaje a Telegram (ES + RU)."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        resp = requests.post(url, data=data, timeout=15)
        if resp.status_code != 200:
            log(f"⚠️ Telegram {resp.status_code}: {resp.text}")
        else:
            log("📨 Telegram OK")
    except Exception as e:
        log(f"❌ Telegram error: {e}")

def init_mt5():
    """Inicializa MT5, loguea, suscribe símbolo y fuerza datos (D)."""
    log("🔌 Inicializando MetaTrader 5...")
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init error: {mt5.last_error()}")
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"No se pudo suscribir a {SYMBOL}")
    tick = mt5.symbol_info_tick(SYMBOL)
    log(f"ℹ️ Tick inicial {SYMBOL}: {tick}")

def copy_rates(symbol, timeframe, n=500):
    """Obtiene velas y las convierte a Europe/Madrid (C)."""
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if data is None or len(data) == 0:
        raise RuntimeError("No se pudieron obtener datos de velas")
    df = pd.DataFrame(data)
    # MT5 da epoch UTC; localizamos y convertimos
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
    return df

def already_sent(signal_time) -> bool:
    """True si ya enviamos una señal con ese timestamp (B)."""
    if not os.path.exists(STATE_FILE):
        return False
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_time") == str(signal_time)
    except Exception as e:
        log(f"⚠️ No se pudo leer {STATE_FILE}: {e}")
        return False

def mark_sent(signal_time):
    """Guarda el timestamp de la última señal enviada (B)."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_time": str(signal_time)}, f)
    except Exception as e:
        log(f"⚠️ No se pudo guardar {STATE_FILE}: {e}")

# ===========================
#      INDICADORES
# ===========================
def atr(df, length=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def rsi(close: pd.Series, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Promedios exponenciales (Wilder)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

# ===========================
#      LÓGICA DE SEÑAL (RSI)
# ===========================
def find_rsi_signal(df) -> Optional[tuple]:
    """
    Señal basada en RSI de la ÚLTIMA VELA CERRADA.
    - RSI < 30  => posible LONG (sobrevendido)
    - RSI > 70  => posible SHORT (sobrecomprado)
    Devuelve: (index_ultima_señal_cerrada, "long"|"short") o None
    """
    if len(df) < 3 or "rsi" not in df.columns:
        return None
    i = df.index[-2]  # última vela CERRADA
    val = df["rsi"].iloc[i]
    if pd.isna(val):
        return None
    if val < RSI_OVERSOLD:
        return (i, "long")
    elif val > RSI_OVERBOUGHT:
        return (i, "short")
    return None

def get_trade_setup(df, atr_val) -> Optional[dict]:
    """
    Construye setup LONG/SHORT según RSI extremo en la vela cerrada.
    Entrada por ruptura del extremo de la vela señal, SL con colchón ATR, TP = 2R.
    """
    out = find_rsi_signal(df)
    log(f"🔎 Señal RSI: {out}")
    if out is None:
        log("ℹ️ No hay señales RSI válidas aún.")
        return None

    i, direction = out
    if direction == "long":
        entry_price = float(df["high"].iloc[i])                   # buy stop en ruptura del máximo
        stop_price  = float(min(df["low"].iloc[i-1], df["low"].iloc[i] - atr_val))
        risk = entry_price - stop_price
        take_profit = float(entry_price + 2 * risk)
        cond_es = "esperar la ruptura del máximo de la vela de la señal."
        cond_ru = "дождаться пробоя максимума сигнальной свечи."
    else:  # "short"
        entry_price = float(df["low"].iloc[i])                    # sell stop en ruptura del mínimo
        stop_price  = float(max(df["high"].iloc[i-1], df["high"].iloc[i] + atr_val))
        risk = stop_price - entry_price
        take_profit = float(entry_price - 2 * risk)
        cond_es = "esperar la ruptura del mínimo de la vela de la señal."
        cond_ru = "дождаться пробоя минимума сигнальной свечи."

    setup = {
        "signal_time": df["time"].iloc[i],
        "entry": entry_price,
        "stop": stop_price,
        "tp": take_profit,
        "index": i,
        "direction": direction,
        "cond_es": cond_es,
        "cond_ru": cond_ru,
    }
    log(f"✅ Setup RSI: {setup}")
    return setup

def is_recent_signal(df, signal_index, max_closed_bars_age=2) -> bool:
    """
    A) 'Reciente' == dentro de las últimas N velas cerradas.
       df.index[-1]  -> vela en curso (sin cerrar)
       df.index[-2]  -> última vela cerrada
    """
    if len(df) < 3:
        return False
    last_closed_idx = df.index[-2]
    min_allowed_idx = last_closed_idx - (max_closed_bars_age - 1)
    ok = signal_index >= min_allowed_idx
    log(f"🕒 Reciente? {ok} (signal_idx={signal_index}, min_allowed_idx={min_allowed_idx}, last_closed_idx={last_closed_idx})")
    return ok

def build_message(setup, atr_val) -> str:
    """
    Mantiene EXACTAMENTE la estructura del mensaje original de Telegram.
    Solo cambia dinámicamente la línea de 'Condición' según LONG/SHORT por RSI.
    """
    sig_time_str = str(setup["signal_time"])
    mensaje = (
        "📊 Señal USDJPY detectada / Обнаружен сигнал USDJPY\n\n"
        "🇪🇸 ES\n"
        f"• Hora de la señal: {sig_time_str}\n"
        f"• Entrada: {setup['entry']:.3f}\n"
        f"• Stop: {setup['stop']:.3f}\n"
        f"• Take Profit: {setup['tp']:.3f}\n"
        f"• ATR(14): {atr_val:.3f}\n"
        f"➡️ Condición: {setup['cond_es']}\n\n"
        "🇷🇺 RU\n"
        f"• Время сигнала: {sig_time_str}\n"
        f"• Вход: {setup['entry']:.3f}\n"
        f"• Стоп: {setup['stop']:.3f}\n"
        f"• Тейк-профит: {setup['tp']:.3f}\n"
        f"• ATR(14): {atr_val:.3f}\n"
        f"➡️ Условие: {setup['cond_ru']}"
    )
    return mensaje

# ===========================
#       CICLOS / MAIN
# ===========================
def run_once():
    try:
        init_mt5()
        log(f"📊 Analizando {SYMBOL} H1...")

        # Datos para la señal
        df = copy_rates(SYMBOL, TIMEFRAME_SIGNAL, 300)
        log(f"📈 Últimas 3 velas: {list(df['time'].tail(3))}")

        # --- RSI para señal ---
        df["rsi"] = rsi(df["close"], RSI_LEN)

        # --- ATR para gestión ---
        df_atr = copy_rates(SYMBOL, TIMEFRAME_ATR, 200)
        atr_series = atr(df_atr, ATR_LEN)
        atr_val = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else np.nan

        if np.isnan(atr_val):
            log("ℹ️ ATR insuficiente (NaN).")
            return

        # --- Setup por RSI ---
        setup = get_trade_setup(df, atr_val)
        if setup is None:
            log("❌ Ninguna señal activa por ahora (criterios RSI no cumplidos).")
            return

        if not is_recent_signal(df, setup["index"], MAX_CLOSED_BARS_AGE):
            log("ℹ️ Señal histórica (no reciente). No envío.")
            return

        if already_sent(setup["signal_time"]):
            log("ℹ️ Señal ya enviada antes. No repito.")
            return

        mensaje = build_message(setup, atr_val)
        print(mensaje)
        send_telegram(mensaje)
        mark_sent(setup["signal_time"])

    finally:
        mt5.shutdown()
        log("🔚 MT5 cerrado.")

def run_loop(every_minutes: int):
    sleep_seconds = max(1, int(every_minutes * 60))
    log(f"♻️ Modo LOOP: comprobación cada {every_minutes} minuto(s).")
    while True:
        start_ts = time.time()
        try:
            run_once()
        except Exception as e:
            log(f"❌ Error en iteración: {e}")
            try:
                mt5.shutdown()
            except Exception:
                pass
        # Dormir exactamente hasta completar el intervalo parametrizado
        elapsed = time.time() - start_ts
        remaining = max(0, sleep_seconds - int(elapsed))
        log(f"⏳ Siguiente comprobación en ~{remaining} s.")
        time.sleep(remaining)

def parse_args():
    parser = argparse.ArgumentParser(description="Detector de señales USDJPY (RSI + ATR) con alertas Telegram.")
    parser.add_argument("--every-min", type=int, default=5,
                        help="Intervalo de comprobación en minutos (por defecto: 5).")
    parser.add_argument("--once", action="store_true",
                        help="Ejecuta una sola vez y termina.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.once:
        run_once()
    else:
        run_loop(args.every_min)
