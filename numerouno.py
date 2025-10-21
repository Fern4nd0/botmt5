import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import time
import argparse
from zoneinfo import ZoneInfo
from typing import Optional

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
TIMEFRAME = mt5.TIMEFRAME_M5           # M5
HORIZON_MIN = 30                       # 30 min => 6 velas M5
BARS_AHEAD = max(1, HORIZON_MIN // 5)  # 6

# Zona horaria y verbosidad
TZ = ZoneInfo("Europe/Madrid")
DEBUG = True

# Gestión de riesgo (para SL/TP si hay compra)
ATR_LEN = 14
R_MULT = 2.0            # TP = 2R
ATR_SL_MULT = 1.0       # SL ~ 1x ATR

# ===========================
#      UTILIDADES
# ===========================
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

def send_telegram(message: str):
    """Envía mensaje a Telegram con las credenciales provistas."""
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            log(f"⚠️ Telegram {r.status_code}: {r.text}")
        else:
            log("📨 Telegram OK")
    except Exception as e:
        log(f"❌ Telegram error: {e}")

def init_mt5():
    log("🔌 Inicializando MetaTrader 5...")
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init error: {mt5.last_error()}")
    # Login con tus credenciales
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"No se pudo suscribir a {SYMBOL}")
    tick = mt5.symbol_info_tick(SYMBOL)
    log(f"ℹ️ Tick inicial {SYMBOL}: {tick}")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass
    log("🔚 MT5 cerrado.")

def copy_rates(symbol, timeframe, n=1000):
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if data is None or len(data) == 0:
        raise RuntimeError("No se pudieron obtener datos de velas")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
    return df

# ===========================
#      INDICADORES
# ===========================
def ema(series: pd.Series, length: int):
    return series.ewm(span=length, adjust=False).mean()

def rsi(close: pd.Series, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ===========================
#      MODELO HEURÍSTICO
# ===========================
def feature_bundle(df: pd.DataFrame) -> dict:
    """Calcula features en la ÚLTIMA VELA CERRADA (idx = -2)."""
    if len(df) < 100:
        raise ValueError("Histórico insuficiente (<100 velas).")

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, ATR_LEN)
    df["ret_6"] = df["close"] / df["close"].shift(6) - 1.0
    df["hh_12"] = df["high"].rolling(12).max()
    df["ll_12"] = df["low"].rolling(12).min()

    i = df.index[-2]  # última vela CERRADA

    # Tendencia por EMAs (normalizada)
    ema_spread = (df.loc[i, "ema20"] - df.loc[i, "ema50"]) / (df.loc[i, "atr14"] + 1e-8)

    # Momentum 6 velas (horizonte ~30m)
    mom6 = df.loc[i, "ret_6"] / (((df["atr14"].loc[i] / df["close"].loc[i]) + 1e-8))

    # RSI centrado en 50
    rsi_pos = (df.loc[i, "rsi14"] - 50.0) / 50.0  # [-1..+1 aprox]

    # Ruptura reciente
    close_i = df.loc[i, "close"]
    high_i  = df.loc[i, "high"]
    low_i   = df.loc[i, "low"]
    hh12    = df.loc[i, "hh_12"]
    ll12    = df.loc[i, "ll_12"]
    breakout_up = 1.0 if close_i > hh12 * 0.999 else 0.0
    breakout_dn = 1.0 if close_i < ll12 * 1.001 else 0.0

    return {
        "i": i,
        "ema_spread": float(ema_spread),
        "mom6": float(mom6),
        "rsi_pos": float(rsi_pos),
        "breakout_up": float(breakout_up),
        "breakout_dn": float(breakout_dn),
        "atr": float(df.loc[i, "atr14"]),
        "price_close": float(close_i),
        "price_high": float(high_i),
        "price_low": float(low_i),
        "time": df.loc[i, "time"],
    }

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def predict_up_probability(feat: dict) -> float:
    """Heurística para prob. de subida a ~30m."""
    w_ema   = 1.2
    w_mom6  = 1.0
    w_rsi   = 0.8
    w_brk   = 1.2
    bias    = 0.0
    x = (
        w_ema  * feat["ema_spread"] +
        w_mom6 * feat["mom6"] +
        w_rsi  * feat["rsi_pos"] +
        w_brk  * (feat["breakout_up"] - 0.8 * feat["breakout_dn"]) +
        bias
    )
    return float(sigmoid(x))

# ===========================
#      DECISIÓN Y MENSAJE
# ===========================
def build_recommendation(feat: dict, p_up: float) -> dict:
    BUY_TH   = 0.60
    AVOID_TH = 0.40

    if p_up >= BUY_TH:
        decision = "buy"
        reason = "Probabilidad alta de subida en ~30 min."
    elif p_up <= AVOID_TH:
        decision = "avoid"
        reason = "Probabilidad baja de subida; posible caída o lateral."
    else:
        decision = "neutral"
        reason = "Señales mixtas; no hay ventaja clara."

    entry = stop = tp = None
    if decision == "buy":
        entry = feat["price_high"]  # buy stop
        stop  = max(feat["price_low"] - ATR_SL_MULT * feat["atr"], entry - 5 * feat["atr"])
        risk  = entry - stop
        if risk <= 0:
            decision = "neutral"
            reason = "Riesgo inválido para SL/TP."
        else:
            tp    = entry + R_MULT * risk

    return {
        "decision": decision,
        "reason": reason,
        "p_up": p_up,
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "time": feat["time"],
    }

def format_bilingual_message(symbol: str, rec: dict, feat: dict) -> str:
    """
    Mensaje ES + RU (solo para compra). Incluye niveles y condición.
    """
    ts = str(rec["time"])
    p = rec["p_up"] * 100.0

    es = (
        f"📈 Predicción {symbol} (M5, +30m)\n\n"
        "🇪🇸 ES\n"
        f"• Hora de evaluación: {ts}\n"
        f"• Prob. de subida (30m): {p:.1f}%\n"
        f"• Recomendación: ✅ COMPRAR (buy stop si rompe el máximo)\n"
        f"• Entrada: {rec['entry']:.3f}\n"
        f"• Stop: {rec['stop']:.3f}\n"
        f"• Take Profit: {rec['tp']:.3f}\n"
        f"• ATR(14) M5: {feat['atr']:.3f}\n"
        "➡️ Condición: colocar buy stop en la ruptura del máximo de la vela evaluada.\n"
    )

    ru = (
        "\n🇷🇺 RU\n"
        f"• Время оценки: {ts}\n"
        f"• Вероятность роста (30м): {p:.1f}%\n"
        f"• Рекомендация: ✅ ПОКУПАТЬ (buy stop при пробое максимума)\n"
        f"• Вход: {rec['entry']:.3f}\n"
        f"• Стоп: {rec['stop']:.3f}\n"
        f"• Тейк-профит: {rec['tp']:.3f}\n"
        f"• ATR(14) M5: {feat['atr']:.3f}\n"
        "➡️ Условие: разместить buy stop при пробое максимума оцененной свечи."
    )

    return es + ru

# ===========================
#       CICLOS / MAIN
# ===========================
def run_once():
    try:
        init_mt5()
        log(f"📊 Analizando {SYMBOL} M5…")
        df = copy_rates(SYMBOL, TIMEFRAME, 600)  # ~50h de datos
        log(f"📈 Últimas 3 velas: {list(df['time'].tail(3))}")

        feat = feature_bundle(df)
        p_up = predict_up_probability(feat)
        rec  = build_recommendation(feat, p_up)

        if rec["decision"] == "buy":
            message = format_bilingual_message(SYMBOL, rec, feat)
            print(message)
            send_telegram(message)
        else:
            log(f"🔕 Sin envío: decisión = {rec['decision']} (p_up={rec['p_up']:.2f})")

    finally:
        shutdown_mt5()

def run_loop(every_minutes: int):
    sleep_seconds = max(1, int(every_minutes * 60))
    log(f"♻️ LOOP: comprobación cada {every_minutes} minuto(s).")
    while True:
        start_ts = time.time()
        try:
            run_once()
        except Exception as e:
            log(f"❌ Error en iteración: {e}")
            shutdown_mt5()
        elapsed = time.time() - start_ts
        remaining = max(0, sleep_seconds - int(elapsed))
        log(f"⏳ Siguiente comprobación en ~{remaining} s.")
        time.sleep(remaining)

def parse_args():
    parser = argparse.ArgumentParser(description="Predicción USDJPY M5 a 30m (solo envía si es compra, ES+RU).")
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
