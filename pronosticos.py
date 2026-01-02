import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import argparse
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
#      CONFIGURACION
# ===========================
SYMBOLS = [
    "BTCUSD",
    "NVDA",
    "GOOG",
    "MSFT",
]

TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK_BARS = 220       # ~55 horas con M15
SWING_LOOKBACK = 96       # ~24 horas con M15
MOM_SHORT = 3             # 45 min
MOM_LONG = 6              # 90 min

TZ = ZoneInfo("Europe/Madrid")
DEBUG = True

ATR_LEN = 14

# ===========================
#      UTILIDADES
# ===========================
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

def send_telegram(message: str):
    """Envia mensaje a Telegram con las credenciales provistas."""
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            log(f"Telegram {r.status_code}: {r.text}")
        else:
            log("Telegram OK")
    except Exception as e:
        log(f"Telegram error: {e}")

def init_mt5():
    log("Inicializando MetaTrader 5...")
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init error: {mt5.last_error()}")
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass
    log("MT5 cerrado.")

def ensure_symbol_ready(symbol: str):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"No se pudo suscribir a {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Sin tick para {symbol}")

def copy_rates(symbol, timeframe, n=1000):
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if data is None or len(data) == 0:
        raise RuntimeError(f"No se pudieron obtener datos para {symbol}")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(TZ)
    return df

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

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
#      HEURISTICA REBOTE
# ===========================
def feature_bundle(df: pd.DataFrame) -> dict:
    if len(df) < max(LOOKBACK_BARS, SWING_LOOKBACK) + 10:
        raise ValueError("Historico insuficiente.")

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, ATR_LEN)
    df["mom3"] = df["close"] / df["close"].shift(MOM_SHORT) - 1.0
    df["mom6"] = df["close"] / df["close"].shift(MOM_LONG) - 1.0
    df["hh"] = df["high"].rolling(SWING_LOOKBACK).max()
    df["ll"] = df["low"].rolling(SWING_LOOKBACK).min()

    i = df.index[-2]  # ultima vela cerrada

    close_i = float(df.loc[i, "close"])
    rsi_i = float(df.loc[i, "rsi14"])
    atr_i = float(df.loc[i, "atr14"])
    ema_spread = float((df.loc[i, "ema20"] - df.loc[i, "ema50"]) / (atr_i + 1e-8))
    mom3 = float(df.loc[i, "mom3"])
    mom6 = float(df.loc[i, "mom6"])
    hh = float(df.loc[i, "hh"])
    ll = float(df.loc[i, "ll"])

    drawdown = (hh - close_i) / hh if hh > 0 else 0.0
    dist_low = (close_i - ll) / ll if ll > 0 else 0.0
    decel = mom3 - mom6

    oversold = clamp01((35.0 - rsi_i) / 15.0)         # 1 si RSI <= 20
    drawdown_n = clamp01(drawdown / 0.04)             # 1 si drawdown >= 4%
    decel_n = clamp01(decel / 0.01)                   # 1 si mejora >= 1%
    trend_down = clamp01((-ema_spread) / 1.5)         # 1 si tendencia bajista fuerte
    mom_down = clamp01((-mom3) / 0.01)                # 1 si caida >= 1% en MOM_SHORT

    p_rebound = clamp01(
        0.40 * oversold +
        0.35 * drawdown_n +
        0.15 * decel_n +
        0.10 * (1.0 - trend_down)
    )
    p_down = clamp01(
        0.45 * trend_down +
        0.35 * mom_down +
        0.20 * (1.0 - oversold)
    )

    if mom3 < 0 and ema_spread < 0:
        state = "bajando"
    elif mom3 > 0 and ema_spread > 0:
        state = "subiendo"
    else:
        state = "lateral"

    return {
        "time": df.loc[i, "time"],
        "close": close_i,
        "rsi": rsi_i,
        "atr": atr_i,
        "ema_spread": ema_spread,
        "mom3": mom3,
        "mom6": mom6,
        "drawdown": drawdown,
        "dist_low": dist_low,
        "p_rebound": p_rebound,
        "p_down": p_down,
        "state": state,
    }

def build_recommendation(feat: dict) -> dict:
    buy_th = 0.75
    sell_th = 0.75

    p_rebound = feat["p_rebound"]
    p_down = feat["p_down"]

    if p_rebound >= buy_th and p_rebound >= p_down:
        decision = "buy"
        reason = "Rebote probable."
    elif p_down >= sell_th and p_down > p_rebound:
        decision = "sell"
        reason = "Probable continuacion de la caida."
    else:
        decision = "neutral"
        reason = "No hay ventaja clara."

    return {
        "decision": decision,
        "reason": reason,
        "p_rebound": p_rebound,
        "p_down": p_down,
    }

def format_message(symbol: str, rec: dict, feat: dict) -> str:
    ts = str(feat["time"])
    p_reb = rec["p_rebound"] * 100.0
    p_down = rec["p_down"] * 100.0
    drawdown = feat["drawdown"] * 100.0

    header = f"{symbol} M15 (lookback {SWING_LOOKBACK} velas ~24h)"
    lines = [
        header,
        f"Hora de evaluacion: {ts}",
        f"Estado: {feat['state']}",
        f"RSI14: {feat['rsi']:.1f}",
        f"Distancia desde max reciente: -{drawdown:.2f}%",
        f"Prob rebote: {p_reb:.1f}%",
        f"Prob seguir cayendo: {p_down:.1f}%",
        f"Recomendacion: {rec['decision'].upper()} ({rec['reason']})",
    ]
    return "\n".join(lines)

# ===========================
#       CICLOS / MAIN
# ===========================
def run_once():
    try:
        init_mt5()
        for symbol in SYMBOLS:
            try:
                ensure_symbol_ready(symbol)
                df = copy_rates(symbol, TIMEFRAME, LOOKBACK_BARS)
                feat = feature_bundle(df)
                rec = build_recommendation(feat)
                message = format_message(symbol, rec, feat)
                print(message)
                send_telegram(message)
            except Exception as e:
                log(f"{symbol}: error {e}")
    finally:
        shutdown_mt5()

def run_loop(every_minutes: int):
    sleep_seconds = max(1, int(every_minutes * 60))
    log(f"LOOP: comprobacion cada {every_minutes} minuto(s).")
    while True:
        start_ts = time.time()
        try:
            run_once()
        except Exception as e:
            log(f"Error en iteracion: {e}")
            shutdown_mt5()
        elapsed = time.time() - start_ts
        remaining = max(0, sleep_seconds - int(elapsed))
        time.sleep(remaining)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebotes multi-ticket (BTC, Nvidia, Google, Microsoft) con aviso Telegram."
    )
    parser.add_argument("--every-min", type=int, default=5,
                        help="Intervalo de comprobacion en minutos (por defecto: 5).")
    parser.add_argument("--once", action="store_true",
                        help="Ejecuta una sola vez y termina.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.once:
        run_once()
    else:
        run_loop(args.every_min)
