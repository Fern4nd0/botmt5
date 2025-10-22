import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
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
#      CONFIGURACIÓN
# ===========================
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
# Horizonte por defecto a 5 minutos (1 vela M5). Se puede modificar por CLI con --horizon-min
HORIZON_MIN_DEFAULT = 5

# Zona horaria y verbosidad
TZ = ZoneInfo("Europe/Madrid")
DEBUG = True

# Gestión de riesgo (para SL/TP)
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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ Telegram desactivado: faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID")
        return
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

def require_env_creds():
    if not MT5_LOGIN or not MT5_PASSWORD or not MT5_SERVER:
        raise RuntimeError("Faltan credenciales MT5 en variables de entorno (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).")

def init_mt5():
    require_env_creds()
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
def feature_bundle(df: pd.DataFrame, horizon_min: int, use_live_candle: bool=False) -> dict:
    """Calcula features sobre la vela seleccionada.
    - Si use_live_candle=False: usa la ÚLTIMA VELA CERRADA (idx = -2) [recomendado].
    - Si use_live_candle=True: usa la vela EN FORMACIÓN (idx = -1) [más reactivo, más ruido].
    """
    if len(df) < 100:
        raise ValueError("Histórico insuficiente (<100 velas).")

    bars_ahead = max(1, horizon_min // 5)

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, ATR_LEN)

    # Retorno a horizonte dinámico (antes ret_6 fijo ≈ 30m)
    df["ret_h"] = df["close"] / df["close"].shift(bars_ahead) - 1.0

    # Lookback para rupturas: por defecto 12 (≈ 60m). Puede ajustarse.
    lookback = 12
    df["hh_lb"] = df["high"].rolling(lookback).max()
    df["ll_lb"] = df["low"].rolling(lookback).min()

    # Índice de vela
    i = df.index[-1] if use_live_candle else df.index[-2]

    # Tendencia por EMAs (normalizada)
    ema_spread = (df.loc[i, "ema20"] - df.loc[i, "ema50"]) / (df.loc[i, "atr14"] + 1e-8)

    # Momentum a horizonte HORIZON_MIN
    mom_h = df.loc[i, "ret_h"] / (((df["atr14"].loc[i] / df["close"].loc[i]) + 1e-8))

    # RSI centrado en 50
    rsi_pos = (df.loc[i, "rsi14"] - 50.0) / 50.0  # [-1..+1 aprox]

    # Ruptura reciente
    close_i = df.loc[i, "close"]
    high_i  = df.loc[i, "high"]
    low_i   = df.loc[i, "low"]
    hh_lb   = df.loc[i, "hh_lb"]
    ll_lb   = df.loc[i, "ll_lb"]
    breakout_up = 1.0 if close_i > hh_lb * 0.999 else 0.0
    breakout_dn = 1.0 if close_i < ll_lb * 1.001 else 0.0

    # Hora de apertura y cierre de la vela evaluada
    tf_min = 5  # TIMEFRAME = M5
    candle_open  = df.loc[i, "time"]
    candle_close = candle_open + pd.Timedelta(minutes=tf_min)

    # Detección de datos "viejos"
    now = pd.Timestamp.now(tz=TZ)
    age_min = (now - candle_close).total_seconds() / 60.0

    return {
        "ema_spread": float(ema_spread),
        "mom_h": float(mom_h),
        "rsi_pos": float(rsi_pos),
        "breakout_up": float(breakout_up),
        "breakout_dn": float(breakout_dn),
        "atr": float(df.loc[i, "atr14"]),
        "price_close": float(close_i),
        "price_high": float(high_i),
        "price_low": float(low_i),
        "time_open": candle_open,
        "time_close": candle_close,
        "time": candle_close,        # ← usaremos el cierre como "hora de evaluación"
        "horizon_min": horizon_min,
        "bars_ahead": bars_ahead,
        "use_live_candle": use_live_candle,
        "age_min": float(age_min),
    }

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def predict_up_probability(feat: dict) -> float:
    """Heurística para prob. de subida a ~HORIZON_MIN minutos."""
    w_ema   = 1.2
    w_mom   = 1.0
    w_rsi   = 0.8
    w_brk   = 1.2
    bias    = 0.0
    x = (
            w_ema  * feat["ema_spread"] +
            w_mom  * feat["mom_h"] +
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
    SELL_TH  = 0.60  # si no hay compra, planteamos venta

    horizon_min = feat.get("horizon_min", HORIZON_MIN_DEFAULT)

    if p_up >= BUY_TH:
        decision = "buy"
        reason = f"Probabilidad alta de subida en ~{horizon_min} min."
    else:
        decision = "sell"
        reason = f"Probabilidad insuficiente de subida; preferible venta en ~{horizon_min} min."

    entry = stop = tp = None

    if decision == "buy":
        entry = feat["price_high"]  # buy stop
        stop  = max(feat["price_low"] - ATR_SL_MULT * feat["atr"], entry - 5 * feat["atr"])
        risk  = entry - stop
        if risk <= 0:
            decision = "neutral"
            reason = "Riesgo inválido para SL/TP (compra)."
        else:
            tp    = entry + R_MULT * risk

    elif decision == "sell":
        entry = feat["price_low"]  # sell stop
        stop  = min(feat["price_high"] + ATR_SL_MULT * feat["atr"], entry + 5 * feat["atr"])
        risk  = stop - entry
        if risk <= 0:
            decision = "neutral"
            reason = "Riesgo inválido para SL/TP (venta)."
        else:
            tp    = entry - R_MULT * risk

    return {
        "decision": decision,
        "reason": reason,
        "p_up": p_up,
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "time": feat["time"],  # cierre de la vela evaluada
        "horizon_min": horizon_min,
        "age_min": feat.get("age_min", None),
        "use_live_candle": feat.get("use_live_candle", False),
    }

def format_bilingual_message(symbol: str, rec: dict, feat: dict) -> str:
    """
    Mensaje ES + RU (para compra o venta). Incluye niveles y condición.
    """
    ts_close = str(rec["time"])  # hora de cierre de la vela evaluada
    ts_open  = str(feat.get("time_open"))
    p = rec["p_up"] * 100.0
    horizon = rec.get("horizon_min", HORIZON_MIN_DEFAULT)
    age_min = rec.get("age_min", None)
    use_live = rec.get("use_live_candle", False)

    age_line_es = f"• Edad del dato: {age_min:.1f} min\n" if age_min is not None else ""
    age_line_ru = f"• Возраст данных: {age_min:.1f} мин\n" if age_min is not None else ""
    live_flag_es = " (vela en formación)" if use_live else ""
    live_flag_ru = " (свеча в формировании)" if use_live else ""

    if rec["decision"] == "buy":
        es_hdr = f"📈 Predicción {symbol} (M5, +{horizon}m)\n\n🇪🇸 ES\n"
        es_body = (
            f"• Hora de apertura vela: {ts_open}{live_flag_es}\n"
            f"• Hora de evaluación (cierre vela): {ts_close}\n"
            f"{age_line_es}"
            f"• Prob. de subida ({horizon}m): {p:.1f}%\n"
            f"• Recomendación: ✅ COMPRAR (buy stop si rompe el máximo)\n"
            f"• Entrada: {rec['entry']:.3f}\n"
            f"• Stop: {rec['stop']:.3f}\n"
            f"• Take Profit: {rec['tp']:.3f}\n"
            f"• ATR(14) M5: {feat['atr']:.3f}\n"
            "➡️ Condición: colocar buy stop en la ruptura del máximo de la vela evaluada.\n"
        )
        ru_hdr = "\n🇷🇺 RU\n"
        ru_body = (
            f"• Время открытия свечи: {ts_open}{live_flag_ru}\n"
            f"• Время оценки (закрытие свечи): {ts_close}\n"
            f"{age_line_ru}"
            f"• Вероятность роста ({horizon}м): {p:.1f}%\n"
            f"• Рекомендация: ✅ ПОКУПАТЬ (buy stop при пробое максимума)\n"
            f"• Вход: {rec['entry']:.3f}\n"
            f"• Стоп: {rec['stop']:.3f}\n"
            f"• Тейк-профит: {rec['tp']:.3f}\n"
            f"• ATR(14) M5: {feat['atr']:.3f}\n"
            "➡️ Условие: разместить buy stop при пробое максимума оцененной свечи."
        )
        return es_hdr + es_body + ru_hdr + ru_body

    else:  # SELL o NEUTRAL -> enviamos mensaje de venta
        es_hdr = f"📉 Predicción {symbol} (M5, +{horizon}m)\n\n🇪🇸 ES\n"
        es_body = (
            f"• Hora de apertura vela: {ts_open}{live_flag_es}\n"
            f"• Hora de evaluación (cierre vela): {ts_close}\n"
            f"{age_line_es}"
            f"• Prob. de subida ({horizon}m): {p:.1f}%\n"
            f"• Recomendación: 🔻 VENDER (sell stop si rompe el mínimo)\n"
            f"• Entrada: {rec['entry']:.3f}\n"
            f"• Stop: {rec['stop']:.3f}\n"
            f"• Take Profit: {rec['tp']:.3f}\n"
            f"• ATR(14) M5: {feat['atr']:.3f}\n"
            "➡️ Condición: colocar sell stop en la ruptura del mínimo de la vela evaluada.\n"
        )
        ru_hdr = "\n🇷🇺 RU\n"
        ru_body = (
            f"• Время открытия свечи: {ts_open}{live_flag_ru}\n"
            f"• Время оценки (закрытие свечи): {ts_close}\n"
            f"{age_line_ru}"
            f"• Вероятность роста ({horizon}м): {p:.1f}%\n"
            f"• Рекомендация: 🔻 ПРОДАВАТЬ (sell stop при пробое минимума)\n"
            f"• Вход: {rec['entry']:.3f}\n"
            f"• Стоп: {rec['stop']:.3f}\n"
            f"• Тейк-профит: {rec['tp']:.3f}\n"
            f"• ATR(14) M5: {feat['atr']:.3f}\n"
            "➡️ Условие: разместить sell stop при пробое минимума оцененной свечи."
        )
        return es_hdr + es_body + ru_hdr + ru_body

# ===========================
#       CICLOS / MAIN
# ===========================
def run_once(horizon_min: int, use_live_candle: bool=False):
    try:
        init_mt5()
        log(f"📊 Analizando {SYMBOL} M5…")
        df = copy_rates(SYMBOL, TIMEFRAME, 600)  # ~50h de datos
        log(f"📈 Últimas 3 velas: {list(df['time'].tail(3))}")

        feat = feature_bundle(df, horizon_min=horizon_min, use_live_candle=use_live_candle)
        if feat["age_min"] > 10 and not use_live_candle:
            log(f"⚠️ Datos retrasados {feat['age_min']:.1f} min; revisa conexión/mercado.")

        p_up = predict_up_probability(feat)
        rec  = build_recommendation(feat, p_up)

        # Enviar SIEMPRE mensaje: compra o venta (según petición)
        message = format_bilingual_message(SYMBOL, rec, feat)
        print(message)
        send_telegram(message)

    finally:
        shutdown_mt5()

def run_loop(every_minutes: int, horizon_min: int, use_live_candle: bool=False):
    sleep_seconds = max(1, int(every_minutes * 60))
    log(f"♻️ LOOP: comprobación cada {every_minutes} minuto(s). Horizonte={horizon_min}m. Live={use_live_candle}.")
    while True:
        start_ts = time.time()
        try:
            run_once(horizon_min=horizon_min, use_live_candle=use_live_candle)
        except Exception as e:
            log(f"❌ Error en iteración: {e}")
            shutdown_mt5()
        elapsed = time.time() - start_ts
        remaining = max(0, sleep_seconds - int(elapsed))
        log(f"⏳ Siguiente comprobación en ~{remaining} s.")
        time.sleep(remaining)

def parse_args():
    parser = argparse.ArgumentParser(description="Predicción USDJPY M5 a horizonte ajustable (envía compra o venta, ES+RU).")
    parser.add_argument("--every-min", type=int, default=5,
                        help="Intervalo de comprobación en minutos (por defecto: 5).")
    parser.add_argument("--horizon-min", type=int, default=HORIZON_MIN_DEFAULT,
                        help=f"Horizonte de predicción en minutos (por defecto: {HORIZON_MIN_DEFAULT}).")
    parser.add_argument("--use-live-candle", action="store_true",
                        help="Usar la última vela en formación (más reactivo, más ruido).")
    parser.add_argument("--once", action="store_true",
                        help="Ejecuta una sola vez y termina.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.once:
        run_once(horizon_min=args.horizon_min, use_live_candle=args.use_live_candle)
    else:
        run_loop(args.every_min, horizon_min=args.horizon_min, use_live_candle=args.use_live_candle)