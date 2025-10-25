import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import argparse
from zoneinfo import ZoneInfo

# ===========================
#      CREDENCIALES
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
SYMBOLS = [
    "AUDUSD",
    "BTCUSD",
    "EURUSD",
    "GBPUSD",
    "Google",     # ajusta si tu broker usa GOOGL/ALPHABET o similar
    "Microsoft",  # ajusta si tu broker usa MSFT
    "NZDUSD",
    "Tesla",      # ajusta si tu broker usa TSLA
    "USDCAD",
    "USDCHF",
    "USDCNH",
    "USDJPY",
    "USDSEK",
    "XAUEUR",
    "XAUUSD",
    "nVidia"      # ajusta si tu broker usa NVDA
]

TIMEFRAME = mt5.TIMEFRAME_M5
HORIZON_MIN_DEFAULT = 5

TZ = ZoneInfo("Europe/Madrid")
DEBUG = True

# Gestión de riesgo
ATR_LEN = 14
R_MULT = 2.0
ATR_SL_MULT = 1.0

# Trading automático
TRADE_VOLUME     = 0.10     # lotes por operación
MAX_SPREAD_PIPS  = 3        # no entrar si spread > X pips aprox (para forex 5 dígitos)
PROB_TRADE_TH    = 0.75     # umbral de probabilidad para ejecutar BUY
AVOID_DUP_BUY    = True     # no abrir nueva BUY si ya existe una abierta en el símbolo

# ===========================
#      UTILIDADES
# ===========================
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

def send_telegram(message: str):
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
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")

def ensure_symbol_ready(symbol: str):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"No se pudo suscribir a {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Sin tick info para {symbol}")
    log(f"ℹ️ Tick inicial {symbol}: {tick}")

def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass
    log("🔚 MT5 cerrado.")

def copy_rates(symbol, timeframe, n=1000):
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if data is None or len(data) == 0:
        raise RuntimeError(f"No se pudieron obtener datos de velas para {symbol}")
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
    if len(df) < 100:
        raise ValueError("Histórico insuficiente (<100 velas).")

    bars_ahead = max(1, horizon_min // 5)

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, ATR_LEN)

    df["ret_h"] = df["close"] / df["close"].shift(bars_ahead) - 1.0

    lookback = 12
    df["hh_lb"] = df["high"].rolling(lookback).max()
    df["ll_lb"] = df["low"].rolling(lookback).min()

    i = df.index[-1] if use_live_candle else df.index[-2]

    ema_spread = (df.loc[i, "ema20"] - df.loc[i, "ema50"]) / (df.loc[i, "atr14"] + 1e-8)
    mom_h = df.loc[i, "ret_h"] / (((df["atr14"].loc[i] / df["close"].loc[i]) + 1e-8))
    rsi_pos = (df.loc[i, "rsi14"] - 50.0) / 50.0

    close_i = df.loc[i, "close"]
    high_i  = df.loc[i, "high"]
    low_i   = df.loc[i, "low"]
    hh_lb   = df.loc[i, "hh_lb"]
    ll_lb   = df.loc[i, "ll_lb"]
    breakout_up = 1.0 if close_i > hh_lb * 0.999 else 0.0
    breakout_dn = 1.0 if close_i < ll_lb * 1.001 else 0.0

    tf_min = 5
    candle_open  = df.loc[i, "time"]
    candle_close = candle_open + pd.Timedelta(minutes=tf_min)

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
        "time": candle_close,
        "horizon_min": horizon_min,
        "bars_ahead": bars_ahead,
        "use_live_candle": use_live_candle,
        "age_min": float(age_min),
    }

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def predict_up_probability(feat: dict) -> float:
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
    horizon_min = feat.get("horizon_min", HORIZON_MIN_DEFAULT)

    if p_up >= BUY_TH:
        decision = "buy"
        reason = f"Probabilidad alta de subida en ~{horizon_min} min."
    else:
        decision = "sell"
        reason = f"Probabilidad insuficiente de subida; preferible venta en ~{horizon_min} min."

    entry = stop = tp = None

    if decision == "buy":
        entry = feat["price_high"]
        stop  = max(feat["price_low"] - ATR_SL_MULT * feat["atr"], entry - 5 * feat["atr"])
        risk  = entry - stop
        if risk <= 0:
            decision = "neutral"
            reason = "Riesgo inválido para SL/TP (compra)."
        else:
            tp    = entry + R_MULT * risk

    elif decision == "sell":
        entry = feat["price_low"]
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
        "time": feat["time"],
        "horizon_min": horizon_min,
        "age_min": feat.get("age_min", None),
        "use_live_candle": feat.get("use_live_candle", False),
    }

def format_bilingual_message(symbol: str, rec: dict, feat: dict) -> str:
    ts_close = str(rec["time"])
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

    else:
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
#      TRADING REAL
# ===========================
def has_open_buy_position(symbol: str) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    for p in positions:
        if p.type == mt5.POSITION_TYPE_BUY:
            return True
    return False

def place_buy_order(symbol: str, sl_price: float, tp_price: float, volume: float):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"{symbol}: symbol_info() devolvió None")
    if not info.trade_allowed:
        raise RuntimeError(f"{symbol}: trading no permitido en este símbolo")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"{symbol}: no hay tick disponible")

    ask = tick.ask
    bid = tick.bid
    spread_points = (ask - bid) / info.point

    # Para forex con 5 dígitos: 1 pip ~ 10 puntos
    try:
        points_per_pip = 10.0
        if spread_points > MAX_SPREAD_PIPS * points_per_pip:
            raise RuntimeError(f"{symbol}: spread demasiado alto ({spread_points:.1f} puntos)")
    except Exception:
        pass

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_BUY,
        "price": float(ask),
        "sl": float(sl_price) if sl_price is not None else 0.0,
        "tp": float(tp_price) if tp_price is not None else 0.0,
        "deviation": 20,
        "magic": 123456,
        "comment": "auto-buy p_up>=0.75",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"{symbol}: order_send() devolvió None, error {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"{symbol}: order_send fallo retcode={result.retcode} detalles={result}")

    return result

# ===========================
#       CICLOS / MAIN
# ===========================
def analyze_symbol(symbol: str, horizon_min: int, use_live_candle: bool=False) -> str:
    ensure_symbol_ready(symbol)
    log(f"📊 Analizando {symbol} M5…")
    df = copy_rates(symbol, TIMEFRAME, 600)
    log(f"📈 Últimas 3 velas {symbol}: {list(df['time'].tail(3))}")

    feat = feature_bundle(df, horizon_min=horizon_min, use_live_candle=use_live_candle)
    if feat["age_min"] > 10 and not use_live_candle:
        log(f"⚠️ {symbol}: datos retrasados {feat['age_min']:.1f} min; revisa conexión/mercado.")

    p_up = predict_up_probability(feat)
    rec  = build_recommendation(feat, p_up)

    message = format_bilingual_message(symbol, rec, feat)

    # Auto-trade: sólo BUY cuando p_up >= PROB_TRADE_TH
    try:
        if p_up >= PROB_TRADE_TH:
            if AVOID_DUP_BUY and has_open_buy_position(symbol):
                log(f"ℹ️ {symbol}: ya existe una BUY abierta. No se abre otra.")
            else:
                if rec["decision"] == "buy":
                    sl_price = rec["stop"]
                    tp_price = rec["tp"]
                    tick_now = mt5.symbol_info_tick(symbol)
                    if tick_now is not None and sl_price is not None and tp_price is not None:
                        current_ask = tick_now.ask
                        if sl_price < current_ask < tp_price:
                            log(f"🚀 Enviando BUY {symbol} vol={TRADE_VOLUME} SL={sl_price:.3f} TP={tp_price:.3f}")
                            trade_result = place_buy_order(symbol, sl_price, tp_price, TRADE_VOLUME)
                            log(f"✅ Orden enviada {symbol}: retcode={trade_result.retcode}")
                            message += (
                                "\n\n[AUTO-TRADE]\n"
                                f"Se envió BUY {symbol} vol={TRADE_VOLUME} SL={sl_price:.3f} TP={tp_price:.3f}\n"
                                f"retcode={trade_result.retcode}"
                            )
                        else:
                            log(f"⛔ {symbol}: SL/TP no válidos respecto al precio actual. No se ejecuta trade.")
                    else:
                        log(f"⛔ {symbol}: no se pudo validar precio/SL/TP. No se ejecuta trade.")
                else:
                    log(f"ℹ️ {symbol}: p_up>={PROB_TRADE_TH:.2f} pero la lógica no es 'buy'. No se compra.")
        else:
            log(f"ℹ️ {symbol}: p_up={p_up:.3f} < {PROB_TRADE_TH:.2f}, no compramos.")
    except Exception as trade_err:
        log(f"❌ Error al intentar operar {symbol}: {trade_err}")

    return message

def run_once(horizon_min: int, use_live_candle: bool=False):
    try:
        init_mt5()
        for symbol in SYMBOLS:
            try:
                msg = analyze_symbol(symbol, horizon_min=horizon_min, use_live_candle=use_live_candle)
                print(msg)
                send_telegram(msg)
            except Exception as e_symbol:
                log(f"❌ Error con {symbol}: {e_symbol}")
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
            log(f"❌ Error en iteración global: {e}")
            shutdown_mt5()
        elapsed = time.time() - start_ts
        remaining = max(0, sleep_seconds - int(elapsed))
        log(f"⏳ Siguiente comprobación en ~{remaining} s.")
        time.sleep(remaining)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predicción multi-símbolo M5 con auto-BUY si p_up>=0.75, SL/TP y alertas ES+RU."
    )
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
