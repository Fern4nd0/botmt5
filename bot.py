import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import time
import math

# ================== CREDENCIALES (tuyas) ==================
MT5_SERVER   = "ForexClubBY-MT5 Demo Server"
MT5_LOGIN    = 520002796
MT5_PASSWORD = "PclNMY2*"
# (Opcional) si en Mac necesitas ruta explícita al terminal:
PATH_TO_TERMINAL = None
# Ejemplo:
# PATH_TO_TERMINAL = "/Applications/MetaTrader 5.app/Contents/Resources/drive_c/Program Files/MetaTrader 5/terminal64.exe"
# ==========================================================

SYMBOL_HINT = "USDJPY"            # patrón base (el bot buscará sufijos: USDJPY*)
TIMEFRAME_EMA = mt5.TIMEFRAME_M5  # timeframe para EMAs (intradía)
TIMEFRAME_RSI = mt5.TIMEFRAME_M15 # timeframe para RSI
TIMEFRAME_ATR = mt5.TIMEFRAME_H1  # timeframe para ATR
EMA_FAST = 50
EMA_SLOW = 200
RSI_LEN = 14
ATR_LEN = 14

def init_mt5():
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init error: {mt5.last_error()}")
    # Login explícito (aunque el terminal ya esté logueado)
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Login error: {mt5.last_error()}")

def find_symbol(base: str) -> str:
    cands = mt5.symbols_get(base + "*")
    if not cands:
        info = mt5.symbol_info(base)
        if info:
            cands = [info]
    if not cands:
        raise RuntimeError(f"No encontré símbolos para patrón '{base}'")
    # escoge visible y con trading habilitado
    def score(s):
        sc = 0
        if s.visible: sc += 2
        if s.trade_mode in (mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_LONGONLY, mt5.SYMBOL_TRADE_MODE_SHORTONLY):
            sc += 1
        if s.name == base: sc += 1
        return sc
    sym = sorted(cands, key=score, reverse=True)[0].name
    if not mt5.symbol_info(sym).visible:
        mt5.symbol_select(sym, True)
    return sym

def pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError("symbol_info None")
    # 3 o 5 dígitos → 1 pip = 10*point (ej. JPY: 3 dígitos)
    return (10 * info.point) if info.digits in (3,5) else info.point

def get_tick(symbol: str):
    t = mt5.symbol_info_tick(symbol)
    if not t:
        raise RuntimeError("Sin tick")
    return t

def copy_rates_df(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError("copy_rates_from_pos vacío")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series_close: pd.Series, length: int) -> pd.Series:
    delta = series_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-10))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int) -> pd.Series:
    # df con columnas: time, open, high, low, close
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, min_periods=length).mean()

def daily_change_pct(symbol: str, mid_price: float) -> float:
    # abre barras diarias y calcula variación desde la apertura de hoy (UTC)
    dfd = copy_rates_df(symbol, mt5.TIMEFRAME_D1, 3)
    today_open = dfd.iloc[-1]["open"]
    return (mid_price - today_open) / today_open * 100.0

def format_price(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"

def analyze(symbol: str) -> str:
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError("No symbol_info")
    digits = info.digits
    pip = pip_size(symbol)
    t = get_tick(symbol)
    mid = (t.bid + t.ask) / 2.0
    spread_pips = (t.ask - t.bid) / pip

    # Datos para indicadores
    df_ema = copy_rates_df(symbol, TIMEFRAME_EMA, 400)
    df_rsi = copy_rates_df(symbol, TIMEFRAME_RSI, 200)
    df_atr = copy_rates_df(symbol, TIMEFRAME_ATR, 300)

    df_ema["ema_fast"] = ema(df_ema["close"], EMA_FAST)
    df_ema["ema_slow"] = ema(df_ema["close"], EMA_SLOW)
    last_ema = df_ema.iloc[-1]
    prev_ema = df_ema.iloc[-2]
    bull_cross = (prev_ema.ema_fast <= prev_ema.ema_slow) and (last_ema.ema_fast > last_ema.ema_slow)
    bear_cross = (prev_ema.ema_fast >= prev_ema.ema_slow) and (last_ema.ema_fast < last_ema.ema_slow)
    ema_trend = "alcista" if last_ema.ema_fast > last_ema.ema_slow else "bajista"

    df_rsi["rsi"] = rsi(df_rsi["close"], RSI_LEN)
    rsi_last = float(df_rsi["rsi"].iloc[-1])

    df_atr["atr"] = atr(df_atr[["time","open","high","low","close"]], ATR_LEN)
    atr_last = float(df_atr["atr"].iloc[-1])

    dchg = daily_change_pct(symbol, mid)

    # Resistencias/soportes recientes (H1 últimos 72 velas)
    df_h1 = df_atr  # ya es H1
    recent = df_h1.tail(72)
    swing_high = recent["high"].rolling(10).max().iloc[-1]
    swing_low  = recent["low"].rolling(10).min().iloc[-1]

    # Veredicto simple
    veredictos = []
    veredictos.append(f"EMAs ({EMA_FAST}/{EMA_SLOW}) → {ema_trend}")
    if bull_cross: veredictos.append("Cruce alcista reciente")
    if bear_cross: veredictos.append("Cruce bajista reciente")
    if rsi_last > 70: veredictos.append("RSI sobrecompra (>70)")
    elif rsi_last < 30: veredictos.append("RSI sobreventa (<30)")
    else: veredictos.append("RSI neutral")

    report = []
    now = datetime.now(timezone.utc).astimezone()
    report.append(f"== USDJPY @ {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ==")
    report.append(f"Símbolo detectado: {symbol}")
    report.append(f"Bid/Ask: {format_price(t.bid, digits)} / {format_price(t.ask, digits)}  | Spread: {spread_pips:.2f} pips")
    report.append(f"Cambio diario: {dchg:+.2f}%")
    report.append(f"EMA{EMA_FAST}/{EMA_SLOW} (M5): {last_ema.ema_fast:.3f} / {last_ema.ema_slow:.3f}  → {ema_trend}")
    report.append(f"RSI{RSI_LEN} (M15): {rsi_last:.1f}")
    report.append(f"ATR{ATR_LEN} (H1): {atr_last:.3f}")
    report.append(f"Zona reciente (H1): Resistencia≈ {format_price(swing_high, digits)} | Soporte≈ {format_price(swing_low, digits)}")
    report.append("Veredicto: " + " | ".join(veredictos))
    return "\n".join(report)

def main():
    init_mt5()
    symbol = find_symbol(SYMBOL_HINT)
    print(analyze(symbol))

if __name__ == "__main__":
    main()
