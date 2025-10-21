import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import time
import argparse
from zoneinfo import ZoneInfo
from typing import Optional, Tuple

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
#      CONFIGURACIÓN BÁSICA
# ===========================
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_M5
TZ = ZoneInfo("Europe/Madrid")
DEBUG = True

# ===========================
#   PARÁMETROS HEDGING+MARTINGALE
# ===========================
# Rejilla simétrica de stops desde un precio ancla.
BASE_LOT          = 0.01          # lote base (capa 0)
MARTI_MULT        = 1.6           # multiplicador por capa (>=1.0)
STEP_PIPS         = 12            # separación de rejilla entre capas
MAX_LAYERS_PER_SIDE = 5           # nº máximo de capas por lado (0..N-1)
USE_HEDGING       = True          # mantiene BUY y SELL simultáneamente

# Gestión de cesta (basket)
BASKET_TP_MONEY   = 2.50          # cerrar todo el símbolo cuando el P/L flotante >= este valor (moneda de la cuenta)
BASKET_TP_PCT     = 0.15          # o cuando P/L >= % balance (ej. 0.15%); si ambas se definen, se usa la primera que se cumpla

# Protección / límites
MAX_TOTAL_VOLUME  = 0.50          # lote total máximo abierto (todas las posiciones del símbolo)
MAX_DRAWDOWN_PCT  = 10.0          # cierra todo si equity cae >10% desde el balance
REBUILD_ON_FLAT   = True          # si no hay posiciones, volver a sembrar rejilla
COMMENT_TAG       = "HMv1"        # etiqueta para comentar órdenes/posiciones del bot

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

def symbol_meta(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError("No symbol info")
    point = info.point
    digits = info.digits
    # pip genérico: en la mayoría de FX con 5/3 dígitos, 1 pip = 10*point
    pip = 10 * point
    return info, point, pip, digits

def account_info():
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("No account info")
    return acc

def now_local():
    return pd.Timestamp.utcnow().tz_localize("UTC").tz_convert(TZ)

# ===========================
#    ÓRDENES / POSICIONES
# ===========================
def total_symbol_positions(symbol: str):
    poss = mt5.positions_get(symbol=symbol)
    if poss is None:
        return []
    return list(poss)

def total_symbol_orders(symbol: str):
    ords = mt5.orders_get(symbol=symbol)
    if ords is None:
        return []
    return list(ords)

def current_floating_pl(symbol: str) -> float:
    pl = 0.0
    for p in total_symbol_positions(symbol):
        pl += p.profit
    return pl

def current_total_volume(symbol: str) -> float:
    vol = 0.0
    for p in total_symbol_positions(symbol):
        vol += p.volume
    return vol

def side_layers_state(symbol: str) -> Tuple[int, int]:
    """
    Devuelve (layers_buy, layers_sell) abiertos (posiciones + órdenes pendientes con nuestro tag).
    Sirve para no exceder MAX_LAYERS_PER_SIDE.
    """
    layers_buy = 0
    layers_sell = 0
    tag = f"{COMMENT_TAG}|"
    for o in total_symbol_orders(symbol):
        if o.comment and o.comment.startswith(tag):
            # Formato: HMv1|side=BUY|layer=2
            if "side=BUY" in o.comment:
                layers_buy = max(layers_buy, parse_layer(o.comment) + 1)
            elif "side=SELL" in o.comment:
                layers_sell = max(layers_sell, parse_layer(o.comment) + 1)
    for p in total_symbol_positions(symbol):
        if p.comment and p.comment.startswith(tag):
            if p.type == mt5.POSITION_TYPE_BUY:
                layers_buy = max(layers_buy, parse_layer(p.comment) + 1)
            elif p.type == mt5.POSITION_TYPE_SELL:
                layers_sell = max(layers_sell, parse_layer(p.comment) + 1)
    return layers_buy, layers_sell

def parse_layer(comment: str) -> int:
    try:
        # espera "...|layer=N"
        parts = comment.split("|")
        for s in parts:
            if s.startswith("layer="):
                return int(s.split("=")[1])
    except Exception:
        return 0
    return 0

def lot_for_layer(layer: int) -> float:
    return round(BASE_LOT * (MARTI_MULT ** layer), 2)

def place_stop(side: str, price: float, layer: int):
    """
    Coloca BUY STOP o SELL STOP con comentario etiquetado para trazar capas.
    Sin SL/TP; la gestión es por cesta.
    """
    _, _, _, digits = symbol_meta(SYMBOL)
    volume = lot_for_layer(layer)
    if volume <= 0:
        return False, "invalid volume"

    if side == "BUY":
        order_type = mt5.ORDER_TYPE_BUY_STOP
    else:
        order_type = mt5.ORDER_TYPE_SELL_STOP

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": round(price, digits),
        "deviation": 20,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
        "comment": f"{COMMENT_TAG}|side={side}|layer={layer}",
    }
    res = mt5.order_send(request)
    if res is None:
        return False, "order_send None"
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        return False, f"retcode={res.retcode}"
    return True, "OK"

def close_all_symbol_positions(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return
    digits = info.digits
    for p in total_symbol_positions(symbol):
        if p.symbol != symbol:
            continue
        if p.type == mt5.POSITION_TYPE_BUY:
            price = tick.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL,
                "position": p.ticket,
                "price": round(price, digits),
                "deviation": 30,
                "comment": f"{COMMENT_TAG}|close_basket",
            }
        else:
            price = tick.ask
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": round(price, digits),
                "deviation": 30,
                "comment": f"{COMMENT_TAG}|close_basket",
            }
        mt5.order_send(req)

def delete_all_symbol_orders(symbol: str):
    for o in total_symbol_orders(symbol):
        if o.symbol != symbol:
            continue
        mt5.order_send({
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": o.ticket,
            "symbol": symbol,
            "comment": f"{COMMENT_TAG}|remove",
        })

# ===========================
#    LÓGICA DE REJILLA
# ===========================
def build_grid(anchor_price: float):
    """
    Coloca una rejilla de BUY STOP por encima y SELL STOP por debajo del ancla.
    Capas: 0..MAX_LAYERS_PER_SIDE-1
    """
    info, point, pip, digits = symbol_meta(SYMBOL)
    placed = 0
    for layer in range(MAX_LAYERS_PER_SIDE):
        offs = (layer + 1) * STEP_PIPS * pip
        buy_price = anchor_price + offs
        sell_price = anchor_price - offs
        ok1, m1 = place_stop("BUY", buy_price, layer)
        ok2, m2 = place_stop("SELL", sell_price, layer)
        if ok1:
            placed += 1
        if ok2:
            placed += 1
    if placed > 0:
        ts = now_local().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            "🧭 Rejilla colocada (Hedging+Martingale)\n\n"
            f"🇪🇸 Ancla: {anchor_price:.5f} | Capas por lado: {MAX_LAYERS_PER_SIDE} | Paso: {STEP_PIPS} pips | Lote base: {BASE_LOT}\n"
            f"🇷🇺 Якорь: {anchor_price:.5f} | Слоёв на сторону: {MAX_LAYERS_PER_SIDE} | Шаг: {STEP_PIPS} пипсов | Базовый лот: {BASE_LOT}\n"
            f"⏰ {ts}"
        )
        send_telegram(msg)

def ensure_grid():
    """
    Asegura que existe rejilla si procede (sin exceder capas).
    Si no hay órdenes/posiciones o REBUILD_ON_FLAT=True, resembrar.
    """
    poss = total_symbol_positions(SYMBOL)
    ords = total_symbol_orders(SYMBOL)

    have_any = (len(poss) + len(ords)) > 0
    if not have_any or (REBUILD_ON_FLAT and len(poss) == 0):
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick:
            return
        anchor = (tick.ask + tick.bid) / 2.0
        delete_all_symbol_orders(SYMBOL)  # limpieza previa
        build_grid(anchor)

def enforce_limits():
    """
    Limita exposición y drawdown. Retorna True si se tomó acción.
    """
    acc = account_info()
    balance = acc.balance
    equity = acc.equity

    # Kill switch por DD
    dd_pct = 0.0 if balance <= 0 else (max(0.0, balance - equity) / balance) * 100.0
    if dd_pct >= MAX_DRAWDOWN_PCT:
        # Cierra todo y borra pendientes
        close_all_symbol_positions(SYMBOL)
        delete_all_symbol_orders(SYMBOL)
        msg = (
            "🛑 Protección activada\n\n"
            f"🇪🇸 Drawdown {dd_pct:.2f}% ≥ {MAX_DRAWDOWN_PCT:.2f}% → se cierran posiciones y órdenes.\n"
            f"🇷🇺 Просадка {dd_pct:.2f}% ≥ {MAX_DRAWDOWN_PCT:.2f}% → закрываем все позиции и ордера."
        )
        send_telegram(msg)
        return True

    # Límite de volumen
    vol = current_total_volume(SYMBOL)
    if vol > MAX_TOTAL_VOLUME:
        # No abrir más: borra órdenes adicionales (dejamos las existentes)
        delete_all_symbol_orders(SYMBOL)
        msg = (
            "⚖️ Límite de exposición\n\n"
            f"🇪🇸 Volumen total {vol:.2f} > máx {MAX_TOTAL_VOLUME:.2f} → se eliminan nuevas órdenes pendientes.\n"
            f"🇷🇺 Общий объём {vol:.2f} > макс {MAX_TOTAL_VOLUME:.2f} → удаляем отложенные ордера."
        )
        send_telegram(msg)
        return True
    return False

def basket_take_profit_reached() -> bool:
    pl = current_floating_pl(SYMBOL)
    acc = account_info()
    cond_money = (BASKET_TP_MONEY is not None) and (pl >= float(BASKET_TP_MONEY))
    cond_pct = False
    if BASKET_TP_PCT is not None and acc and acc.balance > 0:
        cond_pct = (pl >= (acc.balance * (BASKET_TP_PCT / 100.0)))
    return cond_money or cond_pct

def try_close_basket():
    if basket_take_profit_reached():
        pl = current_floating_pl(SYMBOL)
        close_all_symbol_positions(SYMBOL)
        delete_all_symbol_orders(SYMBOL)
        ts = now_local().strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            "✅ Cesta cerrada (beneficio alcanzado)\n\n"
            f"🇪🇸 P/L flotante alcanzado: {pl:.2f}. Se cierra todo y se resembrará la rejilla.\n"
            f"🇷🇺 Достигнута цель по прибыли: {pl:.2f}. Все позиции закрыты, сетка будет создана заново.\n"
            f"⏰ {ts}"
        )
        send_telegram(msg)
        return True
    return False

def notify_new_fills():
    """
    Revisa el historial de posiciones recientes del símbolo y envía aviso de nueva capa (opcional simple).
    """
    # MT5 Python no da push inmediato de llenado. Como opción ligera:
    # resumimos el número de posiciones por lado y lo anunciamos si cambia.
    # (Para algo persistente, guardaríamos estado en disco.)
    pass  # mantener simple para ejemplo

# ===========================
#       LOOP PRINCIPAL
# ===========================
def run_once():
    try:
        init_mt5()
        # 1) Protección y límites
        if enforce_limits():
            return
        # 2) TP de cesta
        if try_close_basket():
            # opcional: resembrar rejilla
            ensure_grid()
            return
        # 3) Asegurar rejilla
        ensure_grid()
        # 4) (Opcional) anunciar nuevas capas llenadas, etc.
        # notify_new_fills()
        # 5) Log informativo
        acc = account_info()
        pl = current_floating_pl(SYMBOL)
        log(f"ℹ️ Balance={acc.balance:.2f} Equity={acc.equity:.2f} PL_flotante({SYMBOL})={pl:.2f}")
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
    parser = argparse.ArgumentParser(description="Hedging+Martingale MT5 (rejilla simétrica con TP por cesta, ES+RU).")
    parser.add_argument("--every-min", type=int, default=1,
                        help="Intervalo de comprobación en minutos (por defecto: 1).")
    parser.add_argument("--once", action="store_true",
                        help="Ejecuta una sola vez y termina.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.once:
        run_once()
    else:
        run_loop(args.every_min)
