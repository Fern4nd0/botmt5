import MetaTrader5 as mt5
import pandas as pd

# ======== CREDENCIALES ========
MT5_SERVER   = "ForexClubBY-MT5 Demo Server"
MT5_LOGIN    = 520002796
MT5_PASSWORD = "PclNMY2*"
PATH_TO_TERMINAL = None
# ==============================

def init_mt5():
    ok = mt5.initialize(PATH_TO_TERMINAL) if PATH_TO_TERMINAL else mt5.initialize()
    if not ok:
        raise RuntimeError(f"Error al inicializar MT5: {mt5.last_error()}")
    if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        raise RuntimeError(f"Error de login: {mt5.last_error()}")
    print("✅ Conectado correctamente a MetaTrader 5\n")

def revisar_posiciones():
    posiciones = mt5.positions_get()
    if posiciones is None:
        print(f"❌ Error al obtener posiciones: {mt5.last_error()}")
        return
    if len(posiciones) == 0:
        print("ℹ️ No tienes posiciones abiertas.")
        return

    df = pd.DataFrame(list(posiciones), columns=posiciones[0]._asdict().keys())
    df["type"] = df["type"].map({mt5.ORDER_TYPE_BUY: "BUY", mt5.ORDER_TYPE_SELL: "SELL"})

    print("== POSICIONES ABIERTAS ==")
    for _, pos in df.iterrows():
        print(f"{pos['symbol']:10} | {pos['type']:4} {pos['volume']:.2f} | "
              f"Entrada: {pos['price_open']:.5f} | Actual: {pos['price_current']:.5f} | "
              f"SL: {pos['sl'] if pos['sl'] != 0 else '-'} | TP: {pos['tp'] if pos['tp'] != 0 else '-'} | "
              f"Beneficio: {pos['profit']:.2f}")
    print(f"Total posiciones: {len(df)} | Beneficio total: {df['profit'].sum():.2f}\n")

def revisar_ordenes_pendientes():
    ordenes = mt5.orders_get()
    if ordenes is None:
        print(f"❌ Error al obtener órdenes: {mt5.last_error()}")
        return
    if len(ordenes) == 0:
        print("ℹ️ No tienes órdenes pendientes.")
        return

    df = pd.DataFrame(list(ordenes), columns=ordenes[0]._asdict().keys())
    tipos = {
        mt5.ORDER_TYPE_BUY_LIMIT: "BUY LIMIT",
        mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
        mt5.ORDER_TYPE_BUY_STOP: "BUY STOP",
        mt5.ORDER_TYPE_SELL_STOP: "SELL STOP",
        mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY STOP LIMIT",
        mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL STOP LIMIT",
    }
    df["type"] = df["type"].map(tipos)

    print("== ÓRDENES PENDIENTES ==")
    for _, o in df.iterrows():
        print(f"{o['symbol']:10} | {o['type']:13} | Volumen: {o['volume_initial']:.2f} | "
              f"Precio: {o['price_open']:.5f} | SL: {o['sl'] if o['sl'] != 0 else '-'} | "
              f"TP: {o['tp'] if o['tp'] != 0 else '-'} | Ticket: {o['ticket']}")
    print(f"Total órdenes pendientes: {len(df)}\n")

def main():
    init_mt5()
    revisar_posiciones()
    revisar_ordenes_pendientes()
    mt5.shutdown()

if __name__ == "__main__":
    main()
