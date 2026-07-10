import os
import sys
import json
import pandas as pd

# Add root directory to sys.path
ROOT_DIR = r"c:\Users\AlvaroJ\Documents\Antigravity Projects\agente_ia_movil"
sys.path.insert(0, ROOT_DIR)

from modulos import sensor, cerebro, actuador, reportador
from modulos.poda_alfa_beta import EvaluadorPodaAlfaBeta, EstadoInventario

# We will run the patched initialization logic here to see what the JSON looks like.
def test_patched_json():
    print("Loading data...")
    df_ventas_full = sensor.cargar_ventas(os.path.join(ROOT_DIR, "datos", "ventas_datasL.csv"))
    df_inventario_full = sensor.cargar_inventario(os.path.join(ROOT_DIR, "datos", "inventario_datasL.csv"))

    # Training range
    df_ventas = df_ventas_full[df_ventas_full["fecha"] <= pd.to_datetime("2026-05-31")].copy()
    df_inventario = df_inventario_full[df_inventario_full["fecha"] <= pd.to_datetime("2026-05-31")].copy()
    ventas_diarias = sensor.obtener_ventas_diarias_completas(df_ventas)

    df_ventas_copia = df_ventas.copy()
    df_ventas_copia["fecha_norm"] = df_ventas_copia["fecha_hora"].dt.normalize()

    df_feriados = df_ventas_copia[df_ventas_copia["es_feriado"]][["fecha_norm", "tipo_feriado"]].drop_duplicates()
    df_feriados = df_feriados.rename(columns={"fecha_norm": "ds", "tipo_feriado": "holiday"})

    df_eventos = df_ventas_copia[df_ventas_copia["evento_especial"] != "ninguno"][["fecha_norm", "evento_especial"]].drop_duplicates()
    df_eventos = df_eventos.rename(columns={"fecha_norm": "ds", "evento_especial": "holiday"})

    df_festivos = pd.concat([df_feriados, df_eventos]).drop_duplicates().reset_index(drop=True)

    print("Training models...")
    agente_cerebro = cerebro.CerebroPredictivo()
    productos_ids = df_inventario["producto_id"].unique()
    predicciones = {}
    patrones_horarios = {}

    for prod_id in productos_ids:
        pred = agente_cerebro.predecir_demanda_diaria(
            ventas_diarias, prod_id, dias_a_predecir=110, df_festivos=df_festivos
        )
        predicciones[prod_id] = pred
        agente_cerebro.entrenar_modelo_horario(df_ventas, prod_id)
        patron = agente_cerebro.predecir_patron_horario(prod_id, pd.to_datetime("2026-06-01"), es_feriado=False)
        patrones_horarios[prod_id] = patron

    print("Bayesian Network...")
    red_bayesiana = cerebro.RedBayesianaMixta()
    observaciones_bn = red_bayesiana.preparar_datos_entrenamiento(df_ventas, df_inventario)
    red_bayesiana.aprender_parametros(observaciones_bn)

    # HERE IS THE PATCH:
    # Filter df_inventario to ONLY the latest date for current alerts evaluation!
    print("Evaluating current alerts (PATCHED)...")
    actuador_opt = actuador.ActuadorOptimizado()
    alertas = []
    
    ultima_fecha_entrenamiento = df_inventario["fecha"].max()
    df_inventario_actual = df_inventario[df_inventario["fecha"] == ultima_fecha_entrenamiento].copy()
    print(f"Latest training inventory date: {ultima_fecha_entrenamiento}")
    
    for _, fila in df_inventario_actual.iterrows():
        prod_id = fila["producto_id"]
        pred_prod = predicciones[prod_id]
        patron_prod = patrones_horarios[prod_id]
        al_prod = actuador_opt.evaluar_producto(fila, pred_prod, patron_prod, [], red_bayesiana=red_bayesiana)
        alertas.extend(al_prod)

    # Construct state
    STATE = {
        "df_ventas": df_ventas_full,
        "df_inventario": df_inventario_full, # Keep full for simulation graphs
        "ventas_diarias": ventas_diarias,
        "df_festivos": df_festivos,
        "agente_cerebro": agente_cerebro,
        "red_bayesiana": red_bayesiana,
        "predicciones": predicciones,
        "patrones_horarios": patrones_horarios,
        "alertas": alertas,
        "actuador_opt": actuador_opt
    }

    # Now run the dashboard endpoint logic
    print("Running dashboard logic...")
    from datetime import datetime, timedelta
    
    alertas_rojas = 0
    alertas_amarillas = 0
    dinero_en_riesgo_vencimiento = 0.0

    # For mapping prices, use the latest row for each product
    df_inv_latest = df_inventario_full.sort_values("fecha").drop_duplicates(subset=["producto_id"], keep="last")
    precios_dict = df_inv_latest.set_index("producto_id")["precio_unitario"].to_dict()
    costo_dict = df_inv_latest.set_index("producto_id")["costo_compra"].to_dict()

    for al in alertas:
        grav = al.get("gravedad", "BAJA")
        tipo = al.get("tipo")
        
        if tipo in ["RIESGO_QUIEBRE", "RIESGO_VENCIMIENTO", "RIESGO_BAYESIANO"]:
            if grav == "ALTA":
                alertas_rojas += 1
            elif grav == "MEDIA":
                alertas_amarillas += 1
        
        if tipo == "RIESGO_VENCIMIENTO":
            prod_id = al.get("producto_id")
            merma_est = al.get("merma_estimada", 0)
            pu = precios_dict.get(prod_id, 0.0)
            costo_u = costo_dict.get(prod_id, pu * 0.7)
            dinero_en_riesgo_vencimiento += float(merma_est * costo_u)

    dinero_en_riesgo_vencimiento = round(dinero_en_riesgo_vencimiento, 2)

    recomendaciones_compra = []
    alertas_quiebre = [al for al in alertas if al.get("tipo") in ["RIESGO_QUIEBRE", "RIESGO_BAYESIANO"]]
    prod_compra_añadidos = set()

    for al in alertas_quiebre:
        prod_id = al.get("producto_id")
        if prod_id in prod_compra_añadidos:
            continue
        prod_compra_añadidos.add(prod_id)
        
        nombre = al.get("producto_nombre", "Producto")
        grav = al.get("gravedad", "MEDIA")
        color_alerta = "rojo" if grav == "ALTA" else "amarillo"
        cant = al.get("cantidad_pedido", 0)
        if cant == 0:
            cant = int(df_inv_latest.set_index("producto_id")["cantidad_minima_pedido"].to_dict().get(prod_id, 12))
            
        motivo = al.get("descripcion", "Stock bajo para cubrir la demanda estimada.")
        if "lead time" in motivo.lower() or "stock de seguridad" in motivo.lower():
            motivo = "El stock actual es insuficiente para cubrir las ventas de los próximos días."
        elif "red bayesiana" in motivo.lower():
            partes = motivo.split("causas activas:")
            if len(partes) > 1:
                causas = partes[1].strip().rstrip(".")
                causas_limpias = causas.replace("es fin semana", "fin de semana").replace("demanda alta", "demanda muy alta").replace("retraso proveedor", "demora del proveedor")
                motivo = f"Hay riesgo de quedarse sin stock debido a: {causas_limpias}."
            else:
                motivo = "Existe riesgo de quedarse sin stock por condiciones de alta demanda o feriados."
        
        sugerencia = f"Se recomienda realizar un pedido de {cant} unidades."
        
        recomendaciones_compra.append({
          "producto_id": prod_id,
          "producto_nombre": nombre,
          "prioridad": grav,
          "color_alerta": color_alerta,
          "motivo": motivo,
          "sugerencia": sugerencia
        })

    if not recomendaciones_compra:
        recomendaciones_compra.append({
          "producto_id": "none",
          "producto_nombre": "Todos los productos",
          "prioridad": "BAJA",
          "color_alerta": "verde",
          "motivo": "Todos los niveles de inventario están en rango seguro para los próximos días.",
          "sugerencia": "No se requieren compras de emergencia."
        })

    alertas_vencimiento = []
    alertas_vence = [al for al in alertas if al.get("tipo") == "RIESGO_VENCIMIENTO"]
    for al in alertas_vence:
        prod_id = al.get("producto_id")
        nombre = al.get("producto_nombre", "Producto")
        grav = al.get("gravedad", "MEDIA")
        color_alerta = "rojo" if grav == "ALTA" else "amarillo"
        merma_est = al.get("merma_estimada", 0)
        precio_oferta = al.get("precio_oferta", 0.0)
        
        fila_prod = df_inv_latest[df_inv_latest["producto_id"] == prod_id]
        vida_util = int(fila_prod.iloc[0].get("vida_util_dias", 5)) if not fila_prod.empty else 5

        alertas_vencimiento.append({
          "producto_id": prod_id,
          "producto_nombre": nombre,
          "color_alerta": color_alerta,
          "dias_para_vencer": max(1, vida_util),
          "cantidad_en_riesgo": int(merma_est),
          "accion_sugerida": f"Aplicar promoción. Vender a S/. {precio_oferta:.2f} para liquidar stock."
        })

    # Output JSON
    result = {
        "resumen_hoy": {
            "alertas_rojas": alertas_rojas,
            "alertas_amarillas": alertas_amarillas,
            "dinero_en_riesgo_vencimiento": dinero_en_riesgo_vencimiento
        },
        "recomendaciones_compra": recomendaciones_compra,
        "alertas_vencimiento": alertas_vencimiento
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "dashboard_patched_response.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print(f"JSON response saved to: {out_path}")
    print("\n--- Patched JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_patched_json()
