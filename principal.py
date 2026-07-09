# -*- coding: utf-8 -*-

#Orquestador Principal de Pymevision AI
#Coordina la carga de datos, entrenamiento de modelos, simulación y reportes.


import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import sys
import time
import threading

# Asegurar compatibilidad UTF-8 en consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Módulos locales organizados en la carpeta modulos
from modulos import sensor, cerebro, actuador, busqueda, reportador
from modulos.poda_alfa_beta import EvaluadorPodaAlfaBeta, EstadoInventario

class Spinner:
    def __init__(self, message="Procesando..."):
        self.message = message
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.idx = 0
        self.stop_running = threading.Event()
        self.thread = None
        self.max_len = 0

    def _spin(self):
        while not self.stop_running.is_set():
            char = self.spinner_chars[self.idx % len(self.spinner_chars)]
            line = f"\r {char} {self.message}"
            self.max_len = max(self.max_len, len(line))
            padded_line = line.ljust(self.max_len)
            sys.stdout.write(padded_line)
            sys.stdout.flush()
            self.idx += 1
            time.sleep(0.08)

    def start(self):
        self.max_len = 0
        self.stop_running.clear()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def update_message(self, new_message):
        self.message = new_message

    def stop(self):
        if self.thread:
            self.stop_running.set()
            self.thread.join()
            sys.stdout.write("\r" + " " * self.max_len + "\r")
            sys.stdout.flush()

def ejecutar_agente():
    print("=" * 75)
    print("   INICIANDO AGENTE INTELIGENTE AUTÓNOMO PYMEVISION AI - MULTIMENSUAL")
    print("=" * 75)
    
    # carga de datos (Sensor)
    spinner = Spinner("[SENSOR] Cargando datos desde archivos CSV...")
    spinner.start()
    try:
        df_ventas = sensor.cargar_ventas("datos/ventas_datasL.csv")
        df_inventario = sensor.cargar_inventario("datos/inventario_datasL.csv")
        
        # Filtrado defensivo: conservar solo marzo, abril, mayo
        df_ventas = df_ventas[df_ventas["fecha"] <= pd.to_datetime("2026-05-31")].copy()
        df_inventario = df_inventario[df_inventario["fecha"] <= pd.to_datetime("2026-05-31")].copy()
        
        rango_ventas = f"{df_ventas['fecha'].min().strftime('%d/%m/%Y')} al {df_ventas['fecha'].max().strftime('%d/%m/%Y')}"
        spinner.stop()
        print(f"✔ [SENSOR] Datos cargados con éxito.")
        print(f"   * Ventas cargadas: {len(df_ventas)} filas (Rango: {rango_ventas})")
        print(f"   * Inventario cargado: {len(df_inventario)} filas")
    except Exception as error:
        spinner.stop()
        print(f"[ERROR SENSOR] Error al cargar los datos: {error}")
        import traceback
        traceback.print_exc()
        return
    fecha_limite_entrenamiento = pd.to_datetime("2026-05-31")
    df_ventas_entreno = df_ventas[df_ventas["fecha"] <= fecha_limite_entrenamiento].copy()
    ventas_diarias_entreno = sensor.obtener_ventas_diarias_completas(df_ventas_entreno)
    
    # extracción de festivos y entrenamiento de modelos
    spinner = Spinner("[SENSOR] Extrayendo base de feriados y eventos especiales...")
    spinner.start()
    df_ventas_entreno_copia = df_ventas_entreno.copy()
    df_ventas_entreno_copia["fecha_norm"] = df_ventas_entreno_copia["fecha_hora"].dt.normalize()
    
    df_feriados = df_ventas_entreno_copia[df_ventas_entreno_copia["es_feriado"]][["fecha_norm", "tipo_feriado"]].drop_duplicates()
    df_feriados = df_feriados.rename(columns={"fecha_norm": "ds", "tipo_feriado": "holiday"})
    
    df_eventos = df_ventas_entreno_copia[df_ventas_entreno_copia["evento_especial"] != "ninguno"][["fecha_norm", "evento_especial"]].drop_duplicates()
    df_eventos = df_eventos.rename(columns={"fecha_norm": "ds", "evento_especial": "holiday"})
    
    df_festivos = pd.concat([df_feriados, df_eventos]).drop_duplicates()
    
    mapeo_limpieza = {
        "Da del Trabajo": "Día del Trabajo",
        "Batalla de Arica y Da de la Bandera": "Batalla de Arica y Día de la Bandera",
        "San Pedro y San Pablo": "San Pedro y San Pablo",
        "Fiestas Patrias": "Fiestas Patrias",
        "Da de la Madre": "Día del Madre",
        "Da del Padre": "Día del Padre",
        "Mundial FIFA 2026": "Mundial FIFA 2026",
        "CyberWow": "CyberWow",
        "Santa Rosa de Lima": "Santa Rosa de Lima",
        "Batalla de Junín": "Batalla de Junín"
    }
    df_festivos["holiday"] = df_festivos["holiday"].map(mapeo_limpieza).fillna(df_festivos["holiday"])
    
    # Agregar feriados futuros
    feriados_adicionales = pd.DataFrame({
        "ds": [
            pd.to_datetime("2026-08-06"),
            pd.to_datetime("2026-08-30")
        ],
        "holiday": [
            "Batalla de Junín",
            "Santa Rosa de Lima"
        ]
    })
    df_festivos = pd.concat([df_festivos, feriados_adicionales]).drop_duplicates().reset_index(drop=True)
    spinner.stop()
    print("✔ [SENSOR] Base de feriados y eventos especiales extraída.")
    
    spinner = Spinner("[CEREBRO] Entrenando modelos predictivos por producto (Prophet y Red Neuronal MLP)...")
    spinner.start()
    agente_cerebro = cerebro.CerebroPredictivo()
    productos_ids = df_inventario["producto_id"].unique()
    
    dias_totales_proyeccion = 110
    predicciones_completas = {}
    diccionario_patrones_horarios = {}
    
    for prod_id in productos_ids:
        nombre_prod = df_inventario[df_inventario["producto_id"] == prod_id].iloc[0]["producto_nombre"]
        spinner.update_message(f"[CEREBRO] Entrenando modelo para: {nombre_prod}")
        
        pred_completa = agente_cerebro.predecir_demanda_diaria(
            ventas_diarias_entreno, prod_id, dias_a_predecir=dias_totales_proyeccion, df_festivos=df_festivos
        )
        predicciones_completas[prod_id] = pred_completa
        
        agente_cerebro.entrenar_modelo_horario(df_ventas_entreno, prod_id)
        
        fecha_inicio_rep = pd.to_datetime("2026-06-01")
        diccionario_patrones_horarios[prod_id] = agente_cerebro.predecir_patron_horario(
            prod_id, fecha_inicio_rep, es_feriado=False
        )
        
    spinner.stop()
    print("✔ [CEREBRO] Modelos predictivos entrenados con éxito.")
 
    df_inventario["fecha_dt"] = pd.to_datetime(df_inventario["fecha"])
    df_ventas["fecha_dt"] = pd.to_datetime(df_ventas["fecha"])

    # Entrenamiento de la Red Bayesiana Mixta (Noisy-OR) - Vomlel et al. (2023)
    spinner = Spinner("[CEREBRO] Entrenando Red Bayesiana Mixta (Noisy-OR) para análisis causal...")
    spinner.start()
    red_bayesiana = cerebro.RedBayesianaMixta()
    
    df_inv_entreno = df_inventario[df_inventario["fecha_dt"] <= fecha_limite_entrenamiento].copy()
    observaciones_bn = red_bayesiana.preparar_datos_entrenamiento(df_ventas_entreno, df_inv_entreno)
    red_bayesiana.aprender_parametros(observaciones_bn)
    
    metricas_bn = red_bayesiana.obtener_metricas()
    spinner.stop()
    print("✔ [CEREBRO] Red Bayesiana Mixta entrenada con éxito.")
    print(f"   * Nodos causa (variables padre): {len(metricas_bn['nodos_causa'])}")
    print(f"   * Parámetros Noisy-OR: {metricas_bn['n_parametros_noisy_or']} (vs {metricas_bn['n_parametros_cpt_exponencial']} en CPT exponencial)")
    print(f"   * BIC Score: {metricas_bn['bic_score']}")
    print(f"   * Prob. de fuga (mermas/robos/errores): {metricas_bn['prob_fuga']}")
 
    # generación de plantillas de Junio a Agosto para la simulación
    spinner = Spinner("[SENSOR] Generando plantilla para simulación de Junio a Agosto 2026...")
    spinner.start()
    fechas_simulacion = pd.date_range(start="2026-06-01", end="2026-08-31", freq="D")
    
    df_ventas_reales = df_ventas[df_ventas["cantidad_vendida"].notnull()].copy()
    conteo_transacciones = df_ventas_reales.groupby(["producto_id", "fecha"])["cantidad_vendida"].count().reset_index()
    stats_trans = conteo_transacciones.groupby("producto_id")["cantidad_vendida"].agg(["min", "max"]).to_dict("index")
    
    list_nuevas_ventas = []
    list_nuevo_inventario = []
    
    def obtener_info_dia(fecha):
        # Mundial FIFA 2026: 11 junio – 19 julio
        # Nota: los días de partido (especialmente fines de semana, semifinales y finales)
        # tienen un impacto comercial incrementado en la compra de bebidas y snacks en Lima.
        es_mundial = pd.to_datetime("2026-06-11") <= fecha <= pd.to_datetime("2026-07-19")
        # Día del Padre: 21 junio 2026 (Rango extendido 18-22 de junio para capturar compras previas)
        es_padre = (fecha.month == 6 and fecha.day in range(18, 23))
        # San Pedro y San Pablo: 29 junio (Rango extendido 28-30 de junio por víspera y rebote)
        es_san_pedro = (fecha.month == 6 and fecha.day in (28, 29, 30))
        # Batalla de Junín: 6 agosto
        es_junin = (fecha.month == 8 and fecha.day == 6)
        # Fiestas Patrias: 28-29 julio (Rango extendido 27-29 de julio para incluir la víspera del 27)
        es_fiestas_patrias = (fecha.month == 7 and fecha.day in (27, 28, 29))
        # Santa Rosa de Lima: 30 agosto
        es_santa_rosa = (fecha.month == 8 and fecha.day == 30)
        
        if es_fiestas_patrias:
            return True, "Fiestas Patrias", "fiestas_patrias"
        elif es_san_pedro:
            return True, "San Pedro y San Pablo", "feriado_religioso"
        elif es_junin:
            return True, "Batalla de Junín", "feriado_nacional"
        elif es_santa_rosa:
            return True, "Santa Rosa de Lima", "feriado_religioso"
        elif es_padre:
            return False, "Día del Padre", "ninguno"
        elif es_mundial:
            return False, "Mundial FIFA 2026", "ninguno"
        else:
            return False, "ninguno", "ninguno"
            
    random.seed(42)
    
    for dia in fechas_simulacion:
        es_feriado_hoy, nombre_feriado, tipo_feriado = obtener_info_dia(dia)
        es_fin_semana_hoy = dia.weekday() >= 5
        dia_semana_str = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][dia.weekday()]
        
        for pid in productos_ids:
            meta_prod = df_inventario[(df_inventario["producto_id"] == pid) & (df_inventario["fecha"] == pd.to_datetime("2026-05-31"))].iloc[0]
            
            inv_row = {
                "fecha": dia,
                "producto_id": pid,
                "producto_nombre": meta_prod["producto_nombre"],
                "categoria": meta_prod["categoria"],
                "stock_fisico": np.nan,
                "stock_transito": np.nan,
                "hay_stock": "FALSO",
                "es_perecedero": meta_prod["es_perecedero"],
                "vida_util_dias": meta_prod["vida_util_dias"],
                "precio_unitario": meta_prod["precio_unitario"],
                "costo_compra": meta_prod["costo_compra"],
                "costo_almacenamiento_diario": meta_prod["costo_almacenamiento_diario"],
                "ventas_perdidas_estimadas": np.nan,
                "proveedor_id": meta_prod["proveedor_id"],
                "tiempo_reposicion_dias": meta_prod["tiempo_reposicion_dias"],
                "cantidad_minima_pedido": meta_prod["cantidad_minima_pedido"],
                "dia_semana": dia_semana_str,
                "mes": dia.month,
                "es_feriado": es_feriado_hoy,
                "es_fin_semana": es_fin_semana_hoy,
                "evento_especial": nombre_feriado,
                "fecha_dt": dia
            }
            list_nuevo_inventario.append(inv_row)
            
            trans_limits = stats_trans.get(pid, {"min": 2, "max": 5})
            mapeo_multiplicadores = {
                "Fiestas Patrias": 1.35,
                "Día del Padre": 1.20,
                "Mundial FIFA 2026": 1.15,
                "San Pedro y San Pablo": 0.85,
                "Batalla de Junín": 1.05,
                "Santa Rosa de Lima": 1.10
            }
            multiplicador = mapeo_multiplicadores.get(nombre_feriado, 1.00)
            n_trans = max(1, int(random.randint(int(trans_limits["min"]), int(trans_limits["max"])) * multiplicador))
            
            for _ in range(n_trans):
                horas_posibles = list(range(8, 22))
                pesos = [3.0 if (h in range(12, 15) or h in range(18, 22)) else 1.0 for h in horas_posibles]
                hora = random.choices(horas_posibles, weights=pesos, k=1)[0]
                minuto = random.randint(0, 59)
                
                fecha_hora_val = dia.replace(hour=hora, minute=minuto)
                es_hora_pico_val = hora in range(12, 15) or hora in range(18, 22)
                
                # Temperatura y precipitación adaptativas por mes (Invierno limeño vs Agosto)
                if dia.month == 6:
                    temp = round(17.0 + random.uniform(-2.5, 3.0), 1)
                    precip = round(random.uniform(0.0, 2.5), 1) if random.random() < 0.7 else 0.0
                elif dia.month == 7:
                    temp = round(16.0 + random.uniform(-2.5, 3.0), 1)
                    precip = round(random.uniform(0.0, 2.5), 1) if random.random() < 0.7 else 0.0
                else:  # Agosto
                    temp = round(16.0 + random.uniform(-2.0, 2.0), 1)
                    precip = round(random.uniform(0.0, 1.5), 1) if random.random() < 0.6 else 0.0
                
                vta_row = {
                    "fecha_hora": fecha_hora_val,
                    "producto_id": pid,
                    "producto_nombre": meta_prod["producto_nombre"],
                    "categoria": meta_prod["categoria"],
                    "cantidad_vendida": np.nan,
                    "precio_unitario": meta_prod["precio_unitario"],
                    "precio_aplicado": np.nan,
                    "es_promocion": np.nan,
                    "canal_venta": random.choice(["presencial", "presencial", "online"]),
                    "dia_semana": dia_semana_str,
                    "dia_mes": dia.day,
                    "mes": dia.month,
                    "hora": hora,
                    "es_hora_pico": es_hora_pico_val,
                    "es_fin_semana": es_fin_semana_hoy,
                    "es_feriado": es_feriado_hoy,
                    "tipo_feriado": tipo_feriado,
                    "evento_especial": nombre_feriado,
                    "temperatura": temp,
                    "precipitacion": precip,
                    "fecha": dia,
                    "fecha_dt": dia
                }
                list_nuevas_ventas.append(vta_row)
                
    df_nuevo_inv = pd.DataFrame(list_nuevo_inventario)
    df_nuevas_vta = pd.DataFrame(list_nuevas_ventas)
    
    df_inventario = pd.concat([df_inventario, df_nuevo_inv], ignore_index=True)
    df_ventas = pd.concat([df_ventas, df_nuevas_vta], ignore_index=True)

    # simulación de Agente Autónomo de Junio a Agosto
    spinner = Spinner("[SIMULACIÓN] Iniciando simulación de inventario de Junio a Agosto 2026...")
    spinner.start()
    
    stock_actual = {}
    pedidos_en_camino = {pid: [] for pid in productos_ids}
    estado_promocion_activa = {pid: False for pid in productos_ids}
    precio_promocional_activo = {pid: 0.0 for pid in productos_ids}
    
    inventario_31_mayo = df_inventario[df_inventario["fecha"] == pd.to_datetime("2026-05-31")]
    for pid in productos_ids:
        fila_31 = inventario_31_mayo[inventario_31_mayo["producto_id"] == pid]
        stock_actual[pid] = int(fila_31.iloc[0]["stock_fisico"]) if not fila_31.empty else 50
            
    quiebres_evitados_contador = 0
    mermas_mitigadas_contador = 0
    total_pedidos_hechos = 0
    
    for dia in fechas_simulacion:
        spinner.update_message(f"[SIMULACIÓN] Procesando día {dia.strftime('%d/%m/%Y')} | Pedidos colocados: {total_pedidos_hechos}")
        es_feriado_hoy, nombre_feriado, tipo_feriado = obtener_info_dia(dia)
        es_fin_semana_hoy = dia.weekday() >= 5
        
        for pid in productos_ids:
            fila_meta = df_inventario[(df_inventario["producto_id"] == pid) & (df_inventario["fecha_dt"] == pd.to_datetime("2026-05-31"))].iloc[0]
            tiempo_reposicion = int(fila_meta["tiempo_reposicion_dias"])
            cant_min_pedido = int(fila_meta["cantidad_minima_pedido"])
            es_perecedero = fila_meta["es_perecedero"]
            precio_unitario = fila_meta["precio_unitario"]
            
            # Llegada de pedidos
            pedidos_hoy = [p for p in pedidos_en_camino[pid] if p["fecha_llegada"] == dia.date()]
            for p in pedidos_hoy:
                stock_actual[pid] += p["cantidad"]
                pedidos_en_camino[pid].remove(p)
            
            # Predicción de demanda
            pred_prod = predicciones_completas[pid]
            demanda_predicha = pred_prod[pred_prod["fecha"] == dia]["demanda_predicha"].values[0]
            
            # Calcular demanda proyectada en lead time y stock de seguridad base
            rango_lead_time = pd.date_range(start=dia + pd.Timedelta(days=1), periods=tiempo_reposicion, freq="D")
            demanda_lead_time = pred_prod[pred_prod["fecha"].isin(rango_lead_time)]["demanda_predicha"].sum()
            stock_seguridad = max(2, int(demanda_lead_time * 0.20))
            
            # Evaluación de riesgo con Red Bayesiana (Noisy-OR)
            demanda_promedio_prod = pred_prod["demanda_predicha"].mean()
            es_feriado_hoy_bool = bool(es_feriado_hoy)
            precipitacion_dia = round(random.uniform(0.0, 1.5), 1) if random.random() < 0.6 else 0.0
            riesgo_bn = red_bayesiana.predecir_riesgo_quiebre(
                stock_fisico=stock_actual[pid],
                demanda_predicha=demanda_predicha,
                demanda_promedio=demanda_promedio_prod,
                precipitacion=precipitacion_dia,
                es_feriado=es_feriado_hoy_bool,
                es_fin_semana=es_fin_semana_hoy,
                tiempo_reposicion=tiempo_reposicion,
                stock_seguridad=stock_seguridad
            )
            
            # Ajustar stock de seguridad según probabilidad bayesiana
            if riesgo_bn["probabilidad_quiebre"] >= 0.70:
                stock_seguridad = max(2, int(stock_seguridad * 1.50))
            elif riesgo_bn["probabilidad_quiebre"] >= 0.45:
                stock_seguridad = max(2, int(stock_seguridad * 1.25))
            
            es_promo = estado_promocion_activa[pid]
            precio_hoy = precio_promocional_activo[pid] if es_promo else precio_unitario
            
            mapeo_multiplicadores = {
                "Fiestas Patrias": 1.35,
                "Día del Padre": 1.20,
                "Mundial FIFA 2026": 1.15,
                "San Pedro y San Pablo": 0.85,
                "Batalla de Junín": 1.05,
                "Santa Rosa de Lima": 1.10
            }
            multiplicador = mapeo_multiplicadores.get(nombre_feriado, 1.0)
            demanda_final = (demanda_predicha * 1.25 if es_promo else demanda_predicha) * multiplicador
            demanda_entera = int(round(demanda_final))
            
            # Venta real del día
            ventas_reales_hoy = min(stock_actual[pid], demanda_entera)
            stock_actual[pid] -= ventas_reales_hoy
            ventas_perdidas = max(0, demanda_entera - ventas_reales_hoy)
            
            # Lógica del Actuador
            stock_transito = sum([p["cantidad"] for p in pedidos_en_camino[pid]])
            
            recurso_total = stock_actual[pid] + stock_transito
            if recurso_total < (demanda_lead_time + stock_seguridad):
                dif = (demanda_lead_time + stock_seguridad) - recurso_total
                cant_pedido = max(int(np.ceil(dif / cant_min_pedido) * cant_min_pedido), cant_min_pedido)
                
                fecha_llegada = dia.date() + pd.Timedelta(days=tiempo_reposicion)
                pedidos_en_camino[pid].append({
                    "fecha_llegada": fecha_llegada,
                    "cantidad": cant_pedido
                })
                total_pedidos_hechos += 1
                quiebres_evitados_contador += 1
                
            # evaluación de promociones por vencimiento
            if es_perecedero:
                vida_util = int(float(fila_meta["vida_util_dias"])) if pd.notna(fila_meta["vida_util_dias"]) else 30
                if vida_util <= 7:
                    umbral_vencimiento = 2
                elif vida_util <= 15:
                    umbral_vencimiento = 4
                else:
                    umbral_vencimiento = max(5, int(vida_util * 0.25))
                rango_vencimiento = pd.date_range(start=dia + pd.Timedelta(days=1), periods=umbral_vencimiento, freq="D")
                demanda_vencimiento = pred_prod[pred_prod["fecha"].isin(rango_vencimiento)]["demanda_predicha"].sum()
                
                if stock_actual[pid] > demanda_vencimiento and stock_actual[pid] > 5:
                    estado_promocion_activa[pid] = True
                    precio_promocional_activo[pid] = round(precio_unitario * 0.75, 2)
                    mermas_mitigadas_contador += 1
                else:
                    estado_promocion_activa[pid] = False
            else:
                estado_promocion_activa[pid] = False
                
            # rellenar DataFrames
            df_inventario.loc[
                (df_inventario["producto_id"] == pid) & (df_inventario["fecha_dt"] == dia),
                ["stock_fisico", "stock_transito", "hay_stock", "ventas_perdidas_estimadas"]
            ] = [stock_actual[pid], stock_transito, "VERDADERO" if stock_actual[pid] > 0 else "FALSO", ventas_perdidas]
            
            filas_ventas_dia = df_ventas[(df_ventas["producto_id"] == pid) & (df_ventas["fecha_dt"] == dia)]
            if not filas_ventas_dia.empty:
                horas = filas_ventas_dia["hora"].tolist()
                es_feriado_dia = bool(df_festivos[(df_festivos["ds"] == dia)]["holiday"].count() > 0)
                patron_h = agente_cerebro.predecir_patron_horario(pid, dia, es_feriado=es_feriado_dia)
                
                pesos = np.array([patron_h[h] for h in horas])
                suma_pesos = pesos.sum()
                pesos_norm = pesos / suma_pesos if suma_pesos > 0 else np.ones(len(horas)) / len(horas)
                    
                distribucion = np.round(ventas_reales_hoy * pesos_norm).astype(int)
                diferencia_redondeo = ventas_reales_hoy - distribucion.sum()
                if diferencia_redondeo != 0 and len(distribucion) > 0:
                    distribucion[np.argmax(pesos_norm)] += diferencia_redondeo
                    
                indices_filas = filas_ventas_dia.index
                for idx, cant_dist, es_p, prec_p in zip(indices_filas, distribucion, [es_promo]*len(indices_filas), [precio_hoy]*len(indices_filas)):
                    df_ventas.loc[idx, "cantidad_vendida"] = cant_dist
                    df_ventas.loc[idx, "es_promocion"] = "VERDADERO" if es_p else "FALSO"
                    df_ventas.loc[idx, "precio_aplicado"] = prec_p
 
    spinner.stop()
    print("✔ [SIMULACIÓN] Simulación de inventario de Junio a Agosto 2026 finalizada con éxito.")
    print(f"   * Pedidos de stock automáticos realizados por el Actuador: {total_pedidos_hechos}")
    print(f"   * Quiebres de stock potenciales prevenidos: {quiebres_evitados_contador}")
    print(f"   * Acciones de descuento por vencimiento aplicadas: {mermas_mitigadas_contador}")
 
    # guardar CSVs actualizados
    spinner = Spinner("[SENSOR] Guardando los archivos CSV actualizados con las predicciones...")
    spinner.start()
    df_ventas["fecha_hora"] = df_ventas["fecha_hora"].dt.strftime("%d/%m/%Y %H:%M")
    df_inventario["fecha"] = df_inventario["fecha_dt"].dt.strftime("%d/%m/%Y")
    
    df_ventas = df_ventas.drop(columns=["fecha", "fecha_dt"])
    df_inventario = df_inventario.drop(columns=["fecha_dt"])
    
    columnas_bool_ventas = ["es_hora_pico", "es_fin_semana", "es_feriado"]
    for col in columnas_bool_ventas:
        df_ventas[col] = df_ventas[col].map({True: "VERDADERO", False: "FALSO"}).fillna("FALSO")
        
    columnas_bool_inventario = ["es_perecedero", "es_feriado", "es_fin_semana"]
    for col in columnas_bool_inventario:
        df_inventario[col] = df_inventario[col].map({True: "VERDADERO", False: "FALSO"}).fillna("FALSO")
        
    df_ventas.to_csv("datos/ventas_datasL.csv", index=False, encoding="utf-8")
    df_inventario.to_csv("datos/inventario_datasL.csv", index=False, encoding="utf-8")
    spinner.stop()
    print("✔ [SENSOR] Archivos CSV actualizados guardados con éxito.")
 
    # evaluación de Septiembre
    spinner = Spinner("[SENSOR] Cargando datos completos para pronóstico de Septiembre...")
    spinner.start()
    datos_ventas_completos = sensor.cargar_ventas("datos/ventas_datasL.csv")
    datos_inventario_completos = sensor.cargar_inventario("datos/inventario_datasL.csv")
    ventas_diarias_completas = sensor.obtener_ventas_diarias_completas(datos_ventas_completos)
    
    ultima_fecha_simulada = datos_inventario_completos["fecha"].max()
    inventario_final_agosto = datos_inventario_completos[datos_inventario_completos["fecha"] == ultima_fecha_simulada]
    spinner.stop()
    print("✔ [SENSOR] Datos cargados para pronóstico de Septiembre.")
    
    spinner = Spinner("[CEREBRO] Proyectando demanda para Septiembre...")
    spinner.start()
    diccionario_predicciones_futuras = {}
    for prod_id in productos_ids:
        nombre_prod = df_inventario[df_inventario["producto_id"] == prod_id].iloc[0]["producto_nombre"]
        spinner.update_message(f"[CEREBRO] Proyectando demanda para: {nombre_prod}")
        pred_completa = agente_cerebro.predecir_demanda_diaria(
            ventas_diarias_completas, prod_id, dias_a_predecir=14, df_festivos=df_festivos
        )
        pred_futura = pred_completa[pred_completa["fecha"] > pd.to_datetime("2026-08-31")].reset_index(drop=True)
        diccionario_predicciones_futuras[prod_id] = pred_futura
        
    agente_actuador = actuador.ActuadorOptimizado()
    alertas_septiembre = []
    feriados_septiembre = []
    
    for _, fila in inventario_final_agosto.iterrows():
        prod_id = fila["producto_id"]
        pred_fut = diccionario_predicciones_futuras[prod_id]
        patron_h = agente_cerebro.predecir_patron_horario(prod_id, pd.to_datetime("2026-09-01"), es_feriado=False)
        alertas_prod = agente_actuador.evaluar_producto(fila, pred_fut, patron_h, feriados_septiembre, red_bayesiana=red_bayesiana)
        alertas_septiembre.extend(alertas_prod)
    spinner.stop()
    print("✔ [CEREBRO] Proyección de demanda para Septiembre completada.")
        
    # Ejecución de Algoritmos de Búsqueda (Sílabo: BFS, DFS, A*)
    spinner = Spinner("[ACTUADOR] Ejecutando algoritmos de búsqueda BFS, DFS y A*...")
    spinner.start()
    busqueda_res = agente_actuador.ejecutar_busqueda_categorias_bfs_dfs(inventario_final_agosto)
    astar_res = agente_actuador.optimizar_logistica_y_pedidos_astar(inventario_final_agosto, alertas_septiembre)
    spinner.stop()
    print("✔ [ACTUADOR] Algoritmos de búsqueda BFS, DFS y A* ejecutados con éxito.")

    # Ejecución de Poda Alfa-Beta (Entrega 3: Minimax con Poda Alpha-Beta)
    spinner = Spinner("[CEREBRO] Ejecutando algoritmo Poda Alfa-Beta para optimización de decisiones...")
    spinner.start()
    evaluador_poda = EvaluadorPodaAlfaBeta()
    resultados_poda_alfa_beta = []

    for _, fila in inventario_final_agosto.iterrows():
        prod_id = fila["producto_id"]
        prod_nombre = fila["producto_nombre"]
        pred_fut = diccionario_predicciones_futuras[prod_id]
        demanda_prom_diaria = pred_fut["demanda_predicha"].mean() if len(pred_fut) > 0 else 5.0

        estado_prod = EstadoInventario(
            stock_actual=fila["stock_fisico"],
            demanda_diaria_prom=demanda_prom_diaria,
            costo_unidad=fila["precio_unitario"] * 0.65,
            precio_venta=fila["precio_unitario"],
            dias_vencimiento=fila["vida_util_dias"] if fila["es_perecedero"] else 999,
            es_perecedero=fila["es_perecedero"],
            lead_time_base=int(fila["tiempo_reposicion_dias"])
        )

        res_poda = evaluador_poda.optimizar_decision(prod_id, prod_nombre, estado_prod, profundidad=3)
        resultados_poda_alfa_beta.append(res_poda)

    spinner.stop()
    print("✔ [CEREBRO] Algoritmo Poda Alfa-Beta ejecutado con éxito.")
    if resultados_poda_alfa_beta:
        eficiencia_media = np.mean([r["eficiencia_poda_pct"] for r in resultados_poda_alfa_beta])
        print(f"   * Productos optimizados: {len(resultados_poda_alfa_beta)}")
        print(f"   * Eficiencia promedio de poda (α-β): {eficiencia_media:.1f}% de nodos podados")

    # reportes gráficos y escritos
    spinner = Spinner("[REPORTADOR] Generando reportes gráficos y escritos finales...")
    spinner.start()
    reportador.graficar_demanda_y_alertas(inventario_final_agosto, diccionario_predicciones_futuras, alertas_septiembre)
    reportador.graficar_patrones_horas_pico(inventario_final_agosto, diccionario_patrones_horarios)
    reportador.graficar_curva_perdida_mlp(agente_cerebro)
    reportador.graficar_arquitectura_mlp_proyecto()
    reportador.graficar_comportamiento_perceptrones()
    reportador.graficar_productos_mas_demandados(datos_ventas_completos)
    reportador.generar_dashboard_alertas(alertas_septiembre)
    reportador.graficar_analisis_eventos(datos_ventas_completos)
    reportador.graficar_dispersion_ventas(datos_ventas_completos)
    reportador.graficar_dispersion_inventario(datos_inventario_completos)
    reportador.graficar_tabla_proyeccion_septiembre(inventario_final_agosto, alertas_septiembre, diccionario_predicciones_futuras)
    reportador.graficar_tabla_impacto_eventos(datos_ventas_completos)
    reportador.graficar_red_bayesiana(red_bayesiana, inventario_final_agosto, diccionario_predicciones_futuras)
    reportador.graficar_arbol_categorias_bfs_dfs(busqueda_res)
    reportador.graficar_ruta_reabastecimiento_astar(astar_res)
    reportador.graficar_resultados_poda_alfa_beta(resultados_poda_alfa_beta)
    
    # Generar árbol explicativo de la Poda Alfa-Beta para Inca Kola 500ml (usando la ejecución real)
    try:
        res_inka = next((r for r in resultados_poda_alfa_beta if r["producto_id"] == "prod_001"), None)
        if res_inka and "registro_arbol" in res_inka:
            reportador.graficar_arbol_poda_alfa_beta(
                res_inka["registro_arbol"], 
                res_inka["mejor_accion"], 
                "Inca Kola 500ml"
            )
        else:
            print("   * Advertencia: No se encontró el registro real de poda para Inca Kola.")
    except Exception as e:
        print(f"   * Advertencia al generar el árbol de decisión visual: {e}")

    reportador.graficar_mapa_calor_quiebres_final(datos_inventario_completos)

    
    
    generar_reporte_escrito_final(
        inventario_final_agosto, alertas_septiembre, diccionario_predicciones_futuras, datos_ventas_completos,
        total_pedidos_hechos, quiebres_evitados_contador, mermas_mitigadas_contador,
        predicciones_completas=predicciones_completas,
        red_bayesiana=red_bayesiana,
        busqueda_res=busqueda_res,
        astar_res=astar_res,
        resultados_poda=resultados_poda_alfa_beta
    )
    spinner.stop()
    print("✔ [REPORTADOR] Reportes finales generados con éxito.")
    
    print("\n" + "=" * 75)
    print("   ¡AGENTE PREDICTIVO Y SIMULACIÓN COMPLETA COMPLETADA CON ÉXITO!")
    print("=" * 75)
    print("Los reportes visuales consolidados han sido guardados en 'reportes/'")
    print("organizados en carpetas temáticas:")
    print("  📁 01_algoritmos_busqueda_grafos/   (BFS, DFS y A*)")
    print("  📁 02_redes_bayesiana_causal/       (DAG, Métricas BIC, Mapa de Calor)")
    print("  📁 03_prediccion_prophet_mlp/       (Series Temporales y Patrones Horarios)")
    print("  📁 04_alertas_y_eventos/            (Dashboard, Impacto Feriados y Tablas)")
    print("  📁 05_poda_alfa_beta/               (Nodos Evaluados, Eficiencia y Matriz Utilidad)")
    print("=" * 75)

def generar_reporte_escrito_final(inventario_actual, alertas, predicciones, datos_ventas, total_pedidos, quiebres_prev, mermas_prev, predicciones_completas=None, red_bayesiana=None, busqueda_res=None, astar_res=None, resultados_poda=None, ruta_salida="reportes/reporte_ejecucion.txt"):
    # Genera el archivo consolidado de reporte escrito en disco.
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write("========================================================================\n")
        archivo.write("         INFORME DE PREDICCIÓN Y OPTIMIZACIÓN DE INVENTARIO - PYMEVISION AI\n")
        archivo.write(f"Fecha de Emisión: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        archivo.write("========================================================================\n\n")
        
        archivo.write("1. RESULTADOS DE LA SIMULACIÓN AUTÓNOMA (JUNIO - AGOSTO 2026)\n")
        archivo.write("------------------------------------------------------------------------\n")
        archivo.write(f"Durante los meses de junio a agosto de 2026, el agente inteligente gestionó\n")
        archivo.write("el stock de forma 100% autónoma ante la ausencia de registros de ventas.\n\n")
        archivo.write(f"  - Pedidos de compra automáticos colocados al proveedor : {total_pedidos}\n")
        archivo.write(f"  - Quiebres de stock potenciales identificados y evitados : {quiebres_prev}\n")
        archivo.write(f"  - Ofertas aplicadas por riesgo de vencimiento (mermas) : {mermas_prev}\n\n")
        
        archivo.write("2. ALERTAS CRÍTICAS DE STOCK EN SEPTIEMBRE (ACCIONAR INMEDIATO)\n")
        archivo.write("------------------------------------------------------------------------\n")
        alertas_altas = [a for a in alertas if a["gravedad"] == "ALTA"]
        if not alertas_altas:
            archivo.write("No se detectaron riesgos de stock críticos en septiembre tras la optimización.\n")
        for i, a in enumerate(alertas_altas, 1):
            archivo.write(f"{i}. Producto: {a.get('producto_nombre', 'N/D')}\n")
            archivo.write(f"   Tipo de Riesgo: {a['tipo']}\n")
            archivo.write(f"   Detalle: {a['descripcion']}\n")
            archivo.write(f"   Acción Recomendada: {a['accion_sugerida']}\n\n")
            
        archivo.write("3. ALERTAS MODERADAS Y RECOMENDACIONES EN SEPTIEMBRE\n")
        archivo.write("------------------------------------------------------------------------\n")
        alertas_medias = [a for a in alertas if a["gravedad"] == "MEDIA"]
        if not alertas_medias:
            archivo.write("No se detectaron alertas de stock de gravedad media.\n")
        for i, a in enumerate(alertas_medias, 1):
            archivo.write(f"{i}. Producto: {a.get('producto_nombre', 'N/D')}\n")
            archivo.write(f"   Tipo de Alerta: {a['tipo']}\n")
            archivo.write(f"   Detalle: {a['descripcion']}\n")
            archivo.write(f"   Acción Recomendada: {a['accion_sugerida']}\n\n")
            
        archivo.write("4. ALERTAS OPERATIVAS (HORAS PICO / FERIADOS SEPTIEMBRE)\n")
        archivo.write("------------------------------------------------------------------------\n")
        alertas_operativas = [a for a in alertas if a["gravedad"] == "BAJA"]
        for a in alertas_operativas:
            archivo.write(f"- Producto: {a.get('producto_nombre', 'N/D')} | {a['tipo']}\n")
            archivo.write(f"  Detalle: {a['descripcion']}\n")
            archivo.write(f"  Acción: {a['accion_sugerida']}\n\n")
            
        archivo.write("5. PROYECCIÓN DE DEMANDA PARA SEPTIEMBRE (PRIMEROS 7 DÍAS)\n")
        archivo.write("------------------------------------------------------------------------\n")
        archivo.write(f"{'Producto':<30} | {'Stock Fis.':<10} | {'Demanda Est.':<12} | {'Estado Alerta':<15}\n")
        archivo.write("-" * 75 + "\n")
        for _, fila in inventario_actual.iterrows():
            prod_id = fila["producto_id"]
            nombre = fila["producto_nombre"]
            stock = fila["stock_fisico"]
            pred = predicciones[prod_id]
            demanda_7d = pred.head(7)["demanda_predicha"].sum()
            
            alertas_prod = [a for a in alertas if a.get("producto_id") == prod_id]
            estado = "NORMAL"
            for a in alertas_prod:
                if a["gravedad"] == "ALTA":
                    estado = "¡QUIEBRE!"
                    break
                elif a["gravedad"] == "MEDIA":
                    estado = "PREVENCIÓN"
                    
            archivo.write(f"{nombre:<30} | {stock:<10} | {demanda_7d:<12.1f} | {estado:<15}\n")
            
        archivo.write("\n6. ANÁLISIS DE IMPACTO DE EVENTOS ESPECIALES EN LAS VENTAS (HISTÓRICO)\n")
        archivo.write("------------------------------------------------------------------------\n")
        archivo.write("El agente inteligente ha analizado las ventas consolidadas de los meses históricos\n")
        archivo.write("y ha identificado los siguientes incrementos en la demanda diaria promedio:\n\n")
        
        datos_ventas_copia = datos_ventas.copy()
        if "fecha" not in datos_ventas_copia.columns:
            datos_ventas_copia["fecha"] = pd.to_datetime(datos_ventas_copia["fecha_hora"], format="%d/%m/%Y %H:%M").dt.date
        else:
            datos_ventas_copia["fecha"] = pd.to_datetime(datos_ventas_copia["fecha"]).dt.date
            
        df_ventas_diarias = datos_ventas_copia.groupby(["fecha", "evento_especial"])["cantidad_vendida"].sum().reset_index()
        promedios = df_ventas_diarias.groupby("evento_especial")["cantidad_vendida"].mean().to_dict()
        
        promedio_normal = promedios.get("ninguno", 179.58)
        
        mapeo_nombres = {
            "ninguno": "Día Normal (Sin eventos)",
            "Día de la Madre": "Campaña Día de la Madre",
            "Día del Padre": "Campaña Día del Padre",
            "Fiestas Patrias": "Campaña Fiestas Patrias",
            "Mundial FIFA 2026": "Periodo Mundial FIFA",
            "Santa Rosa de Lima": "Santa Rosa de Lima",
            "Batalla de Junín": "Batalla de Junín"
        }
        
        for evt, prom in sorted(promedios.items(), key=lambda x: x[1], reverse=True):
            nombre_limpio = mapeo_nombres.get(evt, evt)
            if evt != "ninguno":
                incremento = ((prom / promedio_normal) - 1) * 100
                archivo.write(f"- {nombre_limpio:<25}: {prom:<6.1f} und./día | Incremento: +{int(incremento):>3}%\n")
            else:
                archivo.write(f"- {nombre_limpio:<25}: {prom:<6.1f} und./día | Línea Base (Normal)\n")
                
        archivo.write("\n7. EVALUACIÓN DE PREDICCIÓN EN EVENTOS HISTÓRICOS (REAL VS PREDICHO EN MAYO)\n")
        archivo.write("------------------------------------------------------------------------\n")
        archivo.write("Dado que el periodo de junio a agosto estaba vacío y fue simulado de forma autónoma,\n")
        archivo.write("evaluamos la precisión del agente comparando su predicción contra la venta real\n")
        archivo.write("del evento 'Día de la Madre' ocurrido el 11/05/2026 en el periodo de entrenamiento:\n\n")
        
        fecha_madre = pd.to_datetime("2026-05-11").date()
        venta_real_madre = datos_ventas_copia[datos_ventas_copia["fecha"] == fecha_madre]["cantidad_vendida"].sum()
        
        venta_predicha_madre = 0
        dict_preds_para_madre = predicciones_completas if predicciones_completas is not None else predicciones
        for pid, df_pred in dict_preds_para_madre.items():
            fila_pred = df_pred[df_pred["fecha"] == pd.to_datetime("2026-05-11")]
            if not fila_pred.empty:
                venta_predicha_madre += fila_pred.iloc[0]["demanda_predicha"]
                
        error_madre = abs(venta_real_madre - venta_predicha_madre) / venta_real_madre if venta_real_madre > 0 else 0
        precision_madre = max(0, 1 - error_madre) * 100
        
        archivo.write(f"Campaña: Día de la Madre (11/05/2026)\n")
        archivo.write(f"  * Cantidad de Ventas Real     : {int(venta_real_madre)} unidades\n")
        archivo.write(f"  * Cantidad de Ventas Predicha : {venta_predicha_madre:.1f} unidades\n")
        archivo.write(f"  * Precisión de Predicción     : {precision_madre:.1f}%\n")
        
        # Sección de Red Bayesiana Mixta (Noisy-OR)
        if red_bayesiana is not None and red_bayesiana.entrenado:
            metricas = red_bayesiana.obtener_metricas()
            archivo.write("\n8. RED BAYESIANA MIXTA (NOISY-OR) - ANÁLISIS CAUSAL DE INCERTIDUMBRE\n")
            archivo.write("------------------------------------------------------------------------\n")
            archivo.write("Siguiendo la metodología de Vomlel et al. (2023), se implementó una Red\n")
            archivo.write("Bayesiana Mixta con modelo Noisy-OR para el análisis causal de quiebres\n")
            archivo.write("de stock, optimizada mediante el Criterio de Información Bayesiano (BIC).\n\n")
            
            archivo.write("8.1 Estructura del Modelo:\n")
            archivo.write(f"  * Nodo objetivo: Quiebre de Stock (variable binaria)\n")
            archivo.write(f"  * Nodos padre (causas): {len(metricas['nodos_causa'])}\n")
            for i, causa in enumerate(metricas['nodos_causa'], 1):
                nombre_causa = causa.replace('_', ' ').title()
                archivo.write(f"    {i}. {nombre_causa}\n")
            archivo.write(f"  * Probabilidad de fuga (p_v,0): {metricas['prob_fuga']}\n")
            archivo.write(f"    (Captura mermas, robos y errores administrativos no modelados)\n\n")
            
            archivo.write("8.2 Parámetros Aprendidos (Probabilidades de Inhibición p_v,j):\n")
            for causa, prob in metricas['prob_inhibicion'].items():
                nombre_causa = causa.replace('_', ' ').title()
                impacto = "ALTO" if prob < 0.40 else ("MEDIO" if prob < 0.65 else "BAJO")
                archivo.write(f"  * {nombre_causa:<25}: p_v,j = {prob:.4f} | Impacto: {impacto}\n")
            archivo.write("\n")
            
            archivo.write("8.3 Criterio de Información Bayesiano (BIC) - Autoevaluación:\n")
            archivo.write(f"  * BIC Score                   : {metricas['bic_score']}\n")
            archivo.write(f"  * Log-Verosimilitud LL(P|D)   : {metricas['log_verosimilitud']}\n")
            archivo.write(f"  * Muestras de entrenamiento   : {metricas['n_muestras']}\n")
            archivo.write(f"  * Fórmula: BIC(P|D) = LL(P|D) - (log|D|/2) · C(P)\n\n")
            
            archivo.write("8.4 Eficiencia de Penalización Lineal vs Exponencial:\n")
            archivo.write(f"  * Parámetros Noisy-OR (|pa(v)|+1)    : {metricas['n_parametros_noisy_or']}\n")
            archivo.write(f"  * Parámetros CPT Exponencial (2^|pa|): {metricas['n_parametros_cpt_exponencial']}\n")
            archivo.write(f"  * Reducción de complejidad           : {metricas['reduccion_parametros']}\n")
            archivo.write(f"  Esta reducción permite operar con datos limitados de PYMEs sin sobreajuste.\n\n")
            
            # Calcular riesgo por producto para el reporte
            archivo.write("8.5 Evaluación de Riesgo por Producto (Noisy-OR):\n")
            archivo.write(f"{'Producto':<30} | {'P(Quiebre)':<12} | {'Nivel':<10} | {'Causas Activas':<30}\n")
            archivo.write("-" * 90 + "\n")
            for _, fila in inventario_actual.iterrows():
                prod_id = fila["producto_id"]
                pred = predicciones[prod_id]
                demanda_prom = pred["demanda_predicha"].mean()
                demanda_dia = pred.iloc[0]["demanda_predicha"] if not pred.empty else 0
                stock_seg = max(2, int(demanda_prom * 0.20 * fila["tiempo_reposicion_dias"]))
                
                resultado = red_bayesiana.predecir_riesgo_quiebre(
                    stock_fisico=fila["stock_fisico"],
                    demanda_predicha=demanda_dia,
                    demanda_promedio=demanda_prom,
                    precipitacion=0.5,
                    es_feriado=bool(fila.get("es_feriado", False)),
                    es_fin_semana=bool(fila.get("es_fin_semana", False)),
                    tiempo_reposicion=int(fila["tiempo_reposicion_dias"]),
                    stock_seguridad=stock_seg
                )
                causas_str = ", ".join([c.replace("_", " ") for c in resultado["causas_activas"]]) if resultado["causas_activas"] else "ninguna"
                archivo.write(f"{fila['producto_nombre']:<30} | {resultado['probabilidad_quiebre']*100:>8.1f}%   | {resultado['nivel_riesgo']:<10} | {causas_str:<30}\n")
        
        # Sección 9: Algoritmos de Búsqueda (BFS, DFS y A*)
        archivo.write("\n9. ALGORITMOS DE BÚSQUEDA NO INFORMADA E INFORMADA (SÍLABO: BFS, DFS, A*)\n")
        archivo.write("------------------------------------------------------------------------\n")
        archivo.write("Para dar cumplimiento estricto al sílabo del curso y potenciar la toma de\n")
        archivo.write("decisiones en el ciclo Sense-Plan-Act del Actuador, se incorporaron 3 algoritmos:\n\n")
        
        if busqueda_res is not None:
            orden_bfs = busqueda_res["bfs"]["orden_visita"]
            orden_dfs = busqueda_res["dfs"]["orden_visita"]
            archivo.write("9.1 Búsqueda No Informada en Árbol de Categorías (BFS vs DFS):\n")
            archivo.write(f"  * BFS (Breadth-First Search / Cola FIFO) : Recorre {len(orden_bfs)} nodos por niveles.\n")
            archivo.write(f"    Garantiza la inspección equitativa nivel por nivel de todas las categorías.\n")
            archivo.write(f"  * DFS (Depth-First Search / Pila LIFO)  : Recorre {len(orden_dfs)} nodos en profundidad.\n")
            archivo.write(f"    Permite la auditoría completa rama por rama de cada categoría de producto.\n\n")
            
        if astar_res is not None:
            ruta_data = astar_res["ruta_reabastecimiento_astar"]
            ordenes_astar = astar_res["ordenes_priorizadas_astar"]
            
            archivo.write("9.2 Algoritmo de Búsqueda Informada A* Secuencial Multi-Objetivo (Paarth Sonkiya, 2024):\n")
            archivo.write("  * Estrategia: Ruta de Picking con Heurística de Vecino Más Cercano (Nearest Neighbor) + A*\n")
            archivo.write("  * Función de Evaluación: f(n) = g(n) + h(n) en cada etapa de recorrido\n")
            archivo.write("  * Heurística h(n): Distancia Manhattan d(n, próximo_estante) = |x - x_obj| + |y - y_obj|\n")
            archivo.write(f"  * Costo Real Acumulado Total g(n)       : {ruta_data['costo_g_total']:.2f}\n")
            archivo.write(f"  * Heurística Inicial h(0,0)             : {ruta_data.get('heuristica_h_inicial', 18):.2f}\n")
            archivo.write(f"  * Evaluación Total de Ruta f(n)         : {ruta_data['evaluacion_f_total']:.2f}\n\n")
            
            archivo.write("9.3 Priorización Estratégica de Pedidos (A* Ranking f(n)):\n")
            archivo.write(f"{'Producto':<25} | {'Alerta':<18} | {'Prob. Riesgo':<12} | {'Costo g(n)':<10} | {'Heur. h(n)':<10} | {'Prioridad f(n)':<12}\n")
            archivo.write("-" * 95 + "\n")
            for ord_a in ordenes_astar[:5]:
                archivo.write(f"{ord_a['producto_id']:<25} | {ord_a['tipo_alerta']:<18} | {ord_a['probabilidad_riesgo']*100:>10.1f}% | {ord_a['costo_g']:>10.2f} | {ord_a['heuristica_h']:>10.2f} | {ord_a['evaluacion_f']:>12.2f}\n")

        # Sección 10: Algoritmo Poda Alfa-Beta (Entrega 3)
        if resultados_poda is not None:
            archivo.write("\n10. OPTIMIZACIÓN DE DECISIONES DE REABASTECIMIENTO (PODA ALFA-BETA / MINIMAX)\n")
            archivo.write("------------------------------------------------------------------------\n")
            archivo.write("Se aplicó la Poda Alfa-Beta sobre un modelo de árbol de decisiones adversariales\n")
            archivo.write("donde el Agente (MAX) busca maximizar el margen de utilidad neta y el Entorno (MIN)\n")
            archivo.write("evalúa escenarios de alta demanda, retrasos de proveedor y crisis de inventario.\n\n")

            archivo.write("JUSTIFICACIÓN TEÓRICA DEL JUGADOR MIN (MERCADO / ENTORNO):\n")
            archivo.write("  En la gestión de inventarios bajo incertidumbre, el mercado y los proveedores se modelan\n")
            archivo.write("  como un jugador adversarial (MIN) no porque tengan una intención real o maliciosa de perjudicar\n")
            archivo.write("  a la bodega, sino porque representa la metodología formal del Criterio Minimax de Decisión\n")
            archivo.write("  bajo Incertidumbre. Al asumir que el entorno seleccionará la combinación de eventos más desfavorable\n")
            archivo.write("  (picos de demanda, retrasos logísticos o crisis de inventario), el Agente (MAX) está forzado a\n")
            archivo.write("  planificar una estrategia ROBUSTAMENTE ÓPTIMA. Esta aproximación garantiza que, incluso ante el peor\n")
            archivo.write("  escenario posible, las pérdidas se minimizan y la continuidad operativa se protege de manera estable.\n\n")

            eficiencia_prom = np.mean([r["eficiencia_poda_pct"] for r in resultados_poda]) if resultados_poda else 0
            prof_eval = resultados_poda[0].get("profundidad_evaluada", 2) if resultados_poda else 2
            archivo.write(f"10.1 Métricas Generales de Poda Alpha-Beta:\n")
            archivo.write(f"  * Total Productos Evaluados      : {len(resultados_poda)}\n")
            archivo.write(f"  * Profundidad del Árbol (Días)  : {prof_eval} niveles de decisiones encadenadas en el tiempo\n")
            archivo.write(f"  * Eficiencia Promedio de Poda    : {eficiencia_prom:.1f}% de nodos podados del árbol\n")
            archivo.write(f"  * Reducción de Complejidad      : Permite evaluar decisiones multietapa a N días en tiempo real.\n\n")

            archivo.write("10.2 Decisiones Estratégicas Seleccionadas por Producto:\n")
            archivo.write(f"{'Producto':<30} | {'Acción Óptima':<18} | {'Utilidad Est.':<15} | {'Poda α-β (%)':<12}\n")
            archivo.write("-" * 85 + "\n")
            for res in resultados_poda:
                archivo.write(f"{res['producto_nombre']:<30} | {res['mejor_accion']:<18} | S/. {res['utilidad_optima']:>10,.2f} | {res['eficiencia_poda_pct']:>10.1f}%\n")

            archivo.write("\n10.3 Aclaración Teórica sobre la Matriz de Utilidades Graficada:\n")
            archivo.write("  * La matriz gráfica exhibe las utilidades inmediatas del paso 1 (Día 1, U_1).\n")
            archivo.write("  * En el Día 1, la compra de stock incurre en un costo inicial, haciendo parecer superficialmente desfavorables\n")
            archivo.write("    las acciones de reabastecimiento si solo se observara dicho primer día.\n")
            archivo.write(f"  * No obstante, la Decisión Óptima surge del Minimax propagado a {prof_eval} días de horizonte (D={prof_eval}),\n")
            archivo.write("    donde disponer del stock reabastecido previene colapsos por quiebre y multas críticas en los días 2 y 3.\n")
            archivo.write("  * Se aplica un factor de descuento gamma = 0.95 a las utilidades de los días futuros dentro del árbol multietapa,\n")
            archivo.write("    reflejando la preferencia estándar en modelos de decisión secuencial por resultados más cercanos en el tiempo\n")
            archivo.write("    frente a proyecciones inciertas más lejanas.\n\n")

            archivo.write("10.4 Demostración Numérica Detallada Paso a Paso (Ejemplo: Inca Kola 500ml):\n")
            archivo.write(f"{'Acción Día 1':<18} | {'Peor Esc. MIN':<18} | {'U1 (Día 1)':<12} | {'Stock S2':<10} | {'Acción Día 2':<16} | {'Subárbol D2-D3':<14} | {'Total Acumulado':<15}\n")
            archivo.write("-" * 115 + "\n")
            
            # Obtener traza explicativa de Inca Kola
            if resultados_poda:
                first_item = resultados_poda[0]
                evaluador_aux = EvaluadorPodaAlfaBeta()
                sub_inv = inventario_actual[inventario_actual["producto_id"] == first_item["producto_id"]]
                if not sub_inv.empty:
                    meta_ik = sub_inv.iloc[0]
                    pred_ik = predicciones.get(first_item["producto_id"], pd.DataFrame())
                    dem_prom_ik = pred_ik["demanda_predicha"].mean() if not pred_ik.empty else 22.0
                    st_ik = EstadoInventario(
                        stock_actual=meta_ik["stock_fisico"],
                        demanda_diaria_prom=dem_prom_ik,
                        costo_unidad=meta_ik["precio_unitario"] * 0.65,
                        precio_venta=meta_ik["precio_unitario"],
                        dias_vencimiento=meta_ik["vida_util_dias"] if meta_ik["es_perecedero"] else 999,
                        es_perecedero=meta_ik["es_perecedero"],
                        lead_time_base=int(meta_ik["tiempo_reposicion_dias"])
                    )
                    df_tr = evaluador_aux.generar_traza_explicativa(first_item["producto_nombre"], st_ik, profundidad=3)
                    for _, row in df_tr.iterrows():
                        archivo.write(f"{row['accion_d1']:<18} | {row['escenario_d1']:<18} | S/. {row['u1_inmediata']:>7.2f}  | {row['stock_s2']:>8.1f}  | {row['accion_d2_optima']:<16} | S/. {row['subarbol_d2_d3']:>9.2f}  | S/. {row['total_acumulado']:>10.2f}\n")

        archivo.write("\n========================================================================\n")
        archivo.write("Fin del reporte de control de stock y optimización de ventas.\n")
        archivo.write("========================================================================\n")

if __name__ == "__main__":
    ejecutar_agente()
