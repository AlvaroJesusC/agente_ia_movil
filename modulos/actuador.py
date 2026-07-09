# -*- coding: utf-8 -*-

# Módulo Actuador para pymevision ai
# Evaluación de riesgos de inventario y recomendaciones de abastecimiento.

import pandas as pd
import numpy as np
from modulos.sistema_experto import Hecho, MotorInferenciaForward, inicializar_base_conocimientos

class ActuadorOptimizado:
    # Evalúa riesgos de stock y genera recomendaciones utilizando un Sistema Experto formal.
    
    def __init__(self):
        # Base de conocimientos compartida (reglas de producción)
        self.base_reglas = inicializar_base_conocimientos()
        # Almacena las trazas paso a paso del motor de inferencia por producto para auditoría
        self.trazas_inferencia = {}
        
    def evaluar_producto(self, fila_inventario, prediccion_diaria, patron_horario, feriados_proximos, red_bayesiana=None):
        """
        Analiza el estado del inventario para generar alertas utilizando inferencia por encadenamiento
        hacia adelante (Forward Chaining) en base a Hechos del entorno y Reglas lógicas de negocio.
        """
        producto_id = fila_inventario["producto_id"]
        nombre_producto = fila_inventario["producto_nombre"]
        stock_fisico = fila_inventario["stock_fisico"]
        stock_transito = fila_inventario["stock_transito"]
        tiempo_reposicion = int(fila_inventario["tiempo_reposicion_dias"])
        cant_min_pedido = int(fila_inventario["cantidad_minima_pedido"])
        es_perecedero = fila_inventario["es_perecedero"]
        precio_unitario = fila_inventario["precio_unitario"]
        
        # -------------------------------------------------------------------------
        # FASE 1: OBTENCIÓN DE DATOS NUMÉRICOS PREVIOS (Fase de Percepción)
        # -------------------------------------------------------------------------
        prediccion_reposicion = prediccion_diaria.head(tiempo_reposicion)
        demanda_lead_time = prediccion_reposicion["demanda_predicha"].sum()
        stock_seguridad_base = max(2, int(demanda_lead_time * 0.20))
        
        # 1.1 Evaluación bayesiana preliminar si está disponible
        prob_quiebre_bayesiana = 0.0
        causas_activas_bayesiana = []
        if red_bayesiana is not None and red_bayesiana.entrenado:
            demanda_promedio = prediccion_diaria["demanda_predicha"].mean()
            res_bayesiano = red_bayesiana.predecir_riesgo_quiebre(
                stock_fisico=stock_fisico,
                demanda_predicha=demanda_lead_time / max(tiempo_reposicion, 1),
                demanda_promedio=demanda_promedio,
                precipitacion=0.5,
                es_feriado=bool(fila_inventario.get("es_feriado", False)),
                es_fin_semana=bool(fila_inventario.get("es_fin_semana", False)),
                tiempo_reposicion=tiempo_reposicion,
                stock_seguridad=stock_seguridad_base
            )
            prob_quiebre_bayesiana = res_bayesiano["probabilidad_quiebre"]
            causas_activas_bayesiana = res_bayesiano["causas_activas"]

        # 1.2 Evaluación del vencimiento dinámico
        vida_util_raw = fila_inventario.get("vida_util_dias", 30)
        vida_util = int(float(vida_util_raw)) if pd.notna(vida_util_raw) else 30
        if vida_util <= 7:
            umbral_dias = 2
        elif vida_util <= 15:
            umbral_dias = 4
        else:
            umbral_dias = max(5, int(vida_util * 0.25))
            
        prediccion_vencimiento = prediccion_diaria.head(umbral_dias)
        demanda_critica = prediccion_vencimiento["demanda_predicha"].sum()
        
        # 1.3 Búsqueda de feriados próximos
        feriado_proximo = False
        demanda_feriado_alta = False
        fecha_feriado_proximo = None
        for fecha_f, es_feriado_f in feriados_proximos:
            if es_feriado_f:
                demanda_promedio = prediccion_diaria["demanda_predicha"].mean()
                demanda_feriado_val = prediccion_diaria[prediccion_diaria["fecha"] == fecha_f]["demanda_predicha"].values
                if len(demanda_feriado_val) > 0:
                    demanda_fer = demanda_feriado_val[0]
                    if demanda_fer > (demanda_promedio * 1.30):
                        feriado_proximo = True
                        demanda_feriado_alta = True
                        fecha_feriado_proximo = fecha_f
                        break

        # 1.4 Búsqueda de horas pico
        umbral_hora = np.percentile(patron_horario, 85)
        horas_pico = [h for h in range(24) if patron_horario[h] > umbral_hora and patron_horario[h] > 1.5]
        
        # -------------------------------------------------------------------------
        # FASE 2: CONSTRUCCIÓN DE LA BASE DE HECHOS INICIALES (Memoria de Trabajo)
        # -------------------------------------------------------------------------
        recurso_total = stock_fisico + stock_transito
        
        hechos_iniciales = []
        
        # Hechos sobre quiebre
        if recurso_total < demanda_lead_time:
            hechos_iniciales.append(Hecho("recurso_criticamente_bajo", True, "El stock disponible no cubre la demanda del lead time."))
        if recurso_total < (demanda_lead_time + stock_seguridad_base):
            hechos_iniciales.append(Hecho("recurso_bajo_seguridad", True, "El stock disponible no cubre la demanda + stock seguridad."))
        if tiempo_reposicion <= 2:
            hechos_iniciales.append(Hecho("tiempo_reposicion_inminente", True, "El tiempo de reposición del proveedor es muy corto (<= 2 días)."))
        else:
            hechos_iniciales.append(Hecho("tiempo_reposicion_estandar", True, "El tiempo de reposición es normal (> 2 días)."))

        # Hechos sobre vencimiento (perecederos)
        if es_perecedero:
            hechos_iniciales.append(Hecho("es_perecedero", True, "El producto es perecedero y tiene fecha de caducidad."))
            if stock_fisico > (demanda_critica * 1.5) and stock_fisico > 5:
                hechos_iniciales.append(Hecho("stock_excedente_critico", True, "El stock físico supera ampliamente la demanda hasta su fecha límite."))
            elif stock_fisico > demanda_critica and stock_fisico > 5:
                hechos_iniciales.append(Hecho("stock_excedente_moderado", True, "El stock físico supera la demanda hasta su fecha límite."))
            
            if vida_util <= 7:
                hechos_iniciales.append(Hecho("vencimiento_muy_cercano", True, "Vida útil muy corta (<= 7 días)."))
            elif vida_util <= 15:
                hechos_iniciales.append(Hecho("vencimiento_moderadamente_cercano", True, "Vida útil intermedia (<= 15 días)."))

        # Hechos sobre feriados y eventos especiales
        if feriado_proximo:
            hechos_iniciales.append(Hecho("feriado_proximo_detectado", True, "Hay un feriado nacional programado en los próximos días."))
        if demanda_feriado_alta:
            hechos_iniciales.append(Hecho("demanda_feriado_superior_promedio", True, "La demanda proyectada del feriado es 30% mayor al promedio."))

        # Hechos sobre horas pico (red neuronal MLP)
        hechos_iniciales.append(Hecho("patron_horario_disponible", True))
        if len(horas_pico) > 0:
            hechos_iniciales.append(Hecho("horas_pico_sobre_percentil_85", True, "El patrón horario muestra picos de demanda significativos."))

        # Hechos sobre incertidumbre bayesiana (Red Bayesiana Noisy-OR)
        if prob_quiebre_bayesiana >= 0.70:
            hechos_iniciales.append(Hecho("probabilidad_quiebre_bayesiano_critica", True))
        elif prob_quiebre_bayesiana >= 0.45:
            hechos_iniciales.append(Hecho("probabilidad_quiebre_bayesiano_alta", True))

        # -------------------------------------------------------------------------
        # FASE 3: EJECUCIÓN DEL MOTOR DE INFERENCIA (Forward Chaining)
        # -------------------------------------------------------------------------
        motor = MotorInferenciaForward(self.base_reglas)
        hechos_inferidos, alertas_deducidas = motor.inferir(hechos_iniciales)
        
        # Registrar la traza para auditorías e informes del usuario
        self.trazas_inferencia[producto_id] = {
            "producto_nombre": nombre_producto,
            "traza": motor.traza_inferencia,
            "hechos_iniciales": [h.nombre for h in hechos_iniciales],
            "hechos_finales": list(hechos_inferidos.keys()),
            "alertas_deducidas": alertas_deducidas
        }

        # -------------------------------------------------------------------------
        # FASE 4: ACCIÓN Y MAPEO DE CONSECUIENTES A ALERTAS CONCRETAS
        # -------------------------------------------------------------------------
        alertas = []
        
        # 4.1 Ajuste del stock de seguridad deducido por reglas bayesianas
        factor_ajuste = 1.0
        if hechos_inferidos.get("ACCION_INCREMENTO_SEGURIDAD_MAX", False):
            factor_ajuste = 1.50
        elif hechos_inferidos.get("ACCION_INCREMENTO_SEGURIDAD_MOD", False):
            factor_ajuste = 1.25
        
        stock_seguridad_final = max(2, int(stock_seguridad_base * factor_ajuste))

        # 4.2 Alerta de Quiebre de Stock (Crítico o Moderado)
        cant_pedir = 0
        if hechos_inferidos.get("ACCION_ORDEN_EMERGENCIA", False) or hechos_inferidos.get("ACCION_ORDEN_NORMAL", False):
            diferencia = (demanda_lead_time + stock_seguridad_final) - recurso_total
            cant_pedir = int(np.ceil(diferencia / cant_min_pedido) * cant_min_pedido)
            
            gravedad = "ALTA" if hechos_inferidos.get("ACCION_ORDEN_EMERGENCIA", False) else "MEDIA"
            desc = (
                f"El stock total ({recurso_total} und.) no cubrirá la demanda proyectada ({demanda_lead_time:.1f} und.) "
                f"más el stock de seguridad ajustado ({stock_seguridad_final} und.) durante los {tiempo_reposicion} días de reposición."
            )
            accion = f"Generar pedido automático de {cant_pedir} unidades al proveedor."
            
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "RIESGO_QUIEBRE",
                "gravedad": gravedad,
                "descripcion": desc,
                "accion_sugerida": accion,
                "cantidad_pedido": cant_pedir,
                "probabilidad_bayesiana": prob_quiebre_bayesiana
            })

        # 4.3 Alerta de Vencimiento y Pérdidas (Crítico o Moderado)
        if hechos_inferidos.get("ACCION_PROMO_DESCUENTO_30", False) or hechos_inferidos.get("ACCION_PROMO_DESCUENTO_15", False):
            merma_est = int(stock_fisico - demanda_critica)
            porcentaje_desc = 30 if hechos_inferidos.get("ACCION_PROMO_DESCUENTO_30", False) else 15
            precio_promocional = round(precio_unitario * (1 - porcentaje_desc / 100), 2)
            
            desc = (
                f"El stock físico ({stock_fisico} und.) supera la demanda proyectada ({demanda_critica:.1f} und.) "
                f"en los próximos {umbral_dias} días antes del vencimiento. Riesgo de merma de {merma_est} und."
            )
            accion = f"Aplicar promoción del {porcentaje_desc}% de descuento. Reducir precio a S/. {precio_promocional:.2f} para acelerar venta."
            
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "RIESGO_VENCIMIENTO",
                "gravedad": "ALTA" if porcentaje_desc == 30 else "MEDIA",
                "descripcion": desc,
                "accion_sugerida": accion,
                "merma_estimada": merma_est,
                "precio_oferta": precio_promocional
            })

        # 4.4 Alerta por Feriado Próximo
        if hechos_inferidos.get("ACCION_REFORZAR_FERIADO", False):
            demanda_fer = prediccion_diaria[prediccion_diaria["fecha"] == fecha_feriado_proximo]["demanda_predicha"].values[0]
            demanda_prom = prediccion_diaria["demanda_predicha"].mean()
            inc_porc = int(((demanda_fer / demanda_prom) - 1) * 100)
            
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "PICO_FERIADO",
                "gravedad": "BAJA",
                "descripcion": f"Se proyecta un pico de demanda de {demanda_fer:.1f} und. para el feriado de {fecha_feriado_proximo.strftime('%d/%m/%Y')} (incremento del {inc_porc}% sobre promedio).",
                "accion_sugerida": f"Abastecer un stock de seguridad adicional de al menos {int(demanda_fer * 0.4)} und. antes de esa fecha."
            })

        # 4.5 Alerta por Horas Pico
        if hechos_inferidos.get("ACCION_PREPARAR_GONDOLAS", False):
            horas_texto = ", ".join([f"{h}:00" for h in horas_pico])
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "HORA_PICO",
                "gravedad": "BAJA",
                "descripcion": f"El modelo MLP predice picos de ventas en las horas: {horas_texto}.",
                "accion_sugerida": f"Tener reabastecido el stock en góndola a partir de las {min(horas_pico)-1 if min(horas_pico) > 0 else 0}:30.",
                "horas_criticas": horas_pico
            })

        # 4.6 Alerta por Incertidumbre de Red Bayesiana
        if hechos_inferidos.get("RIESGO_BAYESIANO_CRITICO", False) or hechos_inferidos.get("RIESGO_BAYESIANO_ALTO", False):
            causas_texto = ", ".join([c.replace("_", " ") for c in causas_activas_bayesiana])
            if not causas_texto:
                causas_texto = "factores no modelados (probabilidad de fuga)"
            
            gravedad_bn = "ALTA" if hechos_inferidos.get("RIESGO_BAYESIANO_CRITICO", False) else "MEDIA"
            
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "RIESGO_BAYESIANO",
                "gravedad": gravedad_bn,
                "descripcion": f"La Red Bayesiana Noisy-OR estima una probabilidad de quiebre del {prob_quiebre_bayesiana*100:.1f}% basada en {len(causas_activas_bayesiana)} causas activas: {causas_texto}.",
                "accion_sugerida": f"Reforzar el abastecimiento preventivo. Factor de ajuste aplicado al stock de seguridad: x{factor_ajuste:.2f}.",
                "probabilidad_bayesiana": prob_quiebre_bayesiana,
                "causas_activas": causas_activas_bayesiana
            })

        return alertas


    def ejecutar_busqueda_categorias_bfs_dfs(self, df_inventario, funcion_filtro=None):
        """
        Ejecuta la búsqueda de productos en la estructura jerárquica de categorías
        usando los algoritmos BFS (Anchura) y DFS (Profundidad).
        """
        from modulos.busqueda import ArbolCategorias
        arbol = ArbolCategorias()
        arbol.construir_desde_dataframe(df_inventario)

        coincidencias_bfs, orden_bfs = arbol.bfs_buscar_productos(funcion_filtro)
        coincidencias_dfs, orden_dfs = arbol.dfs_buscar_productos(funcion_filtro)

        return {
            "bfs": {"coincidencias": coincidencias_bfs, "orden_visita": orden_bfs},
            "dfs": {"coincidencias": coincidencias_dfs, "orden_visita": orden_dfs},
            "arbol": arbol
        }

    def optimizar_logistica_y_pedidos_astar(self, df_inventario, lista_alertas):
        """
        Aplica el Algoritmo A* (A-Star) para:
        1. Priorización estratégica de pedidos con f(n) = g(n) + h(n).
        2. Optimización de la ruta de reabastecimiento en almacén usando Distancia Manhattan.
        """
        from modulos.busqueda import OptimizadorAEstrella
        optimizador = OptimizadorAEstrella(df_inventario=df_inventario)

        # 1. Priorización estratégica de órdenes de compra
        ordenes_priorizadas = optimizador.priorizar_ordenes_compra_astar(lista_alertas)

        # 2. Productos críticos a reabastecer
        productos_criticos = list(set([a.get("producto_nombre", a.get("producto_id")) for a in lista_alertas if a.get("gravedad") in ["ALTA", "CRITICA", "MEDIA"]]))
        if not productos_criticos:
            productos_criticos = list(df_inventario["producto_nombre"].head(4))

        # 3. Ruta de reabastecimiento en almacén con A*
        resultado_ruta = optimizador.resolver_ruta_reabastecimiento(productos_criticos)

        return {
            "ordenes_priorizadas_astar": ordenes_priorizadas,
            "ruta_reabastecimiento_astar": resultado_ruta
        }


