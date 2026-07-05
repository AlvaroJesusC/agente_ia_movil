# -*- coding: utf-8 -*-

# Módulo Actuador para pymevision ai
# Evaluación de riesgos de inventario y recomendaciones de abastecimiento.

import pandas as pd
import numpy as np

class ActuadorOptimizado:
    # Evalúa riesgos de stock y genera recomendaciones.
    
    def __init__(self):
        # Los umbrales de alerta de vencimiento se calculan dinámicamente
        # según la columna 'vida_util_dias' de cada producto en el inventario.
        pass
        
    def evaluar_producto(self, fila_inventario, prediccion_diaria, patron_horario, feriados_proximos, red_bayesiana=None):
        # analiza el estado del inventario para generar alertas de quiebre, vencimiento, feriados y horas pico.
        # Opcionalmente utiliza la Red Bayesiana Mixta (Noisy-OR) para análisis causal de incertidumbre.
        producto_id = fila_inventario["producto_id"]
        nombre_producto = fila_inventario["producto_nombre"]
        stock_fisico = fila_inventario["stock_fisico"]
        stock_transito = fila_inventario["stock_transito"]
        tiempo_reposicion = int(fila_inventario["tiempo_reposicion_dias"])
        cant_min_pedido = int(fila_inventario["cantidad_minima_pedido"])
        es_perecedero = fila_inventario["es_perecedero"]
        precio_unitario = fila_inventario["precio_unitario"]
        
        alertas = []
        
        # riesgo de quiebre de stock
        prediccion_reposicion = prediccion_diaria.head(tiempo_reposicion)
        demanda_lead_time = prediccion_reposicion["demanda_predicha"].sum()
        stock_seguridad = max(2, int(demanda_lead_time * 0.20))
        
        # Ajuste dinámico del stock de seguridad usando Red Bayesiana
        factor_bayesiano = 1.0
        resultado_bayesiano = None
        if red_bayesiana is not None and red_bayesiana.entrenado:
            demanda_promedio = prediccion_diaria["demanda_predicha"].mean()
            resultado_bayesiano = red_bayesiana.predecir_riesgo_quiebre(
                stock_fisico=stock_fisico,
                demanda_predicha=demanda_lead_time / max(tiempo_reposicion, 1),
                demanda_promedio=demanda_promedio,
                precipitacion=0.5,  # Valor promedio por defecto para proyección
                es_feriado=bool(fila_inventario.get("es_feriado", False)),
                es_fin_semana=bool(fila_inventario.get("es_fin_semana", False)),
                tiempo_reposicion=tiempo_reposicion,
                stock_seguridad=stock_seguridad
            )
            # Si la probabilidad bayesiana es alta, incrementar stock de seguridad
            prob_quiebre = resultado_bayesiano["probabilidad_quiebre"]
            if prob_quiebre >= 0.70:
                factor_bayesiano = 1.50  # +50% stock de seguridad
            elif prob_quiebre >= 0.45:
                factor_bayesiano = 1.25  # +25% stock de seguridad
            
            stock_seguridad = max(2, int(stock_seguridad * factor_bayesiano))
        
        recurso_total = stock_fisico + stock_transito
        
        if recurso_total < (demanda_lead_time + stock_seguridad):
            diferencia = (demanda_lead_time + stock_seguridad) - recurso_total
            cantidad_pedido = int(np.ceil(diferencia / cant_min_pedido) * cant_min_pedido)
            
            alerta_quiebre = {
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "RIESGO_QUIEBRE",
                "gravedad": "ALTA" if recurso_total < demanda_lead_time else "MEDIA",
                "descripcion": f"El stock disponible ({recurso_total} und.) no cubrirá la demanda proyectada ({demanda_lead_time:.1f} und.) más stock de seguridad ({stock_seguridad} und.) durante los {tiempo_reposicion} días de reposición.",
                "accion_sugerida": f"Generar pedido automático de {cantidad_pedido} unidades al proveedor.",
                "cantidad_pedido": cantidad_pedido
            }
            if resultado_bayesiano is not None:
                alerta_quiebre["probabilidad_bayesiana"] = resultado_bayesiano["probabilidad_quiebre"]
            alertas.append(alerta_quiebre)
            
        # riesgo de vencimiento
        # Se calcula de manera dinámica:
        # - Para productos de corta vida útil (<= 7 días), el umbral es de 2 días.
        # - Para vida útil intermedia (<= 15 días), el umbral es de 4 días.
        # - Para vida útil larga, el umbral es el 25% de su vida útil.
        vida_util_raw = fila_inventario.get("vida_util_dias", 30)
        vida_util = int(float(vida_util_raw)) if pd.notna(vida_util_raw) else 30
        
        if vida_util <= 7:
            umbral_dias = 2
        elif vida_util <= 15:
            umbral_dias = 4
        else:
            umbral_dias = max(5, int(vida_util * 0.25))
        if es_perecedero:
            prediccion_vencimiento = prediccion_diaria.head(umbral_dias)
            demanda_critica = prediccion_vencimiento["demanda_predicha"].sum()
            
            if stock_fisico > demanda_critica and stock_fisico > 5:
                merma_estimada = int(stock_fisico - demanda_critica)
                porcentaje_descuento = 30 if merma_estimada > (stock_fisico * 0.5) else 15
                precio_promocional = round(precio_unitario * (1 - porcentaje_descuento / 100), 2)
                
                alertas.append({
                    "producto_id": producto_id,
                    "producto_nombre": nombre_producto,
                    "tipo": "RIESGO_VENCIMIENTO",
                    "gravedad": "ALTA" if merma_estimada > (stock_fisico * 0.5) else "MEDIA",
                    "descripcion": f"Stock físico ({stock_fisico} und.) supera la demanda proyectada ({demanda_critica:.1f} und.) para los próximos {umbral_dias} días antes del vencimiento. Riesgo de merma de {merma_estimada} und.",
                    "accion_sugerida": f"Aplicar promoción del {porcentaje_descuento}% de descuento. Reducir precio a S/. {precio_promocional:.2f} para acelerar venta.",
                    "merma_estimada": merma_estimada,
                    "precio_oferta": precio_promocional
                })
                
        # picos de demanda por feriados
        for fecha_f, es_feriado_f in feriados_proximos:
            if es_feriado_f:
                demanda_promedio = prediccion_diaria["demanda_predicha"].mean()
                demanda_feriado = prediccion_diaria[prediccion_diaria["fecha"] == fecha_f]["demanda_predicha"].values
                
                if len(demanda_feriado) > 0:
                    demanda_fer = demanda_feriado[0]
                    if demanda_fer > (demanda_promedio * 1.30):
                        incremento_porcentaje = int(((demanda_fer / demanda_promedio) - 1) * 100)
                        
                        alertas.append({
                            "producto_id": producto_id,
                            "producto_nombre": nombre_producto,
                            "tipo": "PICO_FERIADO",
                            "gravedad": "BAJA",
                            "descripcion": f"Se proyecta un pico de demanda de {demanda_fer:.1f} und. para el feriado del {fecha_f.strftime('%d/%m/%Y')} (incremento del {incremento_porcentaje}% sobre promedio).",
                            "accion_sugerida": f"Abastecer un stock de seguridad adicional de al menos {int(demanda_fer * 0.4)} und. antes de esa fecha."
                        })
                        
        # horas pico del día
        umbral_hora = np.percentile(patron_horario, 85)
        horas_pico = [h for h in range(24) if patron_horario[h] > umbral_hora and patron_horario[h] > 1.5]
        
        if len(horas_pico) > 0:
            horas_texto = ", ".join([f"{h}:00" for h in horas_pico])
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "HORA_PICO",
                "gravedad": "BAJA",
                "descripcion": f"El modelo detecta picos de afluencia y ventas en las horas: {horas_texto}.",
                "accion_sugerida": f"Tener listo y reabastecido el stock en góndola a partir de las {min(horas_pico)-1 if min(horas_pico) > 0 else 0}:30.",
                "horas_criticas": horas_pico
            })
        
        # Análisis causal con Red Bayesiana Mixta (Noisy-OR)
        if resultado_bayesiano is not None and resultado_bayesiano["probabilidad_quiebre"] >= 0.40:
            causas_texto = ", ".join([c.replace("_", " ") for c in resultado_bayesiano["causas_activas"]])
            if not causas_texto:
                causas_texto = "factores no modelados (probabilidad de fuga)"
            
            gravedad_bn = "ALTA" if resultado_bayesiano["nivel_riesgo"] == "CRITICO" else (
                "MEDIA" if resultado_bayesiano["nivel_riesgo"] == "ALTO" else "BAJA"
            )
            
            alertas.append({
                "producto_id": producto_id,
                "producto_nombre": nombre_producto,
                "tipo": "RIESGO_BAYESIANO",
                "gravedad": gravedad_bn,
                "descripcion": f"La Red Bayesiana (Noisy-OR) estima una probabilidad de quiebre del {resultado_bayesiano['probabilidad_quiebre']*100:.1f}% basada en {resultado_bayesiano['n_causas_activas']} causas activas: {causas_texto}.",
                "accion_sugerida": f"Reforzar el abastecimiento preventivo. Factor de ajuste aplicado al stock de seguridad: x{factor_bayesiano:.2f}.",
                "probabilidad_bayesiana": resultado_bayesiano["probabilidad_quiebre"],
                "causas_activas": resultado_bayesiano["causas_activas"]
            })
            
        return alertas

    def ejecutar_busqueda_categorias_bfs_dfs(self, df_inventario, funcion_filtro=None):
        """
        Ejecuta la búsqueda de productos en la estructura jerárquica de categorías
        usando los algoritmos BFS (Anchura) y DFS (Profundidad).
        """
        from busqueda import ArbolCategorias
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
        from busqueda import OptimizadorAEstrella
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


