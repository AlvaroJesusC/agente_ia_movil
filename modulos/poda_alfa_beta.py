# -*- coding: utf-8 -*-

"""
Módulo de Poda Alfa-Beta para Pymevision AI
-------------------------------------------
Implementa la optimización de decisiones estratégicas de inventario mediante
el algoritmo Poda Alfa-Beta (Minimax con Poda Alpha-Beta) en condiciones de incertidumbre.

JUSTIFICACIÓN TEÓRICA DEL JUGADOR MIN (MERCADO / ENTORNO):
En la gestión de inventarios bajo incertidumbre, el mercado y los proveedores se modelan 
como un jugador adversarial (MIN) no porque tengan una intención real o maliciosa de perjudicar 
a la bodega, sino porque representa la metodología formal del Criterio Minimax de Decisión bajo Incertidumbre. 
Al asumir que el entorno seleccionará la combinación de eventos más desfavorable (picos de demanda, 
retrasos logísticos o crisis de inventario), el Agente (MAX) está forzado a planificar una estrategia 
ROBUSTAMENTE ÓPTIMA. Esta aproximación garantiza que, incluso ante el peor escenario posible, las pérdidas 
se minimizan y la continuidad operativa se protege de manera estable.

FACTOR DE DESCUENTO TEMPORAL (gamma = 0.95):
Se aplica un factor de descuento gamma = 0.95 a las utilidades de los días futuros dentro del árbol multietapa,
reflejando la preferencia estándar en modelos de decisión secuencial por resultados más cercanos en el tiempo
frente a proyecciones inciertas más lejanas.
"""

import math
import numpy as np
import pandas as pd


class AccionAgente:
    """Acciones estratégicas disponibles para el Agente (Jugador MAX)."""
    ORDEN_NORMAL = "ORDEN_NORMAL"
    ORDEN_EMERGENCIA = "ORDEN_EMERGENCIA"
    PROMO_DESCUENTO = "PROMO_DESCUENTO"
    MANTENER = "MANTENER"

    @classmethod
    def obtener_todas(cls):
        return [cls.ORDEN_NORMAL, cls.ORDEN_EMERGENCIA, cls.PROMO_DESCUENTO, cls.MANTENER]


class EscenarioEntorno:
    """Escenarios de incertidumbre del Mercado/Proveedor (Jugador MIN)."""
    ESTABLE = "ESTABLE"
    ALTA_DEMANDA = "ALTA_DEMANDA"
    RETRASO_PROVEEDOR = "RETRASO_PROVEEDOR"
    CRISIS_INVENTARIO = "CRISIS_INVENTARIO"

    @classmethod
    def obtener_todos(cls):
        return [cls.ESTABLE, cls.ALTA_DEMANDA, cls.RETRASO_PROVEEDOR, cls.CRISIS_INVENTARIO]


class EstadoInventario:
    """Representa la condición operativa actual de un producto."""
    def __init__(self, stock_actual, demanda_diaria_prom, costo_unidad, precio_venta, 
                 dias_vencimiento=999, es_perecedero=False, lead_time_base=2):
        self.stock_actual = float(stock_actual)
        self.demanda_diaria_prom = float(demanda_diaria_prom)
        self.costo_unidad = float(costo_unidad)
        self.precio_venta = float(precio_venta)
        self.dias_vencimiento = int(dias_vencimiento) if pd.notnull(dias_vencimiento) else 999
        self.es_perecedero = bool(es_perecedero)
        self.lead_time_base = int(lead_time_base)

    def copiar(self):
        return EstadoInventario(
            stock_actual=self.stock_actual,
            demanda_diaria_prom=self.demanda_diaria_prom,
            costo_unidad=self.costo_unidad,
            precio_venta=self.precio_venta,
            dias_vencimiento=self.dias_vencimiento,
            es_perecedero=self.es_perecedero,
            lead_time_base=self.lead_time_base
        )


class EvaluadorPodaAlfaBeta:
    """
    Motor del algoritmo Poda Alfa-Beta aplicado a la gestión óptima de inventario
    con transiciones de estado multietapa (múltiples días de horizonte).
    """
    def __init__(self, costo_almacenamiento_pct=0.02, penalizacion_quiebre_mult=1.5, factor_descuento=0.95):
        self.costo_almacenamiento_pct = costo_almacenamiento_pct
        self.penalizacion_quiebre_mult = penalizacion_quiebre_mult
        self.factor_descuento = factor_descuento
        self.nodos_visitados = 0
        self.nodos_podados = 0
        self.registro_arbol = {}

    def simular_paso_paso(self, estado_actual, accion, escenario):
        """
        Calcula la utilidad inmediata de 1 día ($U_t$) y genera el estado siguiente ($S_{t+1}$).
        """
        mult_demanda = 1.0
        dias_retraso = 0

        if escenario == EscenarioEntorno.ESTABLE:
            mult_demanda = 1.0
            dias_retraso = 0
        elif escenario == EscenarioEntorno.ALTA_DEMANDA:
            mult_demanda = 1.35
            dias_retraso = 0
        elif escenario == EscenarioEntorno.RETRASO_PROVEEDOR:
            mult_demanda = 1.0
            dias_retraso = 3
        elif escenario == EscenarioEntorno.CRISIS_INVENTARIO:
            mult_demanda = 1.40
            dias_retraso = 3

        demanda_dia = estado_actual.demanda_diaria_prom * mult_demanda
        precio_efectivo = estado_actual.precio_venta
        cantidad_pedido = 0

        if accion == AccionAgente.ORDEN_NORMAL:
            cantidad_pedido = max(10.0, demanda_dia * 1.5)
        elif accion == AccionAgente.ORDEN_EMERGENCIA:
            cantidad_pedido = max(15.0, demanda_dia * 2.2)
        elif accion == AccionAgente.PROMO_DESCUENTO:
            if estado_actual.es_perecedero and estado_actual.dias_vencimiento <= 7:
                precio_efectivo = estado_actual.precio_venta * 0.75  # 25% descuento por vencimiento próximo
                demanda_dia *= 1.35  # Aumento efectivo de rotación para mitigar mermas
            elif estado_actual.stock_actual > (estado_actual.demanda_diaria_prom * 5):
                precio_efectivo = estado_actual.precio_venta * 0.85
                demanda_dia *= 1.20
            else:
                precio_efectivo = estado_actual.precio_venta * 0.80  # Descuento no rentable sin riesgo de vencimiento
                demanda_dia *= 1.08
            cantidad_pedido = 0
        elif accion == AccionAgente.MANTENER:
            cantidad_pedido = 0

        lead_time_efectivo = estado_actual.lead_time_base + dias_retraso
        pedido_llega_hoy = (cantidad_pedido > 0) and (lead_time_efectivo <= 1)

        stock_disponible = estado_actual.stock_actual
        if pedido_llega_hoy:
            stock_disponible += cantidad_pedido

        ventas_realizadas = min(stock_disponible, demanda_dia)
        quiebre_stock = max(0.0, demanda_dia - stock_disponible)
        stock_remanente = max(0.0, stock_disponible - ventas_realizadas)

        merma_unidades = 0
        if estado_actual.es_perecedero and estado_actual.dias_vencimiento <= 1:
            merma_unidades = stock_remanente
            stock_remanente = 0.0

        ingresos = ventas_realizadas * precio_efectivo
        # Costo de Ventas (COGS) contable del periodo sobre unidades vendidas
        costo_ventas = ventas_realizadas * estado_actual.costo_unidad

        costo_pedido_fijo = 0.0
        if accion == AccionAgente.ORDEN_EMERGENCIA and cantidad_pedido > 0:
            costo_pedido_fijo = cantidad_pedido * estado_actual.costo_unidad * 0.15  # Recargo logístico por envío rápido

        costo_almacenamiento = stock_remanente * estado_actual.costo_unidad * self.costo_almacenamiento_pct
        penalizacion_quiebre = quiebre_stock * (estado_actual.precio_venta * self.penalizacion_quiebre_mult)
        costo_merma = merma_unidades * estado_actual.costo_unidad

        utilidad_paso = ingresos - costo_ventas - costo_pedido_fijo - costo_almacenamiento - penalizacion_quiebre - costo_merma

        # Generar Estado Siguiente para el Día t+1
        nuevo_stock = stock_remanente
        if cantidad_pedido > 0 and not pedido_llega_hoy:
            nuevo_stock += cantidad_pedido  # Arribo diferido de pedidos

        estado_siguiente = estado_actual.copiar()
        estado_siguiente.stock_actual = nuevo_stock
        if estado_siguiente.es_perecedero:
            estado_siguiente.dias_vencimiento = max(0, estado_siguiente.dias_vencimiento - 1)

        return round(utilidad_paso, 2), estado_siguiente

    def obtener_acciones_ordenadas(self, estado):
        """Ordenamiento heurístico de acciones para acelerar cutoffs en Alpha-Beta."""
        if estado.stock_actual < (estado.demanda_diaria_prom * (estado.lead_time_base + 1)):
            return [AccionAgente.ORDEN_NORMAL, AccionAgente.ORDEN_EMERGENCIA, AccionAgente.MANTENER, AccionAgente.PROMO_DESCUENTO]
        elif estado.es_perecedero and estado.dias_vencimiento <= 7:
            return [AccionAgente.PROMO_DESCUENTO, AccionAgente.MANTENER, AccionAgente.ORDEN_NORMAL, AccionAgente.ORDEN_EMERGENCIA]
        else:
            return [AccionAgente.MANTENER, AccionAgente.PROMO_DESCUENTO, AccionAgente.ORDEN_NORMAL, AccionAgente.ORDEN_EMERGENCIA]

    def alfa_beta_minimax(self, estado, profundidad, alfa, beta, es_maximizando, accion_actual=None, ruta_nodo=None):
        """
        Ejecuta la Poda Alfa-Beta con evaluación recursiva multietapa (profundidad real >= 2).
        """
        if ruta_nodo is None:
            ruta_nodo = ("RAIZ",)

        self.nodos_visitados += 1

        # Registrar nodo actual
        self.registro_arbol[ruta_nodo] = {
            "tipo": "MAX" if es_maximizando else "MIN",
            "alfa_in": alfa,
            "beta_in": beta,
            "valor": None,
            "visitado": True,
            "podado": False
        }

        if profundidad == 0:
            self.registro_arbol[ruta_nodo]["valor"] = 0.0
            return 0.0, None

        if es_maximizando:
            max_eval = -math.inf
            mejor_accion = None
            acciones = self.obtener_acciones_ordenadas(estado)

            poda_ocurrida = False
            for accion in acciones:
                ruta_hijo = ruta_nodo + (accion,)
                if poda_ocurrida:
                    # Registrar como podado (no visitado)
                    self.registro_arbol[ruta_hijo] = {
                        "tipo": "MIN",
                        "alfa_in": alfa,
                        "beta_in": beta,
                        "valor": None,
                        "visitado": False,
                        "podado": True
                    }
                    continue

                # Simular respuesta adversarial de MIN ante la acción de MAX
                eval_nodo, _ = self.alfa_beta_minimax(
                    estado, profundidad, alfa, beta, es_maximizando=False, 
                    accion_actual=accion, ruta_nodo=ruta_hijo
                )

                if eval_nodo > max_eval:
                    max_eval = eval_nodo
                    mejor_accion = accion

                alfa = max(alfa, eval_nodo)
                if beta <= alfa:
                    self.nodos_podados += 1
                    poda_ocurrida = True  # Poda Alfa (Alpha Cutoff)

            self.registro_arbol[ruta_nodo]["valor"] = max_eval
            self.registro_arbol[ruta_nodo]["alfa_out"] = alfa
            self.registro_arbol[ruta_nodo]["beta_out"] = beta
            return max_eval, mejor_accion
        else:
            min_eval = math.inf
            escenarios = EscenarioEntorno.obtener_todos()

            poda_ocurrida = False
            for escenario in escenarios:
                ruta_hijo = ruta_nodo + (escenario,)
                if poda_ocurrida:
                    self.registro_arbol[ruta_hijo] = {
                        "tipo": "MAX",
                        "alfa_in": alfa,
                        "beta_in": beta,
                        "valor": None,
                        "visitado": False,
                        "podado": True
                    }
                    continue

                # 1. Utilidad del día actual + Transición al día siguiente
                utilidad_paso, estado_sig = self.simular_paso_paso(estado, accion_actual, escenario)

                # 2. Si hay más días por evaluar (profundidad > 1), continuar recursión en MAX para el día t+1
                if profundidad > 1:
                    val_futuro, _ = self.alfa_beta_minimax(
                        estado_sig, profundidad - 1, alfa, beta, es_maximizando=True, 
                        ruta_nodo=ruta_hijo
                    )
                    eval_acumulada = utilidad_paso + (self.factor_descuento * val_futuro)
                    # Guardar el valor acumulado (U_1 + gamma * U_futura) en el nodo hijo
                    if ruta_hijo in self.registro_arbol:
                        self.registro_arbol[ruta_hijo]["valor"] = eval_acumulada
                else:
                    eval_acumulada = utilidad_paso
                    self.registro_arbol[ruta_hijo] = {
                        "tipo": "MAX",
                        "alfa_in": alfa,
                        "beta_in": beta,
                        "valor": eval_acumulada,
                        "visitado": True,
                        "podado": False
                    }

                if eval_acumulada < min_eval:
                    min_eval = eval_acumulada

                beta = min(beta, eval_acumulada)
                if beta <= alfa:
                    self.nodos_podados += 1
                    poda_ocurrida = True  # Poda Beta (Beta Cutoff)

            self.registro_arbol[ruta_nodo]["valor"] = min_eval
            self.registro_arbol[ruta_nodo]["alfa_out"] = alfa
            self.registro_arbol[ruta_nodo]["beta_out"] = beta
            return min_eval, None

    def optimizar_decision(self, producto_id, producto_nombre, estado_inventario, profundidad=2):
        """
        Punto de entrada principal para evaluar y optimizar la decisión de un producto
        a una profundidad especificada (ej. 2 o 3 niveles multietapa).
        """
        self.nodos_visitados = 0
        self.nodos_podados = 0
        self.registro_arbol = {}

        # Matriz completa de utilidades inmediatas (Día 1) para análisis explicativo
        tabla_evaluacion = []
        acciones = AccionAgente.obtener_todas()
        escenarios = EscenarioEntorno.obtener_todos()

        for acc in acciones:
            fila = {"accion": acc}
            for esc in escenarios:
                u_inmediata, _ = self.simular_paso_paso(estado_inventario, acc, esc)
                fila[esc] = u_inmediata
            tabla_evaluacion.append(fila)

        df_matriz = pd.DataFrame(tabla_evaluacion)

        # Ejecución del Minimax Alfa-Beta Multietapa
        utilidad_optima, mejor_accion = self.alfa_beta_minimax(
            estado=estado_inventario,
            profundidad=profundidad,
            alfa=-math.inf,
            beta=math.inf,
            es_maximizando=True
        )

        eficiencia_poda_pct = round((self.nodos_podados / max(1, self.nodos_visitados + self.nodos_podados)) * 100, 1)

        # Determinar justificación estratégica
        justificacion = ""
        if mejor_accion == AccionAgente.ORDEN_EMERGENCIA:
            justificacion = "Riesgo crítico de quiebre de stock detectado bajo escenarios de alta demanda o retraso."
        elif mejor_accion == AccionAgente.ORDEN_NORMAL:
            justificacion = "Demanda proyectada requiere reabastecimiento estándar para mantener stock de seguridad."
        elif mejor_accion == AccionAgente.PROMO_DESCUENTO:
            justificacion = "Riesgo de merma por vencimiento o exceso de inventario; se recomienda acelerar rotación."
        else:
            justificacion = "Nivel de stock actual cubre la demanda esperada sin generar costos innecesarios."

        return {
            "producto_id": producto_id,
            "producto_nombre": producto_nombre,
            "mejor_accion": mejor_accion,
            "utilidad_optima": round(utilidad_optima, 2),
            "justificacion": justificacion,
            "nodos_visitados": self.nodos_visitados,
            "nodos_podados": self.nodos_podados,
            "eficiencia_poda_pct": eficiencia_poda_pct,
            "matriz_evaluacion": df_matriz,
            "profundidad_evaluada": profundidad,
            "registro_arbol": self.registro_arbol.copy()
        }

    def generar_traza_explicativa(self, producto_nombre, estado_inventario, profundidad=3):
        """
        Modo de Depuración Numerica Paso a Paso (Debug Mode).
        Genera el desglose completo del árbol de decisión a N días para comparar
        las acciones del Día 1 y verificar cuantitativamente por qué Minimax elige la mejor acción.
        """
        traza = []
        acciones = AccionAgente.obtener_todas()
        escenarios = EscenarioEntorno.obtener_todos()

        for acc in acciones:
            peor_total = math.inf
            mejor_esc_info = None

            for esc1 in escenarios:
                u1, s2 = self.simular_paso_paso(estado_inventario.copiar(), acc, esc1)
                val_sub, act2 = self.alfa_beta_minimax(s2.copiar(), profundidad=profundidad-1, alfa=-math.inf, beta=math.inf, es_maximizando=True)
                tot_acum = u1 + (self.factor_descuento * val_sub)

                info_esc = {
                    "accion_d1": acc,
                    "escenario_d1": esc1,
                    "u1_inmediata": round(u1, 2),
                    "stock_s2": round(s2.stock_actual, 1),
                    "accion_d2_optima": act2 or "MANTENER",
                    "subarbol_d2_d3": round(val_sub, 2),
                    "total_acumulado": round(tot_acum, 2)
                }

                if tot_acum < peor_total:
                    peor_total = tot_acum
                    mejor_esc_info = info_esc

            if mejor_esc_info:
                traza.append(mejor_esc_info)

        df_traza = pd.DataFrame(traza)
        return df_traza
