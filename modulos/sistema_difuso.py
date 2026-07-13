# -*- coding: utf-8 -*-
"""
MÓDULO DE SISTEMA DIFUSO — PYMEVISION AI
------------------------------------------
Implementa un controlador difuso Mamdani desde cero utilizando NumPy.
Permite evaluar:
1. Urgencia de Reabastecimiento (0% - 100%) basada en la relación Stock/Demanda.
2. Descuento Sugerido de Liquidación (0% - 50%) para productos perecederos según inventario y días para vencer.
"""

import numpy as np

def triangular(x, a, b, c):
    """Calcula la función de membresía triangular."""
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif x == b:
        return 1.0
    else: # b < x < c
        return (c - x) / (c - b)

def trapezoidal(x, a, b, c, d):
    """Calcula la función de membresía trapezoidal."""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else: # c < x < d
        return (d - x) / (d - c)


class ControladorDifuso:
    """
    Controlador Lógico Difuso Mamdani para la optimización de inventarios.
    """
    def __init__(self):
        # Rangos de discretización para la defuzzificación (100 puntos)
        self.rango_urgencia = np.linspace(0, 100, 100)
        self.rango_descuento = np.linspace(0, 50, 100)

    # --- Funciones de membresía de entrada ---
    def membresia_stock(self, x):
        """Membresía del stock ratio (%)."""
        # Truncar x para evitar valores extremos fuera de la escala
        x_eval = min(x, 200.0)
        return {
            "bajo": trapezoidal(x_eval, 0.0, 0.0, 30.0, 60.0),
            "medio": triangular(x_eval, 40.0, 60.0, 90.0),
            "alto": trapezoidal(x_eval, 75.0, 100.0, 200.0, 200.0)
        }

    def membresia_vencimiento(self, x):
        """Membresía de los días para el vencimiento."""
        x_eval = min(x, 30.0)
        return {
            "critico": trapezoidal(x_eval, 0.0, 0.0, 3.0, 7.0),
            "cercano": triangular(x_eval, 5.0, 10.0, 15.0),
            "seguro": trapezoidal(x_eval, 12.0, 20.0, 30.0, 30.0)
        }

    # --- Funciones de membresía de salida ---
    def salida_urgencia(self, y):
        """Membresía de la urgencia de pedido (%)."""
        return {
            "nula": trapezoidal(y, 0.0, 0.0, 20.0, 45.0),
            "moderada": triangular(y, 30.0, 50.0, 75.0),
            "critica": trapezoidal(y, 60.0, 80.0, 100.0, 100.0)
        }

    def salida_descuento(self, y):
        """Membresía del descuento sugerido (%)."""
        return {
            "ninguno": trapezoidal(y, 0.0, 0.0, 5.0, 10.0),
            "moderado": triangular(y, 5.0, 15.0, 25.0),
            "agresivo": trapezoidal(y, 20.0, 35.0, 50.0, 50.0)
        }

    def evaluar_producto(self, stock_actual, demanda_estimada, dias_vencimiento=999, es_perecedero=False):
        """
        Ejecuta la inferencia difusa Mamdani para un producto específico.
        
        Retorna:
            resultado: Diccionario con la urgencia y descuento calculados + traza detallada.
        """
        # 1. Normalizar Stock Ratio
        # El nivel óptimo de inventario para cubrir el periodo es la demanda_estimada.
        if demanda_estimada <= 0:
            stock_ratio = 100.0
        else:
            stock_ratio = (stock_actual / demanda_estimada) * 100.0
        
        # 2. Fuzzificación
        m_stock = self.membresia_stock(stock_ratio)
        
        # Si no es perecedero, el vencimiento es seguro
        dias_vence = dias_vencimiento if (es_perecedero and dias_vencimiento is not None) else 999
        m_vence = self.membresia_vencimiento(min(dias_vence, 30))

        # 3. Motor de Inferencia (Reglas Difusas)
        # Reglas para Urgencia de Reabastecimiento
        r1_urg_critica = m_stock["bajo"]
        r2_urg_moderada = m_stock["medio"]
        r3_urg_nula = m_stock["alto"]

        # Reglas para Descuento Sugerido (solo perecederos y si hay excedente de stock)
        if es_perecedero:
            r4_desc_agresivo = min(m_stock["alto"], m_vence["critico"])
            r5_desc_moderado = min(m_stock["medio"], m_vence["critico"])
            r6_desc_moderado = max(r5_desc_moderado, min(m_stock["alto"], m_vence["cercano"]))
            r7_desc_ninguno = m_vence["seguro"]
            r8_desc_ninguno = max(r7_desc_ninguno, m_stock["bajo"])  # No descontar si el stock es bajo
        else:
            r4_desc_agresivo = 0.0
            r5_desc_moderado = 0.0
            r6_desc_moderado = 0.0
            r7_desc_ninguno = 1.0
            r8_desc_ninguno = 1.0

        # Agregar los consecuentes
        urg_critica = r1_urg_critica
        urg_moderada = r2_urg_moderada
        urg_nula = r3_urg_nula

        desc_agresivo = r4_desc_agresivo
        desc_moderado = r6_desc_moderado
        desc_ninguno = max(r7_desc_ninguno, r8_desc_ninguno)

        # 4. Defuzzificación por el Método del Centroide (Centro de Gravedad)
        # Para Urgencia
        numerador_urg = 0.0
        denominador_urg = 0.0
        for y in self.rango_urgencia:
            m_salida = self.salida_urgencia(y)
            # Implicación (recorte por min) y Agregación (max)
            membresia_y = max(
                min(m_salida["nula"], urg_nula),
                min(m_salida["moderada"], urg_moderada),
                min(m_salida["critica"], urg_critica)
            )
            numerador_urg += y * membresia_y
            denominador_urg += membresia_y

        urgencia_pedido = (numerador_urg / denominador_urg) if denominador_urg > 0 else 0.0

        # Para Descuento
        numerador_desc = 0.0
        denominador_desc = 0.0
        for y in self.rango_descuento:
            m_salida = self.salida_descuento(y)
            membresia_y = max(
                min(m_salida["ninguno"], desc_ninguno),
                min(m_salida["moderado"], desc_moderado),
                min(m_salida["agresivo"], desc_agresivo)
            )
            numerador_desc += y * membresia_y
            denominador_desc += membresia_y

        descuento_sugerido = (numerador_desc / denominador_desc) if denominador_desc > 0 else 0.0

        return {
            "stock_ratio": round(stock_ratio, 2),
            "dias_vencimiento": dias_vence,
            "es_perecedero": es_perecedero,
            "membresia_stock": m_stock,
            "membresia_vencimiento": m_vence,
            "urgencia_pedido_pct": round(urgencia_pedido, 2),
            "descuento_sugerido_pct": round(descuento_sugerido, 2),
            "reglas_urgencia": {
                "critica": round(urg_critica, 2),
                "moderada": round(urg_moderada, 2),
                "nula": round(urg_nula, 2)
            },
            "reglas_descuento": {
                "agresivo": round(desc_agresivo, 2),
                "moderado": round(desc_moderado, 2),
                "ninguno": round(desc_ninguno, 2)
            }
        }
