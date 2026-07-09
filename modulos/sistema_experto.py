# -*- coding: utf-8 -*-
"""
MÓDULO DE SISTEMA EXPERTO — PYMEVISION AI
------------------------------------------
Este módulo implementa un Sistema Experto formal basado en Reglas de Producción
con un Motor de Inferencia por Encadenamiento hacia Adelante (Forward Chaining),
desacoplando completamente la Base de Conocimientos de la lógica de ejecución.

COMPONENTE 1: BASE DE HECHOS (Representa la memoria de trabajo)
COMPONENTE 2: BASE DE CONOCIMIENTOS (Reglas lógicas de tipo IF-THEN)
COMPONENTE 3: MOTOR DE INFERENCIA (Algoritmo de encadenamiento recursivo)
"""

class Hecho:
    """
    Representa una afirmación verdadera (Fact) dentro de la Base de Hechos
    (memoria de trabajo del Sistema Experto).
    """
    def __init__(self, nombre, valor=True, descripcion=""):
        self.nombre = nombre
        self.valor = valor
        self.descripcion = descripcion

    def __repr__(self):
        return f"Fact({self.nombre}={self.valor})"


class Regla:
    """
    Representa una Regla de Producción en la Base de Conocimientos.
    Estructura clásica:
    IF (Antecedente_1 AND Antecedente_2 AND ...) THEN Consecuente
    """
    def __init__(self, nombre, antecedentes, consecuente, descripcion, accion_sugerida, gravedad="MEDIA"):
        self.nombre = nombre
        self.antecedentes = antecedentes  # Lista de nombres de hechos que deben ser verdaderos
        self.consecuente = consecuente    # Nombre del hecho resultante que se inferirá
        self.descripcion = descripcion    # Texto explicativo para el reporte del usuario
        self.accion_sugerida = accion_sugerida  # Acción correctiva
        self.gravedad = gravedad          # Severidad: CRITICA, ALTA, MEDIA, BAJA

    def __repr__(self):
        antecedentes_str = " AND ".join(self.antecedentes)
        return f"Rule {self.nombre}: IF ({antecedentes_str}) THEN {self.consecuente}"


class MotorInferenciaForward:
    """
    Motor de Inferencia que implementa el algoritmo de Encadenamiento hacia Adelante
    (Forward Chaining).
    Busca de manera sistemática reglas cuyos antecedentes se cumplan,
    agregando sus consecuentes a la Base de Hechos hasta que no sea posible deducir nada nuevo.
    """
    def __init__(self, base_conocimientos):
        self.base_conocimientos = base_conocimientos
        self.traza_inferencia = []

    def inferir(self, hechos_iniciales):
        """
        Ejecuta el encadenamiento hacia adelante sobre la lista de hechos iniciales.
        
        Parámetros:
            hechos_iniciales: Lista de objetos Hecho.
            
        Retorna:
            hechos_finales: Diccionario {nombre_hecho: valor} con todos los hechos deducidos.
            alertas_generadas: Lista de diccionarios que describen las alertas y acciones inferidas.
        """
        # Inicializar la memoria de trabajo (Base de Hechos)
        hechos_activos = {h.nombre: h.valor for h in hechos_iniciales}
        reglas_disparadas = set()
        alertas_generadas = []
        self.traza_inferencia = []
        
        self.traza_inferencia.append("=== INICIANDO MOTOR DE INFERENCIA (FORWARD CHAINING) ===")
        self.traza_inferencia.append(f"Hechos iniciales en memoria de trabajo: {list(hechos_activos.keys())}")
        
        ciclo = 1
        cambio = True
        
        while cambio:
            cambio = False
            self.traza_inferencia.append(f"\n--- Ciclo de Evaluación #{ciclo} ---")
            
            for regla in self.base_conocimientos:
                # Si la regla ya fue disparada, la ignoramos para evitar bucles infinitos
                if regla.nombre in reglas_disparadas:
                    continue
                
                # Evaluar antecedentes: Todos deben estar en hechos_activos y ser verdaderos
                antecedentes_satisfechos = True
                for ant in regla.antecedentes:
                    if ant not in hechos_activos or not hechos_activos[ant]:
                        antecedentes_satisfechos = False
                        break
                
                if antecedentes_satisfechos:
                    # ¡FUEGO! Disparamos la regla
                    hechos_activos[regla.consecuente] = True
                    reglas_disparadas.add(regla.nombre)
                    cambio = True
                    
                    self.traza_inferencia.append(
                        f"  ✔ [DISPARADA] {regla.nombre}: "
                        f"Antecedentes {regla.antecedentes} satisfechos. "
                        f"Infiriendo hecho: [{regla.consecuente}]"
                    )
                    
                    alertas_generadas.append({
                        "regla": regla.nombre,
                        "tipo": regla.consecuente,
                        "gravedad": regla.gravedad,
                        "descripcion": regla.descripcion,
                        "accion_sugerida": regla.accion_sugerida
                    })
            
            if not cambio:
                self.traza_inferencia.append("  [FIN] No se dispararon nuevas reglas. Memoria de trabajo estabilizada.")
            
            ciclo += 1
            
        self.traza_inferencia.append("\n=== INFERENCIA COMPLETADA CON ÉXITO ===")
        self.traza_inferencia.append(f"Hechos finales en memoria: {list(hechos_activos.keys())}")
        self.traza_inferencia.append(f"Alertas/Recomendaciones deducidas: {len(alertas_generadas)}")
        
        return hechos_activos, alertas_generadas


def inicializar_base_conocimientos():
    """
    Define y retorna la Base de Conocimientos del Sistema Experto de Inventarios de Pymevision AI.
    Contiene las reglas de producción para quiebre, mermas, feriados, horas pico y riesgo bayesiano.
    """
    base_reglas = [
        # --- REGLAS PARA QUIEBRE DE STOCK ---
        Regla(
            nombre="R1_RIESGO_QUIEBRE_CRITICO",
            antecedentes=["recurso_criticamente_bajo", "tiempo_reposicion_inminente"],
            consecuente="RIESGO_QUIEBRE_CRITICO",
            descripcion="El inventario total disponible es menor que la demanda proyectada para cubrir el lead time y el stock físico está casi agotado.",
            accion_sugerida="Generar pedido automático prioritario urgente al proveedor.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R2_RIESGO_QUIEBRE_MODERADO",
            antecedentes=["recurso_bajo_seguridad", "tiempo_reposicion_estandar"],
            consecuente="RIESGO_QUIEBRE_MODERADO",
            descripcion="El stock acumulado es menor que la demanda esperada más el colchón del stock de seguridad durante el tiempo de reposición.",
            accion_sugerida="Generar orden de abastecimiento estándar para restablecer niveles óptimos.",
            gravedad="MEDIA"
        ),
        Regla(
            nombre="R3_ORDEN_EMERGENCIA",
            antecedentes=["RIESGO_QUIEBRE_CRITICO"],
            consecuente="ACCION_ORDEN_EMERGENCIA",
            descripcion="Inferencia lógica de acción urgente debido al peligro inmediato de rotura de stock.",
            accion_sugerida="Disparar pedido logístico de emergencia con flete rápido.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R4_ORDEN_NORMAL",
            antecedentes=["RIESGO_QUIEBRE_MODERADO"],
            consecuente="ACCION_ORDEN_NORMAL",
            descripcion="Inferencia de pedido preventivo de reabastecimiento estándar.",
            accion_sugerida="Emitir orden de compra estándar según cantidad mínima requerida.",
            gravedad="MEDIA"
        ),

        # --- REGLAS PARA PRODUCTOS PERECEDEROS (VENCIMIENTO) ---
        Regla(
            nombre="R5_RIESGO_VENCIMIENTO_CRITICO",
            antecedentes=["es_perecedero", "stock_excedente_critico", "vencimiento_muy_cercano"],
            consecuente="RIESGO_VENCIMIENTO_CRITICO",
            descripcion="Stock físico supera la demanda total antes de vencer. Riesgo inminente de pérdida financiera por merma.",
            accion_sugerida="Aplicar descuento agresivo del 30% de inmediato y reubicar en góndola principal.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R6_RIESGO_VENCIMIENTO_MODERADO",
            antecedentes=["es_perecedero", "stock_excedente_moderado", "vencimiento_moderadamente_cercano"],
            consecuente="RIESGO_VENCIMIENTO_MODERADO",
            descripcion="El stock actual supera levemente las proyecciones antes de expirar.",
            accion_sugerida="Aplicar promoción preventiva del 15% de descuento para acelerar la rotación.",
            gravedad="MEDIA"
        ),
        Regla(
            nombre="R7_PROMO_VENTA_RAPIDA_30",
            antecedentes=["RIESGO_VENCIMIENTO_CRITICO"],
            consecuente="ACCION_PROMO_DESCUENTO_30",
            descripcion="Acción inferida comercial extrema para mitigar pérdidas por merma.",
            accion_sugerida="Reducir precio al 30% de descuento en la etiqueta comercial.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R8_PROMO_VENTA_RAPIDA_15",
            antecedentes=["RIESGO_VENCIMIENTO_MODERADO"],
            consecuente="ACCION_PROMO_DESCUENTO_15",
            descripcion="Acción inferida comercial de estimulación moderada de la demanda.",
            accion_sugerida="Ofrecer descuento del 15% en cartelera de tienda.",
            gravedad="MEDIA"
        ),

        # --- REGLAS PARA EVENTOS Y FERIADOS ---
        Regla(
            nombre="R9_PICO_DEMANDA_FERIADO",
            antecedentes=["feriado_proximo_detectado", "demanda_feriado_superior_promedio"],
            consecuente="RIESGO_PICO_FERIADO",
            descripcion="Se proyecta un incremento extraordinario en las ventas debido a festividades nacionales o eventos especiales.",
            accion_sugerida="Preparar un colchón de stock de seguridad adicional antes del evento.",
            gravedad="BAJA"
        ),
        Regla(
            nombre="R10_PREPARAR_INVENTARIO_FERIADO",
            antecedentes=["RIESGO_PICO_FERIADO"],
            consecuente="ACCION_REFORZAR_FERIADO",
            descripcion="Inferencia logística para aprovechar el incremento de tráfico del feriado y evitar quiebres ocultos.",
            accion_sugerida="Abastecer stock de seguridad adicional correspondiente al 40% del pico proyectado.",
            gravedad="BAJA"
        ),

        # --- REGLAS PARA PATRÓN HORARIO Y HORAS PICO ---
        Regla(
            nombre="R11_HORA_PICO_DETECTADA",
            antecedentes=["patron_horario_disponible", "horas_pico_sobre_percentil_85"],
            consecuente="RIESGO_HORA_PICO",
            descripcion="El modelo de red neuronal (MLP) predice concentraciones críticas de demanda en horas específicas.",
            accion_sugerida="Programar personal y reabastecer góndolas antes del inicio de la hora crítica.",
            gravedad="BAJA"
        ),
        Regla(
            nombre="R12_PROGRAMAR_REPISAS_GONDOLA",
            antecedentes=["RIESGO_HORA_PICO"],
            consecuente="ACCION_PREPARAR_GONDOLAS",
            descripcion="Acción operativa de cara al cliente para garantizar disponibilidad física del producto.",
            accion_sugerida="Tener el stock físico en góndola listo y ordenado 30 minutos antes del pico.",
            gravedad="BAJA"
        ),

        # --- REGLAS PARA EVALUACIÓN CAUSAL BAYESIANA ---
        Regla(
            nombre="R13_RIESGO_INCERTIDUMBRE_CRITICO",
            antecedentes=["probabilidad_quiebre_bayesiano_critica"],
            consecuente="RIESGO_BAYESIANO_CRITICO",
            descripcion="La Red Bayesiana Noisy-OR infiere una probabilidad muy alta de quiebre (>= 70%) basada en múltiples causas activas.",
            accion_sugerida="Aplicar factor de ajuste preventivo del +50% al stock de seguridad.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R14_RIESGO_INCERTIDUMBRE_ALTO",
            antecedentes=["probabilidad_quiebre_bayesiano_alta"],
            consecuente="RIESGO_BAYESIANO_ALTO",
            descripcion="La Red Bayesiana Noisy-OR estima una probabilidad moderada/alta de quiebre (>= 45% y < 70%).",
            accion_sugerida="Aplicar factor de ajuste de +25% sobre el stock de seguridad base.",
            gravedad="MEDIA"
        ),
        Regla(
            nombre="R15_AJUSTE_BAYESIANO_MAXIMO",
            antecedentes=["RIESGO_BAYESIANO_CRITICO"],
            consecuente="ACCION_INCREMENTO_SEGURIDAD_MAX",
            descripcion="Inferencia preventiva bayesiana para protegerse de incertidumbre extrema.",
            accion_sugerida="Aplicar multiplicador logístico x1.50 al stock de seguridad del actuador.",
            gravedad="ALTA"
        ),
        Regla(
            nombre="R16_AJUSTE_BAYESIANO_MODERADO",
            antecedentes=["RIESGO_BAYESIANO_ALTO"],
            consecuente="ACCION_INCREMENTO_SEGURIDAD_MOD",
            descripcion="Inferencia preventiva bayesiana para protegerse de incertidumbre moderada.",
            accion_sugerida="Aplicar multiplicador logístico x1.25 al stock de seguridad del actuador.",
            gravedad="MEDIA"
        )
    ]
    return base_reglas
