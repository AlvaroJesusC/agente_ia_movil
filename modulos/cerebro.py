
#Módulo Cerebro para Pymevision AI
#Modelos predictivos del agente inteligente en Prophet y MLPRegressor.


import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from sklearn.neural_network import MLPRegressor

logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

class CerebroPredictivo:
    #Orquesta los modelos predictivos del agente inteligente.
    
    def __init__(self):
        self.modelos_horarios = {}
        
    def predecir_demanda_diaria(self, datos_historicos, producto_id, dias_a_predecir=7, df_festivos=None):
        #Predice la demanda diaria de un producto usando Prophet.
        
        datos_producto = datos_historicos[datos_historicos["producto_id"] == producto_id].copy()
        
        if len(datos_producto) < 5:
            promedio_ventas = datos_producto["cantidad_vendida"].mean() if len(datos_producto) > 0 else 0
            fechas_futuras = pd.date_range(
                start=datos_historicos["fecha"].max() + pd.Timedelta(days=1),
                periods=dias_a_predecir,
                freq="D"
            )
            return pd.DataFrame({
                "fecha": fechas_futuras,
                "demanda_predicha": np.maximum(0, promedio_ventas)
            })
            
        df_prophet = datos_producto[["fecha", "cantidad_vendida"]].rename(
            columns={"fecha": "ds", "cantidad_vendida": "y"}
        )
        
        modelo = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            growth='linear',
            holidays=df_festivos
        )
        
        modelo.fit(df_prophet)
        futuro = modelo.make_future_dataframe(periods=dias_a_predecir, freq='D')
        prediccion = modelo.predict(futuro)
        
        prediccion_retorno = prediccion[["ds", "yhat"]].copy()
        prediccion_retorno = prediccion_retorno.rename(columns={"ds": "fecha", "yhat": "demanda_predicha"})
        prediccion_retorno["demanda_predicha"] = np.maximum(0, prediccion_retorno["demanda_predicha"])
        
        return prediccion_retorno

    def entrenar_modelo_horario(self, datos_ventas_hora, producto_id):
        #eentrena una red Neuronal mlpRegressor para las ventas por hora.
        
        df_prod = datos_ventas_hora[datos_ventas_hora["producto_id"] == producto_id].copy()
        
        if len(df_prod) < 10:
            return None
            
        mapeo_dias = {
            "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
            "Viernes": 4, "Sábado": 5, "Domingo": 6
        }
        
        df_prod["dia_semana_num"] = df_prod["dia_semana"].map(mapeo_dias).fillna(0)
        df_prod["es_feriado_num"] = df_prod["es_feriado"].astype(int)
        df_prod["es_fin_semana_num"] = df_prod["es_fin_semana"].astype(int)
        
        X = df_prod[["hora", "dia_semana_num", "es_feriado_num", "es_fin_semana_num"]]
        y = df_prod["cantidad_vendida"]
        
        red_neuronal = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )
        
        red_neuronal.fit(X, y)
        self.modelos_horarios[producto_id] = red_neuronal
        return red_neuronal

    def predecir_patron_horario(self, producto_id, fecha, es_feriado):
        
        #Predice las ventas esperadas para cada una de las 24 horas del día.
        
        modelo = self.modelos_horarios.get(producto_id)
        
        dia_semana_nombre = fecha.strftime("%A")
        mapeo_ingles_espanol = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
            "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6
        }
        
        dia_semana_num = mapeo_ingles_espanol.get(dia_semana_nombre, fecha.weekday())
        es_fin_semana_num = 1 if dia_semana_num >= 5 else 0
        es_feriado_num = 1 if es_feriado else 0
        
        if modelo is None:
            patron_vacio = np.zeros(24)
            for h in range(24):
                if 12 <= h <= 14 or 18 <= h <= 20:
                    patron_vacio[h] = 3.0
                elif 8 <= h <= 22:
                    patron_vacio[h] = 1.0
            return patron_vacio
            
        X_pred = pd.DataFrame({
            "hora": list(range(24)),
            "dia_semana_num": [dia_semana_num] * 24,
            "es_feriado_num": [es_feriado_num] * 24,
            "es_fin_semana_num": [es_fin_semana_num] * 24
        })
        
        prediccion = modelo.predict(X_pred)
        return np.maximum(0, prediccion)


class RedBayesianaMixta:
    #Red Bayesiana Mixta (Mixed Noisy-OR).
    #Modela la probabilidad de quiebre de stock como nodo objetivo,
    #con causas independientes evaluadas mediante el modelo Noisy-OR.
    #Utiliza BIC para autoevaluación y penalización lineal C(P) = |pa(v)| + 1.
    
    def __init__(self):
        #Inicializa la estructura de la red bayesiana.
        #Nodos padre (causas del quiebre de stock):
        self.nodos_causa = [
            "demanda_alta",         # Picos de demanda inesperados
            "retraso_proveedor",    # Retrasos en la cadena de suministro
            "clima_adverso",        # Precipitaciones que afectan logística/demanda
            "es_feriado",           # Feriados nacionales con demanda atípica
            "es_fin_semana",        # Efecto fin de semana en ventas
            "stock_bajo"            # Stock por debajo del nivel de seguridad
        ]
        
        #Probabilidades de inhibición p_v,j (inicializadas con priors)
        #Cada p_v,j es la probabilidad de que la causa j sea "inhibida"
        #(es decir, que no produzca quiebre aunque esté activa)
        self.prob_inhibicion = {
            "demanda_alta": 0.40,
            "retraso_proveedor": 0.35,
            "clima_adverso": 0.75,
            "es_feriado": 0.55,
            "es_fin_semana": 0.70,
            "stock_bajo": 0.25
        }
        
        #Probabilidad de fuga p_v,0 (leak probability)
        #Captura eventos no modelados: mermas, robos, errores administrativos
        self.prob_fuga = 0.05
        
        #Métricas del modelo
        self.log_verosimilitud = None
        self.bic_score = None
        self.n_muestras = 0
        self.entrenado = False
        
        #Probabilidades de activación histórica de cada causa (para reportes)
        self.prob_activa = {k: 0.15 for k in self.nodos_causa}
        self.prob_quiebre_historica = 0.15
        
    def preparar_datos_entrenamiento(self, df_ventas, df_inventario):
        #Construye la matriz de observaciones binarias por producto-día
        #para entrenar los parámetros de la red.
        
        #Agregar ventas diarias por producto
        df_ventas_reales = df_ventas[df_ventas["cantidad_vendida"].notnull()].copy()
        ventas_diarias = df_ventas_reales.groupby(["fecha", "producto_id"]).agg({
            "cantidad_vendida": "sum",
            "temperatura": "mean",
            "precipitacion": "mean",
            "es_feriado": "max",
            "es_fin_semana": "max"
        }).reset_index()
        
        #Calcular promedios de demanda por producto (línea base)
        promedios_demanda = ventas_diarias.groupby("producto_id")["cantidad_vendida"].mean().to_dict()
        
        #Preparar datos de inventario
        df_inv_valido = df_inventario.dropna(subset=["stock_fisico", "ventas_perdidas_estimadas"]).copy()
        
        #Combinar ventas e inventario por fecha y producto
        if "fecha" in df_inv_valido.columns:
            df_combinado = ventas_diarias.merge(
                df_inv_valido[["fecha", "producto_id", "stock_fisico", "ventas_perdidas_estimadas", 
                               "tiempo_reposicion_dias"]],
                on=["fecha", "producto_id"],
                how="inner"
            )
        else:
            return pd.DataFrame()
            
        if df_combinado.empty:
            return pd.DataFrame()
            
        #Construir variables binarias (observaciones)
        observaciones = pd.DataFrame()
        
        #Variable objetivo: ¿hubo quiebre de stock?
        observaciones["quiebre_stock"] = (
            (df_combinado["ventas_perdidas_estimadas"] > 0) | 
            (df_combinado["stock_fisico"] <= 0)
        ).astype(int)
        
        #Causa 1: Demanda alta (demanda del día > 1.3x promedio del producto)
        observaciones["demanda_alta"] = df_combinado.apply(
            lambda fila: 1 if fila["cantidad_vendida"] > promedios_demanda.get(fila["producto_id"], 0) * 1.3 else 0,
            axis=1
        )
        
        #Causa 2: Retraso de proveedor (simulado: probabilidad proporcional al tiempo de reposición)
        np.random.seed(42)
        observaciones["retraso_proveedor"] = df_combinado.apply(
            lambda fila: 1 if np.random.random() < min(0.15 * fila["tiempo_reposicion_dias"] / 3.0, 0.30) else 0,
            axis=1
        )
        
        #Causa 3: Clima adverso (precipitación > 1.0 mm)
        observaciones["clima_adverso"] = (df_combinado["precipitacion"] > 1.0).astype(int)
        
        #Causa 4: Feriado
        observaciones["es_feriado"] = df_combinado["es_feriado"].astype(int)
        
        #Causa 5: Fin de semana
        observaciones["es_fin_semana"] = df_combinado["es_fin_semana"].astype(int)
        
        #Causa 6: Stock bajo (stock < 20% de la demanda promedio * tiempo reposición)
        observaciones["stock_bajo"] = df_combinado.apply(
            lambda fila: 1 if fila["stock_fisico"] < promedios_demanda.get(fila["producto_id"], 0) * 0.20 * fila["tiempo_reposicion_dias"] else 0,
            axis=1
        )
        
        return observaciones
    
    def aprender_parametros(self, observaciones):
        #Aprende las probabilidades de inhibición p_v,j desde los datos históricos.
        #Para cada causa j, calcula P(quiebre=0 | causa_j=1) como estimación de p_v,j.
        #Utiliza suavizado de Laplace para evitar probabilidades extremas (0 o 1).
        
        if observaciones.empty or len(observaciones) < 10:
            self.entrenado = True
            self.n_muestras = len(observaciones)
            self._calcular_metricas(observaciones)
            return
            
        self.n_muestras = len(observaciones)
        suavizado_laplace = 1  # Parámetro de suavizado (pseudoconteos)
        
        if "quiebre_stock" in observaciones.columns:
            self.prob_quiebre_historica = float(observaciones["quiebre_stock"].mean())
        
        for causa in self.nodos_causa:
            # Calcular la tasa histórica de activación de esta causa
            if causa in observaciones.columns:
                self.prob_activa[causa] = float(observaciones[causa].mean())
            else:
                self.prob_activa[causa] = 0.0
                
            #Filtrar observaciones donde la causa está activa
            mascara_causa_activa = observaciones[causa] == 1
            n_causa_activa = mascara_causa_activa.sum()
            
            if n_causa_activa > 0:
                #Contar cuántas veces NO hubo quiebre cuando la causa estaba activa
                n_no_quiebre_con_causa = ((observaciones.loc[mascara_causa_activa, "quiebre_stock"] == 0)).sum()
                
                #P(quiebre=0 | causa=1) con suavizado de Laplace
                p_inhibicion = (n_no_quiebre_con_causa + suavizado_laplace) / (n_causa_activa + 2 * suavizado_laplace)
                
                #Restringir a rango válido [0.05, 0.95]
                self.prob_inhibicion[causa] = np.clip(p_inhibicion, 0.05, 0.95)
        
        #Estimar la probabilidad de fuga p_v,0
        #P(quiebre=1 | ninguna causa activa)
        mascara_sin_causas = (observaciones[self.nodos_causa].sum(axis=1) == 0)
        n_sin_causas = mascara_sin_causas.sum()
        
        if n_sin_causas > 5:
            n_quiebre_sin_causa = observaciones.loc[mascara_sin_causas, "quiebre_stock"].sum()
            tasa_fuga = (n_quiebre_sin_causa + suavizado_laplace) / (n_sin_causas + 2 * suavizado_laplace)
            self.prob_fuga = np.clip(1.0 - tasa_fuga, 0.01, 0.20)
        
        self.entrenado = True
        self._calcular_metricas(observaciones)
    
    def calcular_noisy_or(self, evidencia):
        #Calcula la probabilidad de quiebre usando el modelo Noisy-OR.
        #Fórmula: P(x_v=0 | x_pa(v)) = p_v,0 · ∏ (p_v,j)^x_j
        #Retorna P(quiebre=1) = 1 - P(quiebre=0)
        
        #Calcular P(quiebre=0) = p_fuga * producto de inhibiciones activas
        p_no_quiebre = self.prob_fuga  # p_v,0 (probabilidad de fuga como base)
        
        for causa in self.nodos_causa:
            x_j = evidencia.get(causa, 0)  # 1 si la causa está activa, 0 si no
            if x_j == 1:
                #(p_v,j)^x_j = p_v,j cuando x_j=1, o 1 cuando x_j=0
                p_no_quiebre *= self.prob_inhibicion[causa]
        
        #P(quiebre=1) = 1 - P(quiebre=0)
        p_quiebre = 1.0 - p_no_quiebre
        return np.clip(p_quiebre, 0.0, 1.0)
    
    def calcular_bic(self, observaciones):
        #Calcula el Criterio de Información Bayesiano (BIC).
        #Fórmula: BIC(P|D) = LL(P|D) - (log|D| / 2) · C(P)
        #Donde:
        #  LL(P|D) = log-verosimilitud del modelo dados los datos
        #  |D| = número de muestras
        #  C(P) = penalización de complejidad (lineal: |pa(v)| + 1)
        
        if observaciones.empty:
            return 0.0
            
        n = len(observaciones)
        epsilon = 1e-10  # Para evitar log(0)
        
        #Calcular log-verosimilitud LL(P|D)
        ll = 0.0
        for _, fila in observaciones.iterrows():
            evidencia = {causa: int(fila[causa]) for causa in self.nodos_causa}
            p_quiebre = self.calcular_noisy_or(evidencia)
            quiebre_real = int(fila["quiebre_stock"])
            
            if quiebre_real == 1:
                ll += np.log(max(p_quiebre, epsilon))
            else:
                ll += np.log(max(1.0 - p_quiebre, epsilon))
        
        #Penalización de complejidad C(P) = |pa(v)| + 1
        c_p = self.calcular_penalizacion()
        
        #BIC = LL(P|D) - (log|D| / 2) · C(P)
        bic = ll - (np.log(n) / 2.0) * c_p
        
        return bic
    
    def calcular_penalizacion(self):
        #Calcula la penalización lineal del modelo Noisy-OR.
        #Fórmula: C_v*(P(X_v | X_pa(v))) = |pa(v)| + 1
        #Donde |pa(v)| es el número de nodos padre (causas).
        #Esto contrasta con la penalización exponencial de una CPT general
        #que sería 2^|pa(v)| - 1 (ej: 2^6 - 1 = 63 parámetros vs 6+1 = 7).
        
        return len(self.nodos_causa) + 1
    
    def _calcular_metricas(self, observaciones):
        #Calcula y almacena las métricas internas del modelo.
        
        if observaciones.empty or len(observaciones) < 5:
            self.log_verosimilitud = 0.0
            self.bic_score = 0.0
            return
            
        #Calcular log-verosimilitud
        epsilon = 1e-10
        ll = 0.0
        for _, fila in observaciones.iterrows():
            evidencia = {causa: int(fila[causa]) for causa in self.nodos_causa}
            p_quiebre = self.calcular_noisy_or(evidencia)
            quiebre_real = int(fila["quiebre_stock"])
            
            if quiebre_real == 1:
                ll += np.log(max(p_quiebre, epsilon))
            else:
                ll += np.log(max(1.0 - p_quiebre, epsilon))
        
        self.log_verosimilitud = ll
        self.bic_score = self.calcular_bic(observaciones)
    
    def predecir_riesgo_quiebre(self, stock_fisico, demanda_predicha, demanda_promedio,
                                  precipitacion, es_feriado, es_fin_semana, 
                                  tiempo_reposicion, stock_seguridad):
        #Predice la probabilidad de quiebre de stock para un producto
        #usando el modelo Noisy-OR con la evidencia actual.
        
        #Construir vector de evidencia binaria
        evidencia = {
            "demanda_alta": 1 if demanda_predicha > demanda_promedio * 1.3 else 0,
            "retraso_proveedor": 1 if tiempo_reposicion > 3 else 0,
            "clima_adverso": 1 if precipitacion > 1.0 else 0,
            "es_feriado": 1 if es_feriado else 0,
            "es_fin_semana": 1 if es_fin_semana else 0,
            "stock_bajo": 1 if stock_fisico < stock_seguridad else 0
        }
        
        probabilidad = self.calcular_noisy_or(evidencia)
        
        #Clasificar nivel de riesgo
        if probabilidad >= 0.70:
            nivel = "CRITICO"
        elif probabilidad >= 0.45:
            nivel = "ALTO"
        elif probabilidad >= 0.25:
            nivel = "MODERADO"
        else:
            nivel = "BAJO"
            
        return {
            "probabilidad_quiebre": round(probabilidad, 4),
            "nivel_riesgo": nivel,
            "evidencia": evidencia,
            "causas_activas": [c for c, v in evidencia.items() if v == 1],
            "n_causas_activas": sum(evidencia.values())
        }
    
    def obtener_metricas(self):
        #Retorna las métricas del modelo para reportes.
        
        n_params_noisy_or = self.calcular_penalizacion()
        n_params_cpt_general = (2 ** len(self.nodos_causa)) - 1  # Exponencial
        
        return {
            "bic_score": round(self.bic_score, 2) if self.bic_score is not None else 0.0,
            "log_verosimilitud": round(self.log_verosimilitud, 2) if self.log_verosimilitud is not None else 0.0,
            "n_muestras": self.n_muestras,
            "n_parametros_noisy_or": n_params_noisy_or,
            "n_parametros_cpt_exponencial": n_params_cpt_general,
            "reduccion_parametros": f"{n_params_cpt_general} → {n_params_noisy_or} ({round((1 - n_params_noisy_or/n_params_cpt_general)*100, 1)}% reducción)",
            "prob_fuga": round(self.prob_fuga, 4),
            "prob_inhibicion": {k: round(v, 4) for k, v in self.prob_inhibicion.items()},
            "prob_activa": {k: round(v, 4) for k, v in self.prob_activa.items()},
            "prob_quiebre_historica": round(self.prob_quiebre_historica, 4),
            "nodos_causa": self.nodos_causa,
            "entrenado": self.entrenado
        }
