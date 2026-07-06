# -*- coding: utf-8 -*-
"""
Servidor API HTTP FastAPI para Pymevision AI
-------------------------------------------
Expone los análisis y algoritmos del agente inteligente como servicios HTTP
que devuelven directamente imágenes PNG (media_type="image/png") para su consumo en la app Flutter.
"""

import sys
import os
import io
import pandas as pd
import numpy as np

# Configurar backend de Matplotlib sin GUI para entornos de servidor / headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Agregar la raíz del proyecto a sys.path para importar los módulos sin tocar su código
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from modulos import sensor, cerebro, actuador, busqueda, reportador
from modulos.poda_alfa_beta import EvaluadorPodaAlfaBeta, EstadoInventario, AccionAgente, EscenarioEntorno

# Globales para mantener datos y modelos cargados en memoria
STATE = {
    "df_ventas": None,
    "df_inventario": None,
    "ventas_diarias": None,
    "df_festivos": None,
    "agente_cerebro": None,
    "red_bayesiana": None,
    "predicciones": {},
    "patrones_horarios": {},
    "alertas": [],
    "actuador_opt": None
}

def inicializar_sistema():
    """Carga los datos y entrena los modelos una sola vez al arrancar la API."""
    print("[API] Cargando dataset de ventas e inventario...")
    df_ventas_full = sensor.cargar_ventas(os.path.join(ROOT_DIR, "datos", "ventas_datasL.csv"))
    df_inventario_full = sensor.cargar_inventario(os.path.join(ROOT_DIR, "datos", "inventario_datasL.csv"))

    # Filtrar rango de entrenamiento
    df_ventas = df_ventas_full[df_ventas_full["fecha"] <= pd.to_datetime("2026-05-31")].copy()
    df_inventario = df_inventario_full[df_inventario_full["fecha"] <= pd.to_datetime("2026-05-31")].copy()
    ventas_diarias = sensor.obtener_ventas_diarias_completas(df_ventas)

    # Base de feriados
    df_ventas_copia = df_ventas.copy()
    df_ventas_copia["fecha_norm"] = df_ventas_copia["fecha_hora"].dt.normalize()

    df_feriados = df_ventas_copia[df_ventas_copia["es_feriado"]][["fecha_norm", "tipo_feriado"]].drop_duplicates()
    df_feriados = df_feriados.rename(columns={"fecha_norm": "ds", "tipo_feriado": "holiday"})

    df_eventos = df_ventas_copia[df_ventas_copia["evento_especial"] != "ninguno"][["fecha_norm", "evento_especial"]].drop_duplicates()
    df_eventos = df_eventos.rename(columns={"fecha_norm": "ds", "evento_especial": "holiday"})

    df_festivos = pd.concat([df_feriados, df_eventos]).drop_duplicates().reset_index(drop=True)

    print("[API] Entrenando modelos predictivos (Prophet y MLP)...")
    agente_cerebro = cerebro.CerebroPredictivo()
    productos_ids = df_inventario["producto_id"].unique()
    predicciones = {}
    patrones_horarios = {}

    for prod_id in productos_ids:
        # Prophet
        pred = agente_cerebro.predecir_demanda_diaria(
            ventas_diarias, prod_id, dias_a_predecir=110, df_festivos=df_festivos
        )
        predicciones[prod_id] = pred

        # MLP Horario (usar df_ventas directamente)
        agente_cerebro.entrenar_modelo_horario(df_ventas, prod_id)
        patron = agente_cerebro.predecir_patron_horario(prod_id, pd.to_datetime("2026-06-01"), es_feriado=False)
        patrones_horarios[prod_id] = patron

    print("[API] Entrenando Red Bayesiana Mixta (Noisy-OR)...")
    red_bayesiana = cerebro.RedBayesianaMixta()
    observaciones_bn = red_bayesiana.preparar_datos_entrenamiento(df_ventas, df_inventario)
    red_bayesiana.aprender_parametros(observaciones_bn)

    # Evaluación inicial de alertas
    actuador_opt = actuador.ActuadorOptimizado()
    alertas = []
    for _, fila in df_inventario.iterrows():
        prod_id = fila["producto_id"]
        pred_prod = predicciones[prod_id]
        patron_prod = patrones_horarios[prod_id]
        al_prod = actuador_opt.evaluar_producto(fila, pred_prod, patron_prod, [], red_bayesiana=red_bayesiana)
        alertas.extend(al_prod)

    STATE["df_ventas"] = df_ventas_full
    STATE["df_inventario"] = df_inventario_full
    STATE["ventas_diarias"] = ventas_diarias
    STATE["df_festivos"] = df_festivos
    STATE["agente_cerebro"] = agente_cerebro
    STATE["red_bayesiana"] = red_bayesiana
    STATE["predicciones"] = predicciones
    STATE["patrones_horarios"] = patrones_horarios
    STATE["alertas"] = alertas
    STATE["actuador_opt"] = actuador_opt
    print("[API] Inicialización completada con éxito.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    inicializar_sistema()
    yield

app = FastAPI(
    title="Pymevision AI HTTP API",
    description="API HTTP que expone análisis y gráficos PNG de inteligencia artificial para app Flutter.",
    version="1.0.0",
    lifespan=lifespan
)

# Habilitar CORS para peticiones desde aplicaciones móviles / Flutter web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fig_to_png_response(fig):
    """Convierte una figura Matplotlib en una respuesta FastAPI HTTP PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# ----------------------------------------------------------------------
# ENDPOINTS HTTP API (Devuelven media_type="image/png")
# ----------------------------------------------------------------------

@app.get("/", summary="Estado de la API")
def read_root():
    return {"status": "online", "message": "Pymevision AI API operando. Usa las rutas HTTP para solicitar gráficos PNG."}


@app.get("/predecir", summary="Gráfico PNG de Proyección de Demanda vs Stock")
def endpoint_predecir(producto_id: str = Query(None, description="ID del producto (ej. prod_001)")):
    """Devuelve directamente la imagen PNG con la proyección de demanda y nivel de stock."""
    df_inv = STATE["df_inventario"]
    preds = STATE["predicciones"]
    alertas = STATE["alertas"]

    if df_inv is None:
        raise HTTPException(status_code=503, detail="Sistema inicializando datos...")

    # Si se pasa un producto específico
    if producto_id and producto_id in preds:
        fila_inv = df_inv[df_inv["producto_id"] == producto_id]
        if fila_inv.empty:
            raise HTTPException(status_code=404, detail="Producto no encontrado")
        fila_inv = fila_inv.iloc[0]
        nombre_prod = fila_inv["producto_nombre"]
        stock_fisico = fila_inv["stock_fisico"]
        stock_transito = fila_inv["stock_transito"]
        tiempo_reposicion = int(fila_inv["tiempo_reposicion_dias"])

        pred = preds[producto_id].head(30)
        fechas = pred["fecha"]
        demanda = pred["demanda_predicha"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(fechas, demanda, color=reportador.COLOR_SECUNDARIO, alpha=0.7, label="Demanda Diaria Predicha (Prophet)")
        ax.axhline(y=stock_fisico, color=reportador.COLOR_PRIMARIO, linestyle="-", linewidth=2, label=f"Stock Físico ({stock_fisico} und.)")

        if stock_transito > 0:
            ax.axhline(y=stock_fisico + stock_transito, color=reportador.COLOR_EXITO, linestyle="--", linewidth=1.5,
                       label=f"Stock Total c/Tránsito ({stock_fisico + stock_transito} und.)")

        fecha_fin_rep = fechas.iloc[min(tiempo_reposicion - 1, len(fechas) - 1)]
        ax.axvspan(fechas.iloc[0], fecha_fin_rep, color="#EAECEE", alpha=0.5, label=f"Tiempo Reposición ({tiempo_reposicion} días)")

        ax.set_title(f"Proyección de Demanda vs Stock: {nombre_prod} ({producto_id})", fontsize=13, fontweight="bold", color=reportador.COLOR_PRIMARIO)
        ax.set_ylabel("Unidades")
        ax.set_xlabel("Fecha")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        return fig_to_png_response(fig)

    else:
        # Generar gráfico consolidado con reportador
        reportador.graficar_demanda_y_alertas(df_inv, preds, alertas, directorio="reportes")
        file_path = os.path.join(ROOT_DIR, "reportes", "03_prediccion_prophet_mlp", "proyeccion_demanda_vs_stock.png")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return Response(content=f.read(), media_type="image/png")
        raise HTTPException(status_code=500, detail="Error al generar gráfico")


@app.get("/restock", summary="Gráfico PNG de Evaluación de Reabastecimiento e Inventario")
def endpoint_restock(producto_id: str = Query(None, description="ID del producto (ej. prod_001)")):
    """Devuelve directamente la imagen PNG con la dispersión de inventario y política de reposición."""
    df_inv = STATE["df_inventario"]
    if df_inv is None:
        raise HTTPException(status_code=503, detail="Sistema inicializando datos...")

    if producto_id:
        df_prod = df_inv[df_inv["producto_id"] == producto_id]
        if not df_prod.empty:
            fila = df_prod.iloc[0]
            nombre = fila["producto_nombre"]
            stock_fis = fila["stock_fisico"]
            stock_tra = fila["stock_transito"]
            min_ped = fila["cantidad_minima_pedido"]
            lead = fila["tiempo_reposicion_dias"]

            fig, ax = plt.subplots(figsize=(9, 5))
            categorias = ["Stock Físico Actual", "Stock en Tránsito", "Mínimo Pedido Proveedor"]
            valores = [stock_fis, stock_tra, min_ped]
            colores = [reportador.COLOR_PRIMARIO, reportador.COLOR_SECUNDARIO, reportador.COLOR_ALERTA_MEDIA]

            barras = ax.bar(categorias, valores, color=colores, width=0.5)
            for b in barras:
                yval = b.get_height()
                ax.text(b.get_x() + b.get_width()/2.0, yval + 0.5, f"{int(yval)} und", ha='center', va='bottom', fontweight='bold')

            ax.set_title(f"Evaluación de Reposición (Restock): {nombre}\nTiempo de Reposición Lead Time: {lead} días", 
                         fontsize=12, fontweight="bold", color=reportador.COLOR_PRIMARIO)
            ax.set_ylabel("Cantidad de Unidades")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            return fig_to_png_response(fig)

    reportador.graficar_dispersion_inventario(df_inv, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "04_alertas_y_eventos", "dispersion_inventario.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de restock")


@app.get("/anomalias", summary="Gráfico PNG del Panel de Control de Alertas y Anomalías")
def endpoint_anomalias():
    """Devuelve el gráfico PNG del Dashboard de Alertas Críticas y Anomalías del sistema."""
    alertas = STATE["alertas"]
    reportador.generar_dashboard_alertas(alertas, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "04_alertas_y_eventos", "dashboard_alertas.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar dashboard de alertas")


@app.get("/clasificar", summary="Gráfico PNG de Clasificación de Productos y Análisis Causal")
def endpoint_clasificar(producto_id: str = Query(None, description="ID de producto opcional")):
    """Devuelve el gráfico PNG con la clasificación de demanda histórica por categoría o el mapa bayesiano."""
    df_inv = STATE["df_inventario"]
    preds = STATE["predicciones"]
    red_bayesiana = STATE["red_bayesiana"]

    if producto_id and red_bayesiana:
        reportador.graficar_red_bayesiana(red_bayesiana, df_inv, preds, directorio="reportes")
        file_path = os.path.join(ROOT_DIR, "reportes", "02_redes_bayesiana_causal", "red_bayesiana_mapa_calor.png")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return Response(content=f.read(), media_type="image/png")

    df_ventas = STATE["df_ventas"]
    reportador.graficar_productos_mas_demandados(df_ventas, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "03_prediccion_prophet_mlp", "productos_mas_demandados_categoria.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de clasificación")


@app.get("/busqueda-categorias", summary="Gráfico PNG del Árbol de Búsqueda BFS vs DFS")
def endpoint_busqueda_categorias():
    """Ejecuta los algoritmos de búsqueda BFS y DFS en el árbol de inventario y devuelve el gráfico PNG."""
    df_inv = STATE["df_inventario"]
    actuador_opt = STATE["actuador_opt"]
    res_busqueda = actuador_opt.ejecutar_busqueda_categorias_bfs_dfs(df_inv)
    reportador.graficar_arbol_categorias_bfs_dfs(res_busqueda, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "01_algoritmos_busqueda_grafos", "arbol_categorias_bfs_dfs.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de árbol de categorías")


@app.get("/ruta-reabastecimiento", summary="Gráfico PNG del Algoritmo A* de Ruta de Reabastecimiento")
def endpoint_ruta_reabastecimiento(
    origen: str = Query("0,0", description="Coordenadas de origen (ej. 0,0)"),
    destino: str = Query("9,9", description="Coordenadas de destino opcionales")
):
    """Devuelve la imagen PNG de la ruta óptima de almacén calculada con A* y Distancia Manhattan."""
    df_inv = STATE["df_inventario"]
    alertas = STATE["alertas"]
    actuador_opt = STATE["actuador_opt"]
    res_astar = actuador_opt.optimizar_logistica_y_pedidos_astar(df_inv, alertas)
    reportador.graficar_ruta_reabastecimiento_astar(res_astar, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "01_algoritmos_busqueda_grafos", "ruta_reabastecimiento_astar.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar ruta A*")


@app.get("/poda-alfa-beta", summary="Gráfico PNG del Algoritmo Minimax con Poda Alfa-Beta")
def endpoint_poda_alfa_beta(
    producto_id: str = Query(None, description="ID del producto para la matriz de utilidad (ej. prod_001)"),
    tipo: str = Query("eficiencia", description="Tipo de gráfico a devolver: 'eficiencia', 'matriz', o 'consolidado'")
):
    """Devuelve la imagen PNG del algoritmo Poda Alfa-Beta (Eficiencia de todos los productos, Matriz de Utilidad o Vista Consolidada)."""
    df_inv = STATE["df_inventario"]
    preds = STATE["predicciones"]

    if df_inv is None:
        raise HTTPException(status_code=503, detail="Sistema inicializando datos...")

    # Evaluar TODOS los productos para la comparativa de eficiencia
    evaluador = EvaluadorPodaAlfaBeta()
    resultados_poda = []

    prods_eval = list(preds.keys())

    # Si se especificó un producto_id, colocarlo al inicio para que su matriz sea la seleccionada
    if producto_id and producto_id in prods_eval:
        prods_eval.remove(producto_id)
        prods_eval.insert(0, producto_id)

    for p_id in prods_eval:
        fila = df_inv[df_inv["producto_id"] == p_id].iloc[0]
        pred = preds[p_id]
        demanda_prom = pred["demanda_predicha"].mean()

        estado_init = EstadoInventario(
            stock_actual=fila["stock_fisico"],
            demanda_diaria_prom=demanda_prom,
            costo_unidad=fila["precio_unitario"] * 0.7,
            precio_venta=fila["precio_unitario"],
            dias_vencimiento=fila.get("vida_util_dias", 30),
            es_perecedero=fila["es_perecedero"],
            lead_time_base=fila["tiempo_reposicion_dias"]
        )

        res = evaluador.optimizar_decision(p_id, fila["producto_nombre"], estado_init, profundidad=3)
        resultados_poda.append(res)

    reportador.graficar_resultados_poda_alfa_beta(resultados_poda, directorio="reportes")

    nombre_archivo = "poda_alfa_beta_eficiencia.png"
    if tipo == "matriz":
        nombre_archivo = "poda_alfa_beta_matriz_utilidad.png"
    elif tipo == "consolidado":
        nombre_archivo = "poda_alfa_beta_consolidado.png"

    file_path = os.path.join(ROOT_DIR, "reportes", "05_poda_alfa_beta", nombre_archivo)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de Poda Alfa-Beta")


@app.get("/horas-pico", summary="Gráfico PNG de Patrón de Demanda por Horas Pico")
def endpoint_horas_pico():
    """Devuelve la imagen PNG con el patrón de demanda por hora (MLP)."""
    df_inv = STATE["df_inventario"]
    patrones = STATE["patrones_horarios"]
    reportador.graficar_patrones_horas_pico(df_inv, patrones, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "03_prediccion_prophet_mlp", "patron_horas_pico.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de horas pico")


@app.get("/eventos-especiales", summary="Gráfico PNG del Impacto de Eventos Especiales")
def endpoint_eventos_especiales():
    """Devuelve la imagen PNG con el análisis de impacto de feriados y eventos especiales en ventas."""
    df_ventas = STATE["df_ventas"]
    reportador.graficar_analisis_eventos(df_ventas, directorio="reportes")
    file_path = os.path.join(ROOT_DIR, "reportes", "04_alertas_y_eventos", "analisis_eventos_especiales.png")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/png")
    raise HTTPException(status_code=500, detail="Error al generar gráfico de eventos especiales")



