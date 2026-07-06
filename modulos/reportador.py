# -*- coding: utf-8 -*-

# Módulo Reportador para Pymevision AI
# Genera gráficos e informes visuales consolidados con matplotlib.

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuración de estilos para los gráficos
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linestyle'] = '--'

# Paleta de colores
COLOR_PRIMARIO = "#1A5276"
COLOR_SECUNDARIO = "#2E86C1"
COLOR_ALERTA_ALTA = "#CB4335"
COLOR_ALERTA_MEDIA = "#F39C12"
COLOR_ALERTA_BAJA = "#F1C40F"
COLOR_EXITO = "#28B463"
COLOR_FONDO_GRID = "#F4F6F7"

def asegurar_directorio_reportes(directorio="reportes"):
    subcarpetas = [
        directorio,
        os.path.join(directorio, "01_algoritmos_busqueda_grafos"),
        os.path.join(directorio, "02_redes_bayesiana_causal"),
        os.path.join(directorio, "03_prediccion_prophet_mlp"),
        os.path.join(directorio, "04_alertas_y_eventos")
    ]
    for folder in subcarpetas:
        if not os.path.exists(folder):
            os.makedirs(folder)

def graficar_demanda_y_alertas(df_inventario, dict_predicciones, lista_alertas, directorio="reportes"):
    # Compara stock físico/total con la demanda proyectada para productos con alertas críticas.
    asegurar_directorio_reportes(directorio)
    productos_con_alerta = [a for a in lista_alertas if a["tipo"] in ["RIESGO_QUIEBRE", "RIESGO_VENCIMIENTO"]]
    
    if not productos_con_alerta:
        productos_a_graficar = list(dict_predicciones.keys())[:4]
    else:
        productos_a_graficar = list(set([a["producto_id"] for a in productos_con_alerta]))[:6]
        
    if not productos_a_graficar:
        return
        
    filas = int(np.ceil(len(productos_a_graficar)/2))
    fig, axes = plt.subplots(nrows=filas, ncols=2, figsize=(15, 4 * filas))
    
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, prod_id in enumerate(productos_a_graficar):
        ax = axes[i]
        fila_inv = df_inventario[df_inventario["producto_id"] == prod_id].iloc[0]
        nombre_prod = fila_inv["producto_nombre"]
        stock_fisico = fila_inv["stock_fisico"]
        stock_transito = fila_inv["stock_transito"]
        tiempo_reposicion = int(fila_inv["tiempo_reposicion_dias"])
        
        pred = dict_predicciones[prod_id]
        fechas = pred["fecha"]
        demanda = pred["demanda_predicha"]
        
        ax.bar(fechas, demanda, color=COLOR_SECUNDARIO, alpha=0.6, label="Demanda Diaria Predicha")
        ax.axhline(y=stock_fisico, color=COLOR_PRIMARIO, linestyle="-", linewidth=2, label=f"Stock Físico ({stock_fisico} und.)")
        
        if stock_transito > 0:
            ax.axhline(y=stock_fisico + stock_transito, color=COLOR_EXITO, linestyle="--", linewidth=1.5, 
                       label=f"Stock Total c/Tránsito ({stock_fisico + stock_transito} und.)")
            
        fecha_fin_reposicion = fechas.iloc[min(tiempo_reposicion - 1, len(fechas) - 1)]
        ax.axvspan(fechas.iloc[0], fecha_fin_reposicion, color="#EAECEE", alpha=0.5, 
                   label=f"Tiempo Reposición ({tiempo_reposicion} días)")
        
        alertas_prod = [a for a in lista_alertas if a.get("producto_id") == prod_id]
        color_titulo = "black"
        titulo_sufijo = ""
        for a in alertas_prod:
            if a["tipo"] == "RIESGO_QUIEBRE":
                color_titulo = COLOR_ALERTA_ALTA
                titulo_sufijo = " - ¡Riesgo Quiebre!"
            elif a["tipo"] == "RIESGO_VENCIMIENTO":
                color_titulo = COLOR_ALERTA_MEDIA
                titulo_sufijo = " - ¡Riesgo Vencimiento!"
                
        ax.set_title(f"{nombre_prod}{titulo_sufijo}", fontsize=12, fontweight="bold", color=color_titulo)
        ax.set_ylabel("Cantidad de Productos")
        ax.set_xlabel("Fecha")
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        
    for j in range(len(productos_a_graficar), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "03_prediccion_prophet_mlp", "proyeccion_demanda_vs_stock.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_patrones_horas_pico(df_inventario, dict_patrones, directorio="reportes"):
    # Muestra los patrones de demanda horaria para la asignación de recursos y personal.
    asegurar_directorio_reportes(directorio)
    categorias_clave = ["Bebidas", "Snacks", "Lácteos", "Abarrotes"]
    productos_seleccionados = []
    
    for cat in categorias_clave:
        prods_cat = df_inventario[df_inventario["categoria"] == cat]
        if not prods_cat.empty:
            productos_seleccionados.append((prods_cat.iloc[0]["producto_id"], prods_cat.iloc[0]["producto_nombre"]))
            
    if not productos_seleccionados:
        return
        
    plt.figure(figsize=(12, 6))
    for prod_id, nombre_prod in productos_seleccionados:
        patron = dict_patrones.get(prod_id)
        if patron is not None:
            plt.plot(range(24), patron, marker="o", linewidth=2, label=nombre_prod)
            
    plt.axvspan(12, 14, color="#F5B7B1", alpha=0.3, label="Pico Almuerzo (12-14h)")
    plt.axvspan(18, 21, color="#D4E6F1", alpha=0.3, label="Pico Noche (18-21h)")
    
    plt.title("Patrón de Demanda Proyectada por Hora (Red Neuronal MLP)", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO)
    plt.xlabel("Hora del Día (Formato 24h)", fontsize=11)
    plt.ylabel("Demanda Estimada (Unidades/Hora)", fontsize=11)
    plt.xticks(range(24))
    plt.grid(True)
    plt.legend(fontsize=10, loc="upper left")
    plt.savefig(os.path.join(directorio, "03_prediccion_prophet_mlp", "patron_horas_pico.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_productos_mas_demandados(df_ventas, directorio="reportes"):
    # Identifica el producto más vendido de cada categoría.
    asegurar_directorio_reportes(directorio)
    ventas_totales = df_ventas.groupby(["categoria", "producto_nombre"])["cantidad_vendida"].sum().reset_index()
    indices_mas_vendidos = ventas_totales.groupby("categoria")["cantidad_vendida"].idxmax()
    mas_vendidos_por_cat = ventas_totales.loc[indices_mas_vendidos].sort_values(by="cantidad_vendida", ascending=True)
    
    plt.figure(figsize=(10, 6))
    colores = plt.cm.viridis(np.linspace(0.3, 0.8, len(mas_vendidos_por_cat)))
    barras = plt.barh(mas_vendidos_por_cat["categoria"], mas_vendidos_por_cat["cantidad_vendida"], color=colores, height=0.6)
    
    for barra, nombre_prod, cant in zip(barras, mas_vendidos_por_cat["producto_nombre"], mas_vendidos_por_cat["cantidad_vendida"]):
        plt.text(
            barra.get_width() - (barra.get_width() * 0.05) if barra.get_width() > 10 else barra.get_width() + 1,
            barra.get_y() + barra.get_height()/2,
            f"{nombre_prod} ({int(cant)} und.)",
            va="center",
            ha="right" if barra.get_width() > 10 else "left",
            color="white" if barra.get_width() > 10 else "black",
            fontweight="bold",
            fontsize=9
        )
        
    plt.title("Productos Más Demandados por Categoría (Ventas Históricas)", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO)
    plt.xlabel("Total de Unidades Vendidas", fontsize=11)
    plt.ylabel("Categoría", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(directorio, "03_prediccion_prophet_mlp", "productos_mas_demandados_categoria.png"), dpi=150, bbox_inches="tight")
    plt.close()

def generar_dashboard_alertas(lista_alertas, directorio="reportes"):
    # Genera un panel visual con el resumen de alertas y gravedad.
    asegurar_directorio_reportes(directorio)
    alertas_altas = [a for a in lista_alertas if a["gravedad"] == "ALTA"]
    alertas_medias = [a for a in lista_alertas if a["gravedad"] == "MEDIA"]
    alertas_bajas = [a for a in lista_alertas if a["gravedad"] == "BAJA"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    
    plt.text(0.5, 0.95, "PANEL CONTROL DE ALERTAS DE STOCK - PYMEVISION AI", 
             fontsize=16, fontweight="bold", ha="center", color=COLOR_PRIMARIO)
    
    plt.text(0.2, 0.88, f"Alertas Críticas (Altas): {len(alertas_altas)}", 
             fontsize=12, fontweight="bold", color=COLOR_ALERTA_ALTA, bbox=dict(facecolor='#FDEDEC', edgecolor=COLOR_ALERTA_ALTA, boxstyle='round,pad=0.5'))
    plt.text(0.5, 0.88, f"Alertas Moderadas (Medias): {len(alertas_medias)}", 
             fontsize=12, fontweight="bold", color=COLOR_ALERTA_MEDIA, bbox=dict(facecolor='#FEF9E7', edgecolor=COLOR_ALERTA_MEDIA, boxstyle='round,pad=0.5'))
    plt.text(0.8, 0.88, f"Alertas Operativas (Bajas): {len(alertas_bajas)}", 
             fontsize=12, fontweight="bold", color=COLOR_ALERTA_BAJA, bbox=dict(facecolor='#FCF3CF', edgecolor=COLOR_ALERTA_BAJA, boxstyle='round,pad=0.5'))
    
    y_pos = 0.78
    plt.text(0.05, y_pos, "DETALLE DE RIESGOS DETECTADOS Y ACCIONES SUGERIDAS:", fontsize=13, fontweight="bold", color="#333")
    
    todas_alertas = alertas_altas + alertas_medias + alertas_bajas
    alertas_visibles = todas_alertas[:8]
    
    y_pos -= 0.05
    for a in alertas_visibles:
        color_borde = COLOR_ALERTA_ALTA if a["gravedad"] == "ALTA" else (COLOR_ALERTA_MEDIA if a["gravedad"] == "MEDIA" else COLOR_SECUNDARIO)
        texto_gravedad = "ALTA" if a["gravedad"] == "ALTA" else ("MEDIA" if a["gravedad"] == "MEDIA" else "BAJA")
        tipo_alerta = a["tipo"].replace("_", " ")
        
        plt.text(0.05, y_pos, f"[{texto_gravedad}] {tipo_alerta}", fontsize=10, fontweight="bold", color="white",
                 bbox=dict(facecolor=color_borde, edgecolor=color_borde, boxstyle='round,pad=0.3'))
        
        descripcion_limpia = a["descripcion"][:102] + "..." if len(a["descripcion"]) > 105 else a["descripcion"]
        accion_limpia = a["accion_sugerida"][:102] + "..." if len(a["accion_sugerida"]) > 105 else a["accion_sugerida"]
        
        plt.text(0.28, y_pos + 0.01, f"Riesgo: {descripcion_limpia}", fontsize=9, color="#2C3E50")
        plt.text(0.28, y_pos - 0.018, f"Acción: {accion_limpia}", fontsize=9, fontweight="bold", color=COLOR_PRIMARIO)
        plt.plot([0.05, 0.95], [y_pos - 0.035, y_pos - 0.035], color="#E5E7E9", linewidth=0.8)
        
        y_pos -= 0.075
        
    if len(todas_alertas) > 8:
        plt.text(0.5, y_pos, f"... y {len(todas_alertas) - 8} alertas más en el sistema ...", 
                 fontsize=10, fontstyle="italic", ha="center", color="#7F8C8D")
        
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "dashboard_alertas.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_analisis_eventos(df_ventas, directorio="reportes"):
    # Analiza el impacto de los feriados y eventos especiales frente a días normales.
    asegurar_directorio_reportes(directorio)
    df_ventas_diarias = df_ventas.groupby(["fecha", "evento_especial"])["cantidad_vendida"].sum().reset_index()
    
    promedios_evento = df_ventas_diarias.groupby("evento_especial")["cantidad_vendida"].mean().reset_index()
    mapeo_limpieza = {
        "Da de la Madre": "Día de la Madre", "Da del Padre": "Día del Padre",
        "Día de la Madre": "Día de la Madre", "Día del Padre": "Día del Padre"
    }
    promedios_evento["evento_especial"] = promedios_evento["evento_especial"].replace(mapeo_limpieza)
    promedios_evento = promedios_evento.groupby("evento_especial")["cantidad_vendida"].mean().reset_index()
    promedios_evento = promedios_evento.sort_values(by="cantidad_vendida", ascending=True)
    
    promedio_normal = promedios_evento[promedios_evento["evento_especial"] == "ninguno"]["cantidad_vendida"].values[0]
    
    plt.figure(figsize=(10, 6))
    colores = []
    for evt in promedios_evento["evento_especial"]:
        if evt == "ninguno":
            colores.append("#7F8C8D")
        elif evt in ["Día de la Madre", "Fiestas Patrias"]:
            colores.append(COLOR_ALERTA_ALTA)
        else:
            colores.append(COLOR_SECUNDARIO)
            
    barras = plt.bar(promedios_evento["evento_especial"], promedios_evento["cantidad_vendida"], color=colores, width=0.6)
    plt.axhline(y=promedio_normal, color="#7F8C8D", linestyle="--", linewidth=1.5, label=f"Día Normal ({promedio_normal:.1f} und.)")
    
    for barra, evt, cant in zip(barras, promedios_evento["evento_especial"], promedios_evento["cantidad_vendida"]):
        texto = f"{cant:.1f} und.\n(+{int(((cant / promedio_normal) - 1) * 100)}%)" if evt != "ninguno" else f"{cant:.1f} und."
        plt.text(
            barra.get_x() + barra.get_width()/2,
            barra.get_height() + 10,
            texto,
            ha="center", va="bottom", fontweight="bold", fontsize=9
        )
        
    plt.title("Impacto de Eventos Especiales en las Ventas Diarias (Bodega)", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO)
    plt.ylabel("Promedio de Unidades Vendidas por Día", fontsize=11)
    plt.xlabel("Evento Especial", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, max(promedios_evento["cantidad_vendida"]) * 1.25)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "analisis_eventos_especiales.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_dispersion_ventas(df_ventas, directorio="reportes"):
    # Gráficos de dispersión para clima (temperatura) y elasticidad precio.
    asegurar_directorio_reportes(directorio)
    df_ventas_reales = df_ventas[df_ventas["cantidad_vendida"].notnull()].copy()
    if df_ventas_reales.empty:
        return
        
    df_diario = df_ventas_reales.groupby(["fecha", "producto_nombre", "categoria"]).agg({
        "cantidad_vendida": "sum", "temperatura": "mean", "precio_aplicado": "mean"
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Temperatura vs Cantidad Vendida
    colores_cat = {"Bebidas": "#2E86C1", "Snacks": "#E67E22", "Lácteos": "#27AE60", "Abarrotes": "#8E44AD"}
    for cat in df_diario["categoria"].unique():
        df_cat = df_diario[df_diario["categoria"] == cat]
        color = colores_cat.get(cat, "#7F8C8D")
        ax1.scatter(df_cat["temperatura"], df_cat["cantidad_vendida"], alpha=0.7, edgecolors="none", s=50, label=cat, color=color)
        if len(df_cat) > 1:
            try:
                z = np.polyfit(df_cat["temperatura"], df_cat["cantidad_vendida"], 1)
                p = np.poly1d(z)
                xp = np.linspace(df_cat["temperatura"].min(), df_cat["temperatura"].max(), 100)
                ax1.plot(xp, p(xp), color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            except Exception:
                pass
            
    ax1.set_title("Temperatura vs Cantidad Vendida Diaria", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax1.set_xlabel("Temperatura Promedio (°C)", fontsize=10)
    ax1.set_ylabel("Total de Unidades Vendidas por Día", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Categoría", fontsize=9)
    
    # 2. Precio vs Cantidad en transacciones
    df_promo = df_ventas_reales[
        (df_ventas_reales["es_promocion"] == True) | 
        (df_ventas_reales["es_promocion"].astype(str).str.upper() == "VERDADERO")
    ]
    df_plot_promo = df_promo if not df_promo.empty else df_ventas_reales[df_ventas_reales["producto_nombre"].isin(["Inca Kola 500ml", "Yogurt Gloria 1L", "Pan de Molde Bimbo 500g"])]
    if len(df_plot_promo) > 1000:
        df_plot_promo = df_plot_promo.sample(1000, random_state=42)
        
    scatter = ax2.scatter(df_plot_promo["precio_aplicado"], df_plot_promo["cantidad_vendida"], 
                          c=df_plot_promo["hora"], cmap="coolwarm", alpha=0.6, s=40, edgecolors="w", linewidths=0.2)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label("Hora del Día", fontsize=9)
    
    ax2.set_title("Precio Aplicado vs Cantidad por Transacción", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax2.set_xlabel("Precio Aplicado (S/.)", fontsize=10)
    ax2.set_ylabel("Cantidad Vendida en la Transacción", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Análisis de Dispersión de Ventas (Factores de Clima y Precio)", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "dispersion_ventas.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_dispersion_inventario(df_inventario, directorio="reportes"):
    # Gráficos de dispersión para inventario físico, pérdidas y en tránsito.
    asegurar_directorio_reportes(directorio)
    df_valido = df_inventario.dropna(subset=["stock_fisico", "ventas_perdidas_estimadas", "stock_transito"]).copy()
    if df_valido.empty:
        return
        
    df_valido["stock_fisico"] = df_valido["stock_fisico"].astype(float)
    df_valido["ventas_perdidas_estimadas"] = df_valido["ventas_perdidas_estimadas"].astype(float)
    df_valido["stock_transito"] = df_valido["stock_transito"].astype(float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colores_cat = {"Bebidas": "#2E86C1", "Snacks": "#E67E22", "Lácteos": "#27AE60", "Abarrotes": "#8E44AD"}
    
    for cat in df_valido["categoria"].unique():
        df_cat = df_valido[df_valido["categoria"] == cat]
        color = colores_cat.get(cat, "#7F8C8D")
        ax1.scatter(df_cat["stock_fisico"], df_cat["ventas_perdidas_estimadas"], alpha=0.6, edgecolors="none", s=45, label=cat, color=color)
        ax2.scatter(df_cat["stock_fisico"], df_cat["stock_transito"], alpha=0.6, edgecolors="none", s=45, label=cat, color=color)
        
    ax1.set_title("Stock Físico vs Ventas Perdidas Estimadas", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax1.set_xlabel("Stock Físico (Unidades)", fontsize=10)
    ax1.set_ylabel("Ventas Perdidas Estimadas (Unidades)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Categoría", fontsize=9)
    
    ax2.set_title("Stock Físico vs Stock en Tránsito (Órdenes de Compra)", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax2.set_xlabel("Stock Físico (Unidades)", fontsize=10)
    ax2.set_ylabel("Stock en Tránsito (Unidades)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(title="Categoría", fontsize=9)
    
    plt.suptitle("Análisis de Dispersión de Inventario (Políticas de Reposición y Quiebres)", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "dispersion_inventario.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_tabla_proyeccion_septiembre(inventario_actual, alertas, predicciones, directorio="reportes"):
    # Renderiza la tabla de proyecciones como imagen PNG.
    asegurar_directorio_reportes(directorio)
    rows = []
    for _, fila in inventario_actual.iterrows():
        prod_id = fila["producto_id"]
        nombre = fila["producto_nombre"]
        stock = int(fila["stock_fisico"])
        pred = predicciones[prod_id]
        demanda_7d = round(pred.head(7)["demanda_predicha"].sum(), 1)
        
        alertas_prod = [a for a in alertas if a.get("producto_id") == prod_id]
        estado = "NORMAL"
        for a in alertas_prod:
            if a["gravedad"] == "ALTA":
                estado = "¡QUIEBRE!"
                break
            elif a["gravedad"] == "MEDIA":
                estado = "PREVENCIÓN"
        rows.append([nombre, stock, demanda_7d, estado])
        
    df_table = pd.DataFrame(rows, columns=["Producto", "Stock Físico", "Demanda Est. 7D", "Estado Alerta"])
    
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df_table) + 1.2))
    ax.axis("off")
    
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLOR_PRIMARIO)
        else:
            cell.set_facecolor("#F2F4F4" if row % 2 == 0 else "white")
            if col == 0:
                cell.set_text_props(ha="left")
            if col == 3:
                val = cell.get_text().get_text()
                if val == "¡QUIEBRE!":
                    cell.set_text_props(weight="bold", color=COLOR_ALERTA_ALTA)
                    cell.set_facecolor("#FDEDEC")
                elif val == "PREVENCIÓN":
                    cell.set_text_props(weight="bold", color=COLOR_ALERTA_MEDIA)
                    cell.set_facecolor("#FEF9E7")
                else:
                    cell.set_text_props(weight="bold", color=COLOR_EXITO)
                    cell.set_facecolor("#E8F8F5")
                    
    plt.title("Proyección de Demanda - Primeros 7 Días de Septiembre", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "tabla_proyeccion_septiembre.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_tabla_impacto_eventos(datos_ventas, directorio="reportes"):
    # Renderiza la tabla de impacto de eventos especiales como imagen PNG.
    asegurar_directorio_reportes(directorio)
    datos_ventas_copia = datos_ventas.copy()
    if "fecha" not in datos_ventas_copia.columns:
        datos_ventas_copia["fecha"] = pd.to_datetime(datos_ventas_copia["fecha_hora"], format="%d/%m/%Y %H:%M").dt.date
    else:
        datos_ventas_copia["fecha"] = pd.to_datetime(datos_ventas_copia["fecha"]).dt.date
        
    df_ventas_diarias = datos_ventas_copia.groupby(["fecha", "evento_especial"])["cantidad_vendida"].sum().reset_index()
    promedios = df_ventas_diarias.groupby("evento_especial")["cantidad_vendida"].mean().to_dict()
    promedio_normal = promedios.get("ninguno", 179.58)
    
    mapeo_nombres = {
        "ninguno": "Día Normal (Sin eventos)", "Día de la Madre": "Campaña Día de la Madre",
        "Día del Padre": "Campaña Día del Padre", "Fiestas Patrias": "Campaña Fiestas Patrias",
        "Mundial FIFA 2026": "Periodo Mundial FIFA", "Santa Rosa de Lima": "Santa Rosa de Lima",
        "Batalla de Junín": "Batalla de Junín"
    }
    
    rows = []
    for evt, prom in sorted(promedios.items(), key=lambda x: x[1], reverse=True):
        nombre_limpio = mapeo_nombres.get(evt, evt)
        inc_str = f"+{int(((prom / promedio_normal) - 1) * 100)}%" if evt != "ninguno" else "Línea Base"
        rows.append([nombre_limpio, f"{prom:.1f} und.", inc_str])
        
    df_table = pd.DataFrame(rows, columns=["Evento Especial", "Promedio Ventas Diarias", "Incremento vs normal"])
    
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df_table) + 1.2))
    ax.axis("off")
    
    tbl = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLOR_PRIMARIO)
        else:
            cell.set_facecolor("#F2F4F4" if row % 2 == 0 else "white")
            if col == 0:
                cell.set_text_props(ha="left")
            inc_val = df_table.iloc[row-1, 2]
            if "+" in inc_val:
                pct = int(inc_val.replace("+", "").replace("%", ""))
                if pct > 100:
                    cell.set_text_props(weight="bold", color=COLOR_ALERTA_ALTA)
                elif pct > 10:
                    cell.set_text_props(weight="bold", color=COLOR_SECUNDARIO)
                    
    plt.title("Impacto de Eventos Especiales en las Ventas Diarias", fontsize=12, fontweight="bold", color=COLOR_PRIMARIO, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "04_alertas_y_eventos", "tabla_impacto_eventos.png"), dpi=150, bbox_inches="tight")
    plt.close()

def graficar_red_bayesiana(red_bayesiana, df_inventario, dict_predicciones, directorio="reportes"):
    # Genera gráficos explicativos y de diagnóstico para el modelo de Red Bayesiana Mixta.
    if red_bayesiana is None or not red_bayesiana.entrenado:
        return
        
    asegurar_directorio_reportes(directorio)
    metricas = red_bayesiana.obtener_metricas()
    
    # Nombres legibles para los nodos causa (parents)
    nombres_causas_map_clean = {
        "demanda_alta": "Demanda Alta",
        "retraso_proveedor": "Retraso Proveedor",
        "clima_adverso": "Clima Adverso",
        "stock_bajo": "Stock Bajo",
        "es_feriado": "Es Feriado",
        "es_fin_semana": "Es Fin de Semana"
    }
    
    nodos_causa = metricas["nodos_causa"]
    p_inhib = metricas["prob_inhibicion"]
    p_fuga = metricas["prob_fuga"]
    p_activa = metricas.get("prob_activa", {k: 0.15 for k in nodos_causa})
    p_v_0 = p_fuga
    
    # Helper para barra visual de probabilidad (segura para fuentes estándar)
    def generar_barra(prob):
        cant_iguales = int(round(prob * 8))
        return "=" * cant_iguales + " " * (8 - cant_iguales)
        
    # ----------------------------------------------------
    # FIGURA 1: GRAFO DE LA RED BAYESIANA (DAG)
    # ----------------------------------------------------
    fig_g = plt.figure(figsize=(16, 12))
    ax_graph = fig_g.add_subplot(1, 1, 1)
    ax_graph.axis("off")
    ax_graph.set_facecolor("white")
    
    # Coordenadas de los nodos en el plano
    x_t, y_t = 0.0, 1.2
    x_f, y_f = 0.0, -1.2
    
    coords_padres = {
        "demanda_alta": (-6.5, 2.5),
        "retraso_proveedor": (-3.9, 4.0),
        "clima_adverso": (-1.3, 4.8),
        "stock_bajo": (1.3, 4.8),
        "es_feriado": (3.9, 4.0),
        "es_fin_semana": (6.5, 2.5)
    }
    
    # Dibujar flechas (arcos dirigidos) de los padres al target
    for causa in nodos_causa:
        x_p, y_p = coords_padres[causa]
        ax_graph.annotate("", xy=(x_t, y_t), xytext=(x_p, y_p),
                          arrowprops=dict(arrowstyle="-|>", color="#5D6D7E", lw=2.5, mutation_scale=18, shrinkA=45, shrinkB=55))
                          
    # Dibujar flecha de Fuga al target
    ax_graph.annotate("", xy=(x_t, y_t), xytext=(x_f, y_f),
                      arrowprops=dict(arrowstyle="-|>", color="#E67E22", lw=2.5, mutation_scale=18, shrinkA=45, shrinkB=55))
                      
    # Dibujar las cajas de los nodos padres (formato pyAgrum / Netica)
    for causa in nodos_causa:
        x_p, y_p = coords_padres[causa]
        p_act = p_activa.get(causa, 0.15)
        p_inact = 1.0 - p_act
        p_inh = p_inhib[causa]
        p_imp = 1.0 - p_inh
        
        txt_node = (
            f"      {nombres_causas_map_clean[causa]}\n"
            f"---------------------------------\n"
            f" Activa  | {100*p_act:>5.1f}% [{generar_barra(p_act)}]\n"
            f" Inact.  | {100*p_inact:>5.1f}% [{generar_barra(p_inact)}]\n"
            f"---------------------------------\n"
            f" Inhib. (p_v,j) = {p_inh:.2f}\n"
            f" Causal Impact  = {100*p_imp:.1f}%"
        )
        
        bbox_parent = dict(boxstyle="square,pad=0.5", fc="#EBF5FB", ec="#2E86C1", lw=2)
        ax_graph.text(x_p, y_p, txt_node, family="monospace", fontsize=8.5, fontweight="bold", 
                      color="#1B4F72", ha="center", va="center", bbox=bbox_parent)
                      
    # Dibujar la caja del nodo Fuga
    txt_fuga = (
        f"     Fuga (No Modelado)\n"
        f"---------------------------------\n"
        f" Activa  | 100.0% [{generar_barra(1.0)}]\n"
        f" Inact.  |   0.0% [{generar_barra(0.0)}]\n"
        f"---------------------------------\n"
        f" Leak Prob. (p_v,0) = {p_v_0:.2f}"
    )
    bbox_fuga = dict(boxstyle="square,pad=0.5", fc="#FEF9E7", ec="#F39C12", lw=2)
    ax_graph.text(x_f, y_f, txt_fuga, family="monospace", fontsize=8.5, fontweight="bold", 
                  color="#7E5109", ha="center", va="center", bbox=bbox_fuga)
                  
    # Dibujar la caja del nodo Target (Quiebre de Stock)
    p_q_hist = metricas.get("prob_quiebre_historica", 0.185)
    p_n_hist = 1.0 - p_q_hist
    txt_target = (
        f"        QUIEBRE DE STOCK\n"
        f"---------------------------------\n"
        f" Quiebre | {100*p_q_hist:>5.1f}% [{generar_barra(p_q_hist)}]\n"
        f" Normal  | {100*p_n_hist:>5.1f}% [{generar_barra(p_n_hist)}]\n"
        f"---------------------------------\n"
        f"  Target variable / Effect Node"
    )
    bbox_target = dict(boxstyle="square,pad=0.5", fc="#FDEDEC", ec="#CB4335", lw=3)
    ax_graph.text(x_t, y_t, txt_target, family="monospace", fontsize=9.5, fontweight="bold", 
                  color="#78281F", ha="center", va="center", bbox=bbox_target)
                  
    # Tabla condicional para el nodo Target (CPT resumida)
    p_sb = p_inhib.get("stock_bajo", 1.0)
    p_rp = p_inhib.get("retraso_proveedor", 1.0)
    
    p_no_quiebre_todos = p_v_0
    for c in nodos_causa:
        p_no_quiebre_todos *= p_inhib[c]
    p_quiebre_todos = 1.0 - p_no_quiebre_todos
    
    txt_target_table = (
        f"   QUIEBRE DE STOCK - TABLA CONDICIONAL (CPT Noisy-OR)\n"
        f"  =========================================================\n"
        f"   Causas Activas (Evidencia)  | P(Quiebre = 1)\n"
        f"  -----------------------------+---------------------------\n"
        f"   Ninguna (Solo Fuga)         | {100 * (1.0 - p_v_0):>18.1f}%\n"
        f"   Solo Stock Bajo             | {100 * (1.0 - p_v_0 * p_sb):>18.1f}%\n"
        f"   Solo Retraso Proveedor      | {100 * (1.0 - p_v_0 * p_rp):>18.1f}%\n"
        f"   Stock Bajo + Retraso Prov.  | {100 * (1.0 - p_v_0 * p_sb * p_rp):>18.1f}%\n"
        f"   Todas las Causas Activas    | {100 * p_quiebre_todos:>18.1f}%\n"
        f"  -----------------------------+---------------------------"
    )
    bbox_target_table = dict(boxstyle="square,pad=0.5", fc="#FDFEFE", ec="#E6B0AA", lw=1.5)
    ax_graph.text(-4.5, 0.0, txt_target_table, family="monospace", fontsize=8.5, va="center", ha="center", bbox=bbox_target_table)
    
    # Explicación del modelo Noisy-OR en el lado derecho
    txt_noisyor_expl = (
        f"        METODOLOGIA NOISY-OR\n"
        f"=====================================\n"
        f" Formula de Inferencia:\n"
        f" P(Quiebre=1|Activas) =\n"
        f"   1 - p_v,0 * Prod_{{j in Activas}}(p_v,j)\n\n"
        f" * Simplifica la complejidad CPT.\n"
        f" * Evita el sobreajuste (overfitting).\n"
        f" * Ideal para datos PYME limitados."
    )
    bbox_expl = dict(boxstyle="square,pad=0.5", fc="#E8F8F5", ec="#117A65", lw=1.5)
    ax_graph.text(4.5, 0.0, txt_noisyor_expl, family="monospace", fontsize=8.5, va="center", ha="center", bbox=bbox_expl)
    
    ax_graph.set_xlim(-9.0, 9.0)
    ax_graph.set_ylim(-2.5, 6.0)
    ax_graph.set_title("Grafo Aciclico Dirigido (DAG) - Red Bayesiana Mixta", fontsize=15, fontweight="bold", color=COLOR_PRIMARIO, pad=15)
    
    plt.tight_layout()
    fig_g.savefig(os.path.join(directorio, "02_redes_bayesiana_causal", "red_bayesiana_grafo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_g)
    
    # ----------------------------------------------------
    # FIGURA 2: METRICAS Y BIC COMPARADO
    # ----------------------------------------------------
    fig_m = plt.figure(figsize=(9, 7))
    ax_bic = fig_m.add_subplot(1, 1, 1)
    ax_bic.axis("off")
    ax_bic.set_facecolor(COLOR_FONDO_GRID)
    
    bbox_props = dict(boxstyle="round,pad=1.0", fc="#F8F9F9", ec="#D5F5E3" if metricas["bic_score"] != 0 else "#E5E7E9", lw=2)
    
    texto_metricas = (
        "METRICAS DE LA RED BAYESIANA MIXTA\n"
        "=========================================\n\n"
        f"- Criterio de Informacion Bayesiano (BIC): {metricas['bic_score']}\n"
        f"- Log-Verosimilitud (LL): {metricas['log_verosimilitud']}\n"
        f"- Muestras de Entrenamiento: {metricas['n_muestras']} dias-producto\n\n"
        "COMPARACION DE COMPLEJIDAD (Parametros):\n"
        f"- Red Bayesiana Mixta (Noisy-OR): {metricas['n_parametros_noisy_or']} parametros\n"
        f"- Tabla de Probabilidad Condicional (CPT) General: {metricas['n_parametros_cpt_exponencial']} parametros\n"
        f"- Reduccion de Parametros: {metricas['reduccion_parametros']}\n\n"
        "NOTA METODOLOGICA:\n"
        "El modelo Noisy-OR asume causas independientes para simplificar\n"
        "el aprendizaje y evitar el sobreajuste (overfitting), logrando alta\n"
        "eficiencia computacional en entornos PYME (Vomlel et al., 2023)."
    )
    
    ax_bic.text(0.05, 0.5, texto_metricas, fontsize=11, family="monospace", 
                va="center", ha="left", bbox=bbox_props)
    ax_bic.set_title("Optimizacion de Complejidad y BIC", fontsize=14, fontweight="bold", color=COLOR_PRIMARIO, pad=15)
    
    plt.tight_layout()
    fig_m.savefig(os.path.join(directorio, "02_redes_bayesiana_causal", "red_bayesiana_metricas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    
    # ----------------------------------------------------
    # FIGURA 3: MAPA DE CALOR DE RIESGO
    # ----------------------------------------------------
    fig_h = plt.figure(figsize=(12, 8))
    ax_heatmap = fig_h.add_subplot(1, 1, 1)
    
    # Heatmap de riesgo por producto en los primeros 7 días de Septiembre
    productos_ids = list(dict_predicciones.keys())
    dias_proyectados = 7
    
    matriz_probabilidades = []
    nombres_productos = []
    
    primer_prod_id = productos_ids[0]
    pred_primer = dict_predicciones[primer_prod_id]
    pred_sept = pred_primer[pred_primer["fecha"] >= pd.to_datetime("2026-09-01")].head(dias_proyectados)
    fechas_x = [f.strftime("%d/%m (%a)") for f in pred_sept["fecha"]]
    
    for prod_id in productos_ids:
        fila_inv = df_inventario[df_inventario["producto_id"] == prod_id].iloc[0]
        nombres_productos.append(fila_inv["producto_nombre"])
        
        pred_prod = dict_predicciones[prod_id]
        pred_prod_sept = pred_prod[pred_prod["fecha"] >= pd.to_datetime("2026-09-01")].head(dias_proyectados).reset_index(drop=True)
        
        demanda_promedio = pred_prod["demanda_predicha"].mean()
        stock_seguridad_base = max(2, int(pred_prod_sept.head(int(fila_inv["tiempo_reposicion_dias"]))["demanda_predicha"].sum() * 0.20))
        
        stock_proyectado = float(fila_inv["stock_fisico"])
        probs_prod = []
        
        for idx, fila_dia in pred_prod_sept.iterrows():
            dia_fecha = fila_dia["fecha"]
            demanda_dia = fila_dia["demanda_predicha"]
            
            es_fer = False
            es_fds = dia_fecha.weekday() >= 5
            
            res = red_bayesiana.predecir_riesgo_quiebre(
                stock_fisico=max(0.0, stock_proyectado),
                demanda_predicha=demanda_dia,
                demanda_promedio=demanda_promedio,
                precipitacion=0.0,
                es_feriado=es_fer,
                es_fin_semana=es_fds,
                tiempo_reposicion=int(fila_inv["tiempo_reposicion_dias"]),
                stock_seguridad=stock_seguridad_base
            )
            probs_prod.append(res["probabilidad_quiebre"])
            stock_proyectado -= demanda_dia
            
        matriz_probabilidades.append(probs_prod)
        
    matriz_probabilidades = np.array(matriz_probabilidades)
    
    im = ax_heatmap.imshow(matriz_probabilidades, cmap="YlOrRd", vmin=0, vmax=1.0, aspect="auto")
    
    ax_heatmap.set_yticks(np.arange(len(nombres_productos)))
    ax_heatmap.set_yticklabels(nombres_productos, fontsize=10, fontweight="bold")
    ax_heatmap.set_xticks(np.arange(len(fechas_x)))
    ax_heatmap.set_xticklabels(fechas_x, fontsize=9, rotation=15)
    
    for i in range(len(nombres_productos)):
        for j in range(len(fechas_x)):
            val = matriz_probabilidades[i, j]
            color_texto = "white" if val > 0.6 else "black"
            ax_heatmap.text(j, i, f"{val*100:.1f}%", ha="center", va="center", 
                            color=color_texto, fontweight="bold", fontsize=10)
            
    ax_heatmap.set_title("Mapa de Calor: Riesgo Proyectado de Quiebre de Stock (Septiembre 2026)", 
                         fontsize=13, fontweight="bold", color=COLOR_PRIMARIO, pad=15)
    
    cbar = fig_h.colorbar(im, ax=ax_heatmap, orientation="horizontal", pad=0.15, shrink=0.7)
    cbar.set_label("Probabilidad de Quiebre de Stock", fontsize=11)
    
    plt.tight_layout()
    fig_h.savefig(os.path.join(directorio, "02_redes_bayesiana_causal", "red_bayesiana_mapa_calor.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_h)


def graficar_arbol_categorias_bfs_dfs(resultado_busqueda, directorio="reportes"):
    """
    Genera la gráfica 'arbol_categorias_bfs_dfs.png' comparando las estrategias de
    recorrido por niveles (BFS) vs en profundidad (DFS) en la jerarquía de productos.
    """
    asegurar_directorio_reportes(directorio)
    orden_bfs = resultado_busqueda["bfs"]["orden_visita"]
    orden_dfs = resultado_busqueda["dfs"]["orden_visita"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Graficar BFS (Recorrido por Niveles)
    pasos_bfs = range(1, len(orden_bfs) + 1)
    ax1.barh(orden_bfs[::-1], pasos_bfs[::-1], color=COLOR_PRIMARIO, alpha=0.85)
    ax1.set_title("Búsqueda en Anchura (BFS): Recorrido por Niveles\n(Categorías → Subcategorías → Productos)",
                  fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax1.set_xlabel("Paso / Orden de Exploración (Cola FIFO)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Graficar DFS (Recorrido en Profundidad)
    pasos_dfs = range(1, len(orden_dfs) + 1)
    ax2.barh(orden_dfs[::-1], pasos_dfs[::-1], color=COLOR_SECUNDARIO, alpha=0.85)
    ax2.set_title("Búsqueda en Profundidad (DFS): Inspección por Ramas\n(Exploración profunda de cada categoría)",
                  fontsize=12, fontweight="bold", color=COLOR_PRIMARIO)
    ax2.set_xlabel("Paso / Orden de Exploración (Pila LIFO)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Algoritmos de Búsqueda No Informada en Árbol de Inventario (Sílabo)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(directorio, "01_algoritmos_busqueda_grafos", "arbol_categorias_bfs_dfs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_ruta_reabastecimiento_astar(resultado_astar, directorio="reportes"):
    """
    Genera la gráfica 'ruta_reabastecimiento_astar.png' mostrando el mapa 2D del almacén
    con la ruta óptima calculada por A* usando f(n) = g(n) + h(n) y Distancia Manhattan.
    """
    asegurar_directorio_reportes(directorio)
    ruta_data = resultado_astar["ruta_reabastecimiento_astar"]
    camino = ruta_data["camino_coordenadas"]
    secuencia = ruta_data["secuencia_nodos"]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#F9FBFD")

    # Grilla 10x10 del almacén
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.grid(True, linestyle=":", color="#CCCCCC", alpha=0.7)
    ax.invert_yaxis()  # Origen (0,0) arriba a la izquierda

    # Dibujar camino completo de A*
    if len(camino) > 1:
        xs = [p[1] for p in camino]
        ys = [p[0] for p in camino]
        ax.plot(xs, ys, color=COLOR_ALERTA_ALTA, linewidth=3.5, linestyle="-", marker="o", markersize=6, label="Ruta Óptima A*")

    h_init = ruta_data.get("heuristica_h_inicial", 18)

    # Dibujar puntos de estantes y secuencia
    for paso in secuencia:
        pos_f = paso["pos_hacia"]
        nombre = paso["hacia"]
        f_val = paso["evaluacion_f"]
        g_val = paso["costo_g"]
        h_val = paso["heuristica_h"]

        ax.scatter(pos_f[1], pos_f[0], color=COLOR_PRIMARIO, s=180, zorder=5)
        ax.text(pos_f[1] + 0.15, pos_f[0] - 0.15, f"{nombre}\nf={f_val:.0f} (g={g_val:.0f}+h={h_val:.0f})",
                fontsize=8, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#AAAAAA", alpha=0.85))

    # Marcar origen (0,0)
    ax.scatter(0, 0, color=COLOR_EXITO, s=250, zorder=6, label="Recepción / Inicio (0,0)")
    ax.text(0.15, -0.25, f"INICIO (0,0)\nf={h_init:.0f} (g=0+h={h_init:.0f})", fontsize=8, fontweight="bold", color=COLOR_EXITO)

    ax.set_title(f"Optimización de Ruta Logística de Picking con A* (Secuencial Multi-Objetivo - Nearest Neighbor)\n"
                 f"Función f(n) = g(n) + h(n) | Distancia Manhattan a próximo objetivo (Paarth Sonkiya 2024)\n"
                 f"Costo Total Recorrido g(n) = {ruta_data['costo_g_total']:.0f} | Heurística Inicial h(0,0) = {h_init:.0f}",
                 fontsize=11, fontweight="bold", color=COLOR_PRIMARIO, pad=15)

    ax.set_xlabel("Pasillos / Coordenada Y (Almacén)", fontsize=10)
    ax.set_ylabel("Estantes / Coordenada X (Almacén)", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(directorio, "01_algoritmos_busqueda_grafos", "ruta_reabastecimiento_astar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_resultados_poda_alfa_beta(resultados_poda, directorio="reportes"):
    """
    Genera los gráficos de resultados para el Algoritmo Poda Alfa-Beta:
    1. 'poda_alfa_beta_eficiencia.png': Muestra nodos visitados vs podados y % de eficiencia.
    2. 'poda_alfa_beta_matriz_utilidad.png': Visualiza la matriz de utilidades por acción y escenario.
    """
    asegurar_directorio_reportes(directorio)
    subcarpeta_poda = os.path.join(directorio, "05_poda_alfa_beta")
    if not os.path.exists(subcarpeta_poda):
        os.makedirs(subcarpeta_poda, exist_ok=True)

    if not resultados_poda:
        return

    # 1. Gráfico de Eficiencia y Nodos
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig1.patch.set_facecolor("#FFFFFF")

    nombres = [item["producto_nombre"] for item in resultados_poda]
    visitados = [item["nodos_visitados"] for item in resultados_poda]
    podados = [item["nodos_podados"] for item in resultados_poda]
    eficiencias = [item["eficiencia_poda_pct"] for item in resultados_poda]

    x = np.arange(len(nombres))
    width = 0.35

    rects1 = ax1.bar(x - width/2, visitados, width, label="Nodos Visitados", color=COLOR_PRIMARIO)
    rects2 = ax1.bar(x + width/2, podados, width, label="Nodos Podados", color=COLOR_ALERTA_ALTA)

    ax1.set_ylabel("Cantidad de Nodos Evaluados", fontsize=10, fontweight="bold")
    ax1.set_title("Nodos Visitados vs. Nodos Podados por Producto", fontsize=11, fontweight="bold", color=COLOR_PRIMARIO)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Gráfico de Eficiencia (%)
    ax2.bar(x, eficiencias, color=COLOR_EXITO, alpha=0.85)
    ax2.set_ylabel("Porcentaje de Eficiencia de Poda (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Porcentaje de Reducción del Árbol de Búsqueda (Eficiencia α-β)", fontsize=11, fontweight="bold", color=COLOR_PRIMARIO)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=45, ha="right", fontsize=9)
    min_ef = max(0, int(min(eficiencias)) - 5)
    max_ef = min(100, int(max(eficiencias)) + 8)
    ax2.set_ylim(min_ef, max_ef)
    ax2.grid(True, linestyle="--", alpha=0.4)

    for i, v in enumerate(eficiencias):
        ax2.text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold", fontsize=9)

    plt.suptitle("Rendimiento del Algoritmo Poda Alfa-Beta en Decisiones de Inventario", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1.savefig(os.path.join(subcarpeta_poda, "poda_alfa_beta_eficiencia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2. Matriz de Utilidades para el primer producto representativo
    primer_res = resultados_poda[0]
    df_matriz = primer_res["matriz_evaluacion"]
    prod_nombre = primer_res["producto_nombre"]

    fig2, ax_mat = plt.subplots(figsize=(11, 6.2))
    fig2.patch.set_facecolor("#FFFFFF")
    ax_mat.axis("off")

    col_labels = ["Acción Agente (MAX)"] + [col for col in df_matriz.columns if col != "accion"]
    cell_text = []

    for _, row in df_matriz.iterrows():
        fila_txt = [row["accion"]]
        for col in df_matriz.columns:
            if col != "accion":
                val = row[col]
                fila_txt.append(f"S/. {val:,.2f}")
        cell_text.append(fila_txt)

    tabla = ax_mat.table(cellText=cell_text, colLabels=col_labels, loc="upper center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.scale(1.2, 1.7)

    # Dar formato a la tabla
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLOR_PRIMARIO)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if col == 0:
                cell.set_facecolor("#F2F4F7")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#FFFFFF")

    prof = primer_res.get("profundidad_evaluada", 3)
    ax_mat.set_title(f"Matriz de Utilidades Inmediatas (Día 1): {prod_nombre}\n"
                      f"Acción Óptima Seleccionada tras árbol a D={prof} días: {primer_res['mejor_accion']} "
                      f"(Utilidad Óptima Acumulada: S/. {primer_res['utilidad_optima']:,.2f})",
                      fontsize=11, fontweight="bold", color=COLOR_PRIMARIO, pad=20)

    # Agregar nota aclaratoria teórica para la sustentación
    nota_txt = (
        f"[*] NOTA EXPLICATIVA PARA SUSTENTACIÓN:\n"
        f"• Esta tabla muestra la utilidad inmediata de 1 paso (Día 1, U₁). La compra en el Día 1 incurre en un costo inicial de inventario.\n"
        f"• La decisión óptima '{primer_res['mejor_accion']}' surge de evaluar el árbol multietapa completo a {prof} días (D={prof}).\n"
        f"• Reabastecer en el Día 1 absorbe el costo inicial pero previene quiebres de stock severos y penalizaciones en los Días 2 y 3."
    )
    fig2.text(0.5, 0.04, nota_txt, ha="center", fontsize=8.5, color="#1E293B",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.95))

    plt.tight_layout()
    fig2.savefig(os.path.join(subcarpeta_poda, "poda_alfa_beta_matriz_utilidad.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3. Gráfico Consolidado (Combina ambos gráficos en una vista unificada)
    fig_cons = plt.figure(figsize=(15, 11))
    fig_cons.patch.set_facecolor("#FFFFFF")
    gs = fig_cons.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax_c1 = fig_cons.add_subplot(gs[0, 0])
    ax_c2 = fig_cons.add_subplot(gs[0, 1])
    ax_c3 = fig_cons.add_subplot(gs[1, :])

    # Subplot 1: Nodos Visitados vs Podados
    ax_c1.bar(x - width/2, visitados, width, label="Nodos Visitados", color=COLOR_PRIMARIO)
    ax_c1.bar(x + width/2, podados, width, label="Nodos Podados", color=COLOR_ALERTA_ALTA)
    ax_c1.set_ylabel("Cantidad de Nodos Evaluados", fontsize=9, fontweight="bold")
    ax_c1.set_title("Nodos Visitados vs. Nodos Podados por Producto", fontsize=10, fontweight="bold", color=COLOR_PRIMARIO)
    ax_c1.set_xticks(x)
    ax_c1.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8)
    ax_c1.legend(loc="upper right", fontsize=8)
    ax_c1.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2: % Eficiencia
    ax_c2.bar(x, eficiencias, color=COLOR_EXITO, alpha=0.85)
    ax_c2.set_ylabel("Porcentaje de Eficiencia de Poda (%)", fontsize=9, fontweight="bold")
    ax_c2.set_title("Porcentaje de Reducción del Árbol (Eficiencia α-β)", fontsize=10, fontweight="bold", color=COLOR_PRIMARIO)
    ax_c2.set_xticks(x)
    ax_c2.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8)
    ax_c2.set_ylim(min_ef, max_ef)
    ax_c2.grid(True, linestyle="--", alpha=0.4)
    for i, v in enumerate(eficiencias):
        ax_c2.text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold", fontsize=8)

    # Subplot 3: Matriz de Utilidades
    ax_c3.axis("off")
    tabla_c = ax_c3.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tabla_c.auto_set_font_size(False)
    tabla_c.set_fontsize(8.5)
    tabla_c.scale(1.1, 1.4)

    for (row, col), cell in tabla_c.get_celld().items():
        if row == 0:
            cell.set_facecolor(COLOR_PRIMARIO)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if col == 0:
                cell.set_facecolor("#F2F4F7")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#FFFFFF")

    ax_c3.set_title(f"Matriz de Utilidades Inmediatas (Día 1): {prod_nombre} - Decisión Óptima: {primer_res['mejor_accion']} (Utilidad Acumulada D={prof}: S/. {primer_res['utilidad_optima']:,.2f})",
                     fontsize=10, fontweight="bold", color=COLOR_PRIMARIO, pad=10)

    fig_cons.suptitle("Evaluación Consolidada del Algoritmo Poda Alfa-Beta en Decisiones de Inventario", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_cons.savefig(os.path.join(subcarpeta_poda, "poda_alfa_beta_consolidado.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_cons)


