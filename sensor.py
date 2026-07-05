# -*- coding: utf-8 -*-

# Módulo Sensor para Pymevision AI
# Carga, limpieza y preprocesamiento de datos de ventas e inventario.

import pandas as pd
import numpy as np

def cargar_ventas(ruta_archivo="csv/ventas_datasL.csv"):
    # Carga el archivo de ventas y formatea las columnas de fecha y booleanos.
    datos_ventas = pd.read_csv(ruta_archivo, encoding="utf-8")
    
    datos_ventas["fecha_hora"] = pd.to_datetime(datos_ventas["fecha_hora"], format="%d/%m/%Y %H:%M")
    datos_ventas["fecha"] = datos_ventas["fecha_hora"].dt.normalize()
    
    columnas_booleanas = ["es_hora_pico", "es_fin_semana", "es_feriado"]
    for columna in columnas_booleanas:
        if columna in datos_ventas.columns:
            datos_ventas[columna] = datos_ventas[columna].astype(str).str.upper() == "VERDADERO"
            
    if "es_promocion" in datos_ventas.columns:
        datos_ventas["es_promocion"] = datos_ventas["es_promocion"].astype(object)
        mascara_no_nula = datos_ventas["es_promocion"].notnull()
        datos_ventas.loc[mascara_no_nula, "es_promocion"] = datos_ventas.loc[mascara_no_nula, "es_promocion"].astype(str).str.upper() == "VERDADERO"
            
    return datos_ventas

def cargar_inventario(ruta_archivo="csv/inventario_datasL.csv"):
    # carga el archivo de inventario y formatea fechas y booleanos.
    datos_inventario = pd.read_csv(ruta_archivo, encoding="utf-8")
    datos_inventario["fecha"] = pd.to_datetime(datos_inventario["fecha"], format="%d/%m/%Y")
    
    columnas_booleanas = ["es_perecedero", "es_feriado", "es_fin_semana"]
    for columna in columnas_booleanas:
        if columna in datos_inventario.columns:
            datos_inventario[columna] = datos_inventario[columna].astype(str).str.upper() == "VERDADERO"
            
    if "hay_stock" in datos_inventario.columns:
        datos_inventario["hay_stock"] = datos_inventario["hay_stock"].astype(object)
        mascara_no_nula = datos_inventario["hay_stock"].notnull()
        datos_inventario.loc[mascara_no_nula, "hay_stock"] = datos_inventario.loc[mascara_no_nula, "hay_stock"].astype(str).str.upper() == "VERDADERO"
            
    return datos_inventario

def obtener_ventas_diarias_completas(datos_ventas):
    # agrupar ventas diariamente por producto y completa días sin ventas con 0.
    datos_ventas_reales = datos_ventas[datos_ventas["cantidad_vendida"].notnull()].copy()
    
    fecha_minima = datos_ventas_reales["fecha"].min()
    fecha_maxima = datos_ventas_reales["fecha"].max()
    rango_fechas = pd.date_range(start=fecha_minima, end=fecha_maxima, freq="D")
    
    productos = datos_ventas_reales[["producto_id", "producto_nombre", "categoria"]].drop_duplicates()
    
    indice_combinado = pd.MultiIndex.from_product(
        [rango_fechas, productos["producto_id"].unique()],
        names=["fecha", "producto_id"]
    )
    
    ventas_agrupadas = datos_ventas_reales.groupby(["fecha", "producto_id"]).agg({
        "cantidad_vendida": "sum",
        "precio_unitario": "mean",
        "precio_aplicado": "mean",
        "es_promocion": "max",
        "es_feriado": "max",
        "es_fin_semana": "max",
        "temperatura": "mean",
        "precipitacion": "mean"
    }).reset_index()
    
    ventas_completas = ventas_agrupadas.set_index(["fecha", "producto_id"]).reindex(indice_combinado).reset_index()
    ventas_completas["cantidad_vendida"] = ventas_completas["cantidad_vendida"].fillna(0).astype(int)
    ventas_completas = ventas_completas.merge(productos, on="producto_id", how="left")
    
    ventas_completas["es_promocion"] = ventas_completas["es_promocion"].fillna(False).astype(bool)
    ventas_completas["es_feriado"] = ventas_completas["es_feriado"].fillna(False).astype(bool)
    ventas_completas["es_fin_semana"] = ventas_completas["es_fin_semana"].fillna(False).astype(bool)
    
    ventas_completas["temperatura"] = ventas_completas.groupby("producto_id")["temperatura"].ffill().bfill().fillna(17.0)
    ventas_completas["precipitacion"] = ventas_completas.groupby("producto_id")["precipitacion"].ffill().bfill().fillna(1.0)
    
    precios_promedio = datos_ventas_reales.groupby("producto_id")["precio_unitario"].mean().to_dict()
    ventas_completas["precio_unitario"] = ventas_completas["precio_unitario"].fillna(
        ventas_completas["producto_id"].map(precios_promedio)
    )
    ventas_completas["precio_aplicado"] = ventas_completas["precio_aplicado"].fillna(ventas_completas["precio_unitario"])
    
    return ventas_completas

def obtener_ventas_por_hora(datos_ventas):
    #agrupa ventas por hora del día.
    datos_reales = datos_ventas[datos_ventas["cantidad_vendida"].notnull()]
    ventas_por_hora = datos_reales.groupby(["producto_id", "hora"]).agg({
        "cantidad_vendida": "sum",
        "es_hora_pico": "max"
    }).reset_index()
    
    return ventas_por_hora
