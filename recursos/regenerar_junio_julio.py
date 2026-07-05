# Script para regenerar datos de ventas de Junio y Julio 2026 con multiplicadores


import csv
import random
import os
from datetime import datetime, timedelta

random.seed(42)

CSV_PATH = os.path.join(os.path.dirname(__file__), "csv", "ventas_datasL.csv")

# Catálogo de productos
PRODUCTOS = {
    "prod_001": {"nombre": "Inca Kola 500ml",           "categoria": "Bebidas",    "precio": 2.5},
    "prod_002": {"nombre": "Coca Cola 500ml",            "categoria": "Bebidas",    "precio": 2.5},
    "prod_003": {"nombre": "Agua San Luis 625ml",        "categoria": "Bebidas",    "precio": 1.5},
    "prod_004": {"nombre": "Yogurt Gloria 1L",           "categoria": "Lácteos",    "precio": 5.9},
    "prod_005": {"nombre": "Leche Gloria 1L",            "categoria": "Lácteos",    "precio": 4.5},
    "prod_006": {"nombre": "Lays 42g",                   "categoria": "Snacks",     "precio": 2.0},
    "prod_007": {"nombre": "Doritos 42g",                "categoria": "Snacks",     "precio": 2.0},
    "prod_008": {"nombre": "InkaChips 40g",              "categoria": "Snacks",     "precio": 1.8},
    "prod_009": {"nombre": "Sublime 30g",                "categoria": "Snacks",     "precio": 1.5},
    "prod_010": {"nombre": "Pan de Molde Bimbo 500g",    "categoria": "Panadería",  "precio": 6.5},
    "prod_011": {"nombre": "Arroz Costeño 1kg",          "categoria": "Abarrotes",  "precio": 4.0},
    "prod_012": {"nombre": "Aceite Primor 1L",           "categoria": "Abarrotes",  "precio": 8.5},
    "prod_013": {"nombre": "Fideos Don Vittorio 500g",   "categoria": "Abarrotes",  "precio": 3.2},
    "prod_014": {"nombre": "Atún Florida 170g",          "categoria": "Conservas",  "precio": 5.5},
    "prod_015": {"nombre": "Detergente Ariel 500g",      "categoria": "Limpieza",   "precio": 7.5},
}

# Cantidades base diarias por producto
BASE_DIARIO = {
    "prod_001": (3, 7, 2, 5),
    "prod_002": (3, 7, 2, 5),
    "prod_003": (4, 8, 2, 4),
    "prod_004": (2, 4, 2, 5),
    "prod_005": (1, 4, 2, 4),
    "prod_006": (3, 6, 2, 5),
    "prod_007": (2, 5, 2, 5),
    "prod_008": (3, 6, 2, 5),
    "prod_009": (3, 5, 2, 3),
    "prod_010": (2, 5, 2, 5),
    "prod_011": (1, 4, 2, 5),
    "prod_012": (2, 5, 2, 5),
    "prod_013": (1, 4, 2, 5),
    "prod_014": (1, 2, 2, 5),
    "prod_015": (1, 2, 2, 5),
}

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Funciones de multiplicadores

def es_mundial_fifa(fecha):
    # Mundial FIFA 2026: 11 junio – 19 julio
    inicio = datetime(2026, 6, 11)
    fin = datetime(2026, 7, 19)
    return inicio <= fecha <= fin

def es_dia_padre(fecha):
    # Día del Padre: 21 junio 2026
    return fecha.month == 6 and fecha.day == 21

def es_san_pedro_pablo(fecha):
    # San Pedro y San Pablo: 29 junio
    return fecha.month == 6 and fecha.day == 29

def es_fiestas_patrias(fecha):
    # Fiestas Patrias: 28-29 julio
    return fecha.month == 7 and fecha.day in (28, 29)

def es_quincena(fecha):
    # Quincenas: día 15 y 30 de cada mes
    return fecha.day in (15, 30)

def es_hora_pico(hora):
    # Horas pico: 12-14h y 18-21h
    return hora in range(12, 15) or hora in range(18, 22)

def es_fin_semana(fecha):
    # Sábado=5, Domingo=6
    return fecha.weekday() >= 5

def obtener_evento_especial(fecha):
    # Devuelve el string del evento especial para la columna
    if es_fiestas_patrias(fecha):
        return "Fiestas Patrias"
    if es_dia_padre(fecha):
        return "Día del Padre"
    if es_san_pedro_pablo(fecha):
        return "San Pedro y San Pablo"
    if es_mundial_fifa(fecha):
        return "Mundial FIFA 2026"
    return "ninguno"

def obtener_tipo_feriado(fecha):
    # Devuelve el tipo de feriado
    if es_fiestas_patrias(fecha):
        return "fiestas_patrias"
    if es_san_pedro_pablo(fecha):
        return "feriado_religioso"
    return "ninguno"

def es_feriado(fecha):
    # Devuelve True si es feriado
    return es_fiestas_patrias(fecha) or es_san_pedro_pablo(fecha)

def calcular_multiplicador(fecha, hora, categoria):
    # Calcula el multiplicador total para una venta dada la fecha, hora y categoría.
    mult = 1.0
    es_bebida_snack = categoria in ("Bebidas", "Snacks")
    es_abarrotes = categoria == "Abarrotes"
    
    if es_san_pedro_pablo(fecha):
        mult = 0.5
        if es_hora_pico(hora):
            mult *= random.uniform(1.1, 1.3)
        return mult
    
    if es_fiestas_patrias(fecha):
        if es_bebida_snack or es_abarrotes:
            mult = random.uniform(3.0, 4.0)
        else:
            mult = 2.0
        if es_hora_pico(hora):
            mult *= random.uniform(1.3, 1.7)
        return mult
    
    if es_dia_padre(fecha):
        if es_bebida_snack:
            mult = random.uniform(2.0, 2.5)
        else:
            mult = 1.5
        if es_hora_pico(hora):
            mult *= random.uniform(1.3, 1.7)
        return mult
    
    if es_mundial_fifa(fecha) and es_bebida_snack:
        mult = random.uniform(2.0, 2.5)
    
    if es_quincena(fecha):
        mult *= random.uniform(1.2, 1.4)
    
    if es_fin_semana(fecha):
        mult *= random.uniform(1.1, 1.3)
    
    if es_hora_pico(hora):
        mult *= random.uniform(1.3, 1.7)
    
    return mult

def generar_hora_aleatoria():
    # Genera una hora entre 8 y 21, con mayor probabilidad en horas pico
    horas_posibles = list(range(8, 22))
    pesos = []
    for h in horas_posibles:
        if es_hora_pico(h):
            pesos.append(3.0)
        else:
            pesos.append(1.0)
    return random.choices(horas_posibles, weights=pesos, k=1)[0]

def generar_temperatura_junio_julio(fecha):
    # Temperaturas realistas para Lima en invierno (junio-julio).
    base = 17.0 if fecha.month == 6 else 16.0
    return round(base + random.uniform(-2.5, 3.0), 1)

def generar_precipitacion_junio_julio():
    # Precipitación para Lima en invierno.
    if random.random() < 0.7:
        return round(random.uniform(0.0, 2.5), 1)
    else:
        return 0.0

def generar_ventas_junio_julio():
    # Genera las filas de ventas para junio y julio 2026
    filas = []
    
    inicio = datetime(2026, 6, 1)
    fin = datetime(2026, 7, 31)
    
    fecha_actual = inicio
    while fecha_actual <= fin:
        dia_semana_str = DIAS_SEMANA[fecha_actual.weekday()]
        dia_mes = fecha_actual.day
        mes = fecha_actual.month
        fin_semana = es_fin_semana(fecha_actual)
        feriado = es_feriado(fecha_actual)
        tipo_feriado = obtener_tipo_feriado(fecha_actual)
        evento_especial = obtener_evento_especial(fecha_actual)
        
        for prod_id in sorted(PRODUCTOS.keys()):
            prod = PRODUCTOS[prod_id]
            base = BASE_DIARIO[prod_id]
            
            n_trans_min, n_trans_max = base[2], base[3]
            n_transacciones = random.randint(n_trans_min, n_trans_max)
            
            for _ in range(n_transacciones):
                hora = generar_hora_aleatoria()
                minuto = random.randint(0, 59)
                cant_base = random.randint(base[0], base[1])
                
                mult = calcular_multiplicador(fecha_actual, hora, prod["categoria"])
                cantidad = max(1, round(cant_base * mult))
                
                precio_unitario = prod["precio"]
                if random.random() < 0.15:
                    descuento = random.uniform(0.85, 0.95)
                    precio_aplicado = round(precio_unitario * descuento, 2)
                    es_promo = "FALSO"
                else:
                    precio_aplicado = precio_unitario
                    es_promo = "FALSO"
                
                canal = random.choice(["presencial"] * 7 + ["online"] * 3)
                temp = generar_temperatura_junio_julio(fecha_actual)
                precip = generar_precipitacion_junio_julio()
                
                hora_pico_str = "VERDADERO" if es_hora_pico(hora) else "FALSO"
                fin_semana_str = "VERDADERO" if fin_semana else "FALSO"
                feriado_str = "VERDADERO" if feriado else "FALSO"
                
                fecha_hora = f"{dia_mes:02d}/{mes:02d}/2026 {hora:02d}:{minuto:02d}"
                
                fila = (
                    f"{fecha_hora},"
                    f"{prod_id},"
                    f"{prod['nombre']},"
                    f"{prod['categoria']},"
                    f"{float(cantidad)},"
                    f"{precio_unitario},"
                    f"{precio_aplicado},"
                    f"{es_promo},"
                    f"{canal},"
                    f"{dia_semana_str},"
                    f"{dia_mes},"
                    f"{mes},"
                    f"{hora},"
                    f"{hora_pico_str},"
                    f"{fin_semana_str},"
                    f"{feriado_str},"
                    f"{tipo_feriado},"
                    f"{evento_especial},"
                    f"{temp},"
                    f"{precip}"
                )
                filas.append(fila)
        
        fecha_actual += timedelta(days=1)
    
    return filas

def main():
    print("=" * 70)
    print("  REGENERACIÓN DE DATOS DE JUNIO Y JULIO 2026")
    print("=" * 70)
    
    print("\n[1/4] Leyendo CSV existente...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        todas_lineas = f.readlines()
    
    print(f"      Total de líneas en el archivo: {len(todas_lineas)}")
    
    print("[2/4] Identificando datos de marzo-mayo para preservar...")
    header = todas_lineas[0]
    
    lineas_preservadas = [header]
    for linea in todas_lineas[1:]:
        campos = linea.strip().split(",")
        if len(campos) > 11:
            mes = campos[11].strip()
            if mes not in ("6", "7"):
                lineas_preservadas.append(linea)
    
    while lineas_preservadas and lineas_preservadas[-1].strip() == "":
        lineas_preservadas.pop()
    
    print(f"      Líneas preservadas (header + mar-may): {len(lineas_preservadas)}")
    
    print("[3/4] Generando datos de junio y julio con multiplicadores...")
    filas_nuevas = generar_ventas_junio_julio()
    print(f"      Filas generadas para junio-julio: {len(filas_nuevas)}")
    
    stats = {}
    for fila in filas_nuevas:
        campos = fila.split(",")
        evento = campos[17]
        stats[evento] = stats.get(evento, 0) + 1
    
    print("\n      Distribución por evento especial:")
    for evento, count in sorted(stats.items()):
        print(f"        - {evento}: {count} transacciones")
    
    print("\n[4/4] Escribiendo CSV actualizado...")
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        for linea in lineas_preservadas:
            linea_limpia = linea.rstrip("\r\n")
            f.write(linea_limpia + "\r\n")
        
        for fila in filas_nuevas:
            f.write(fila + "\r\n")
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        total_final = sum(1 for _ in f)
    
    print(f"\n      Total de líneas en el archivo final: {total_final}")
    print(f"      (Header: 1, Datos: {total_final - 1})")
    
    print("\n" + "=" * 70)
    print("  ¡REGENERACIÓN COMPLETADA CON ÉXITO!")
    print("=" * 70)
    print("\nMultiplicadores aplicados:")
    print("  • Mundial FIFA (11 jun – 19 jul): bebidas/snacks x2.0–2.5")
    print("  • Día del Padre (21 jun): bebidas/snacks x2.0–2.5, resto x1.5")
    print("  • San Pedro y San Pablo (29 jun): todas x0.5")
    print("  • Fiestas Patrias (28-29 jul): beb/snacks/abarr x3.0–4.0, resto x2.0")
    print("  • Quincenas (15 y 30): todas x1.2–1.4")
    print("  • Horas pico (12-14h, 18-21h): x1.3–1.7 sobre base del día")

if __name__ == "__main__":
    main()
