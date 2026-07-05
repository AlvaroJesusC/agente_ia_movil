# -*- coding: utf-8 -*-

"""
Módulo de Algoritmos de Búsqueda para Pymevision AI
---------------------------------------------------
Implementa algoritmos de búsqueda no informada e informada exigidos por el sílabo:
1. BFS (Breadth-First Search): Recorrido por niveles del árbol de categorías de productos.
2. DFS (Depth-First Search): Recorrido en profundidad del árbol de categorías e inspección de productos.
3. A* (A-Star): Búsqueda informada para la optimización de rutas de reabastecimiento en almacén
   y priorización estratégica de pedidos basada en la función de evaluación f(n) = g(n) + h(n)
   con la Distancia Manhattan (Paarth Sonkiya, 2024).
"""

from collections import deque
import heapq
import numpy as np
import pandas as pd


class NodoArbol:
    """
    Representa un nodo dentro del árbol jerárquico de inventario:
    Raíz (Tienda) -> Categoría -> Subcategoría -> Producto
    """
    def __init__(self, nombre, tipo="categoria", datos=None):
        self.nombre = nombre
        self.tipo = tipo  # 'raiz', 'categoria', 'subcategoria', 'producto'
        self.datos = datos if datos is not None else {}
        self.hijos = []

    def agregar_hijo(self, nodo_hijo):
        self.hijos.append(nodo_hijo)


class ArbolCategorias:
    """
    Estructura de Árbol/Grafo de inventario para navegación y búsqueda con BFS y DFS.
    """
    def __init__(self):
        self.raiz = NodoArbol("Bodega Pymevision AI", tipo="raiz")

    def construir_desde_dataframe(self, df_inventario):
        """
        Construye el árbol jerárquico a partir del DataFrame de inventario.
        Estructura:
        Bodega
          ├── Categoría (ej. Bebidas)
          │     ├── Subcategoría (ej. Gaseosas)
          │     │     └── Producto (ej. Inca Kola 500ml)
        """
        # Mapeo simple de categorías a subcategorías heurísticas para armar el árbol
        subcategorias_map = {
            "Inca Kola 500ml": "Gaseosas",
            "Coca Cola 500ml": "Gaseosas",
            "Agua San Luis 625ml": "Aguas y Aguas Frutadas",
            "Yogurt Gloria 1L": "Yogurts y Derivados",
            "Leche Gloria 1L": "Leches Evaporadas",
            "Lays 42g": "Papas Fritas y Snacks",
            "Doritos 42g": "Snacks Salados",
            "InkaChips 40g": "Snacks Salados",
            "Sublime 30g": "Chocolates y Golosinas",
            "Pan de Molde Bimbo 500g": "Panadería y Masas",
            "Arroz Costeño 1kg": "Granos y Cereales",
            "Aceite Primor 1L": "Aceites Comestibles",
            "Fideos Don Vittorio 500g": "Pastas y Fideos",
            "Atún Florida 170g": "Enlatados y Conservas",
            "Detergente Ariel 500g": "Cuidado de la Ropa"
        }

        df_inv = df_inventario.copy()
        cats_by_id = {
            "prod_001": "Bebidas", "prod_002": "Bebidas", "prod_003": "Bebidas",
            "prod_004": "Lácteos", "prod_005": "Lácteos",
            "prod_006": "Snacks", "prod_007": "Snacks", "prod_008": "Snacks", "prod_009": "Snacks",
            "prod_010": "Panadería",
            "prod_011": "Abarrotes", "prod_012": "Abarrotes", "prod_013": "Abarrotes",
            "prod_014": "Conservas",
            "prod_015": "Limpieza"
        }

        categorias_dict = {}
        for _, fila in df_inv.iterrows():
            prod_id = fila["producto_id"]
            prod_nombre = fila.get("producto_nombre", prod_id)
            cat_nombre = fila.get("categoria", cats_by_id.get(prod_id, "General"))
            if pd.isna(cat_nombre) or cat_nombre == "General":
                cat_nombre = cats_by_id.get(prod_id, "General")

            subcat_nombre = subcategorias_map.get(prod_nombre, subcategorias_map.get(prod_id, "Varios"))

            if cat_nombre not in categorias_dict:
                nodo_cat = NodoArbol(cat_nombre, tipo="categoria")
                self.raiz.agregar_hijo(nodo_cat)
                categorias_dict[cat_nombre] = {"nodo": nodo_cat, "subcats": {}}

            subcats_dict = categorias_dict[cat_nombre]["subcats"]
            if subcat_nombre not in subcats_dict:
                nodo_subcat = NodoArbol(subcat_nombre, tipo="subcategoria")
                categorias_dict[cat_nombre]["nodo"].agregar_hijo(nodo_subcat)
                subcats_dict[subcat_nombre] = nodo_subcat

            datos_prod = {
                "producto_id": prod_id,
                "producto_nombre": prod_nombre,
                "stock": fila.get("stock_fisico", 0),
                "precio": fila.get("precio_unitario", 0.0),
                "vencimiento": fila.get("dias_para_vencer", 999),
                "es_perecedero": fila.get("es_perecedero", False),
                "proveedor": fila.get("proveedor", "General")
            }
            nodo_prod = NodoArbol(prod_nombre, tipo="producto", datos=datos_prod)
            subcats_dict[subcat_nombre].agregar_hijo(nodo_prod)

    def bfs_buscar_productos(self, funcion_filtro=None):
        """
        Algoritmo BFS (Breadth-First Search / Búsqueda en Anchura):
        Recorre el árbol por niveles utilizando una COLA FIFO (collections.deque).
        Garantiza explorar primero todas las categorías, luego todas las subcategorías
        y finalmente los productos individuales.
        """
        cola = deque([self.raiz])
        orden_visita = []
        coincidencias = []

        while cola:
            nodo_actual = cola.popleft()
            orden_visita.append(nodo_actual.nombre)

            if nodo_actual.tipo == "producto":
                if funcion_filtro is None or funcion_filtro(nodo_actual):
                    coincidencias.append(nodo_actual)

            for hijo in nodo_actual.hijos:
                cola.append(hijo)

        return coincidencias, orden_visita

    def dfs_buscar_productos(self, funcion_filtro=None):
        """
        Algoritmo DFS (Depth-First Search / Búsqueda en Profundidad):
        Recorre el árbol profundizando en cada rama hasta las hojas utilizando una PILA LIFO.
        Permite una inspección exhaustiva de cada categoría completa.
        """
        pila = [self.raiz]
        orden_visita = []
        coincidencias = []

        while pila:
            nodo_actual = pila.pop()
            orden_visita.append(nodo_actual.nombre)

            if nodo_actual.tipo == "producto":
                if funcion_filtro is None or funcion_filtro(nodo_actual):
                    coincidencias.append(nodo_actual)

            for hijo in reversed(nodo_actual.hijos):
                pila.append(hijo)

        return coincidencias, orden_visita


class OptimizadorAEstrella:
    """
    Implementación del Algoritmo A* (A-Star) para:
    1. Optimización de la Ruta de Reabastecimiento / Recolección en el Almacén de la Bodega.
    2. Priorización Estratégica de Pedidos basada en la Función de Evaluación:
       f(n) = g(n) + h(n)
    
    Referencia Matemática (Sección 2.2 del Informe):
    - g(n): Costo real acumulado de operaciones hasta el nodo actual.
    - h(n): Estimación de la función heurística (Distancia Manhattan):
            d(P1, P2) = |x2 - x1| + |y2 - y1|
    - f(n): Evaluación total de prioridad/costo logístico.
    """

    @staticmethod
    def distancia_manhattan(p1, p2):
        """
        Calcula la distancia Manhattan entre dos puntos P1=(x1, y1) y P2=(x2, y2).
        Fórmula: d(P1, P2) = |x2 - x1| + |y2 - y1|
        """
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

    def __init__(self, dimension_almacen=(10, 10), df_inventario=None):
        self.filas, self.columnas = dimension_almacen
        self.posiciones_estantes = {
            "Entrada / Recepcion": (0, 0),
            "Zona de Despacho": (9, 9)
        }
        if df_inventario is not None:
            self.construir_mapa_desde_dataframe(df_inventario)
        else:
            # Mapa por defecto
            self.posiciones_estantes.update({
                "Inca Kola 500ml": (1, 2), "Coca Cola 500ml": (1, 4), "Agua San Luis 625ml": (1, 7),
                "Yogurt Gloria 1L": (3, 2), "Leche Gloria 1L": (3, 5),
                "Lays 42g": (5, 1), "Doritos 42g": (5, 3), "InkaChips 40g": (5, 6), "Sublime 30g": (5, 8),
                "Pan de Molde Bimbo 500g": (7, 2), "Arroz Costeño 1kg": (8, 1), "Aceite Primor 1L": (8, 4),
                "Fideos Don Vittorio 500g": (8, 6), "Atún Florida 170g": (9, 3), "Detergente Ariel 500g": (9, 7)
            })

    def construir_mapa_desde_dataframe(self, df_inventario):
        """
        Construye dinámicamente la ubicación de los estantes en la grilla del almacén (10x10)
        a partir de los datos cargados del CSV de inventario por el Sensor.
        """
        self.posiciones_estantes = {
            "Entrada / Recepcion": (0, 0),
            "Zona de Despacho": (9, 9)
        }
        col_nombre = "producto_nombre" if "producto_nombre" in df_inventario.columns else "producto_id"
        productos_unicos = df_inventario[col_nombre].dropna().unique()

        # Asignar coordenadas fijas del mapa según el orden en el inventario
        posiciones_disponibles = [
            (1, 2), (1, 4), (1, 7),
            (3, 2), (3, 5),
            (5, 1), (5, 3), (5, 6), (5, 8),
            (7, 2), (8, 1), (8, 4), (8, 6),
            (9, 3), (9, 7)
        ]
        for idx, prod in enumerate(productos_unicos):
            if idx < len(posiciones_disponibles):
                self.posiciones_estantes[prod] = posiciones_disponibles[idx]
            else:
                f = 1 + (idx // 4) * 2
                c = 1 + (idx % 4) * 2
                self.posiciones_estantes[prod] = (min(f, 9), min(c, 9))

    def resolver_ruta_reabastecimiento(self, lista_productos_criticos):
        """
        Encuentra la secuencia de reabastecimiento/picking y la ruta óptima en el almacén
        utilizando A* secuencial de múltiples etapas con Heurística de Vecino Más Cercano (Nearest Neighbor).
        
        Lógica:
        1. Inicia en Entrada / Recepción (0,0).
        2. Selecciona dinámicamente el estante de producto no visitado más cercano (menor Distancia Manhattan).
        3. Navega hacia ese estante usando A* (f(n) = g(n) + h(n), con h(n) hacia el objetivo de esa etapa).
        4. Repite el proceso hasta visitar todos los productos críticos en alerta.
        5. En la etapa final, navega desde el último producto visitado hacia la Zona de Despacho (9,9).
        """
        if not lista_productos_criticos:
            lista_productos_criticos = ["Inca Kola 500ml", "Leche Gloria 1L", "Arroz Costeño 1kg"]

        origen = "Entrada / Recepcion"
        destino_final = "Zona de Despacho"
        pos_destino_final = self.posiciones_estantes.get(destino_final, (9, 9))
        pos_origen = self.posiciones_estantes.get(origen, (0, 0))

        # Filtrar productos válidos en el almacén
        pendientes = [p for p in lista_productos_criticos if p in self.posiciones_estantes and p not in (origen, destino_final)]
        if not pendientes:
            pendientes = [p for p in ["Inca Kola 500ml", "Leche Gloria 1L", "Arroz Costeño 1kg"] if p in self.posiciones_estantes]

        camino_completo_coordenadas = []
        secuencia_nodos = []
        costo_g_acumulado = 0.0

        pos_actual = pos_origen
        actual_nombre = origen

        # Heurística inicial desde (0,0) hacia el primer producto más cercano
        if pendientes:
            primer_producto = min(pendientes, key=lambda p: self.distancia_manhattan(pos_origen, self.posiciones_estantes[p]))
            h_inicial = self.distancia_manhattan(pos_origen, self.posiciones_estantes[primer_producto])
        else:
            h_inicial = self.distancia_manhattan(pos_origen, pos_destino_final)

        # Etapas de picking: visitar todos los productos pendientes por Nearest Neighbor
        while pendientes:
            proximo_nombre = min(pendientes, key=lambda p: self.distancia_manhattan(pos_actual, self.posiciones_estantes[p]))
            pos_proximo = self.posiciones_estantes[proximo_nombre]
            pendientes.remove(proximo_nombre)

            sub_camino, g_sub, _, _ = self._astar_punto_a_punto(pos_actual, pos_proximo)

            if camino_completo_coordenadas:
                camino_completo_coordenadas.extend(sub_camino[1:])
            else:
                camino_completo_coordenadas.extend(sub_camino)

            costo_g_acumulado += g_sub

            if pendientes:
                siguiente_temp = min(pendientes, key=lambda p: self.distancia_manhattan(pos_proximo, self.posiciones_estantes[p]))
                h_nodo = self.distancia_manhattan(pos_proximo, self.posiciones_estantes[siguiente_temp])
            else:
                h_nodo = self.distancia_manhattan(pos_proximo, pos_destino_final)

            f_nodo = costo_g_acumulado + h_nodo

            secuencia_nodos.append({
                "desde": actual_nombre,
                "hacia": proximo_nombre,
                "pos_desde": pos_actual,
                "pos_hacia": pos_proximo,
                "costo_g": costo_g_acumulado,
                "heuristica_h": h_nodo,
                "evaluacion_f": f_nodo
            })

            pos_actual = pos_proximo
            actual_nombre = proximo_nombre

        # Etapa final: desde el último producto hacia Zona de Despacho (9,9)
        sub_camino_final, g_final_sub, _, _ = self._astar_punto_a_punto(pos_actual, pos_destino_final)
        if camino_completo_coordenadas:
            camino_completo_coordenadas.extend(sub_camino_final[1:])
        else:
            camino_completo_coordenadas.extend(sub_camino_final)

        costo_g_acumulado += g_final_sub
        h_final = 0

        secuencia_nodos.append({
            "desde": actual_nombre,
            "hacia": destino_final,
            "pos_desde": pos_actual,
            "pos_hacia": pos_destino_final,
            "costo_g": costo_g_acumulado,
            "heuristica_h": h_final,
            "evaluacion_f": costo_g_acumulado + h_final
        })

        return {
            "secuencia_nodos": secuencia_nodos,
            "camino_coordenadas": camino_completo_coordenadas,
            "costo_g_total": costo_g_acumulado,
            "heuristica_h_total": h_inicial,
            "heuristica_h_inicial": h_inicial,
            "evaluacion_f_inicial": h_inicial,
            "evaluacion_f_total": costo_g_acumulado
        }

    def _astar_punto_a_punto(self, pos_inicio, pos_meta):
        """
        Ejecuta el algoritmo A* desde pos_inicio hasta pos_meta en una grilla 2D.
        Usa la función f(n) = g(n) + h(n) con h(n) = Distancia Manhattan.
        """
        open_set = []
        h_inicial = self.distancia_manhattan(pos_inicio, pos_meta)
        heapq.heappush(open_set, (h_inicial, 0, pos_inicio, [pos_inicio]))

        g_score = {pos_inicio: 0}
        f_score = {pos_inicio: h_inicial}
        visitados = set()

        movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            current_f, current_g, current_pos, path = heapq.heappop(open_set)

            if current_pos == pos_meta:
                h_final = self.distancia_manhattan(current_pos, pos_meta)
                return path, current_g, h_final, current_f

            if current_pos in visitados:
                continue

            visitados.add(current_pos)

            for dx, dy in movimientos:
                vecino = (current_pos[0] + dx, current_pos[1] + dy)

                if 0 <= vecino[0] < self.filas and 0 <= vecino[1] < self.columnas:
                    tentative_g = current_g + 1.0

                    if vecino not in g_score or tentative_g < g_score[vecino]:
                        g_score[vecino] = tentative_g
                        h_vecino = self.distancia_manhattan(vecino, pos_meta)
                        f_vecino = tentative_g + h_vecino
                        f_score[vecino] = f_vecino
                        heapq.heappush(open_set, (f_vecino, tentative_g, vecino, path + [vecino]))

        dist_directa = self.distancia_manhattan(pos_inicio, pos_meta)
        return [pos_inicio, pos_meta], dist_directa, 0, dist_directa

    def priorizar_ordenes_compra_astar(self, lista_alertas):
        """
        Priorización Estratégica de Pedidos con A*:
        Ordena y evalúa las órdenes de compra asignando un puntaje f(n) = g(n) + h(n).
        """
        alertas_criticas = [a for a in lista_alertas if a.get("gravedad") in ["ALTA", "CRITICA", "MEDIA"]]

        mapa_nombres = {
            "prod_001": "Inca Kola 500ml", "prod_002": "Coca Cola 500ml", "prod_003": "Agua San Luis 625ml",
            "prod_004": "Yogurt Gloria 1L", "prod_005": "Leche Gloria 1L",
            "prod_006": "Lays 42g", "prod_007": "Doritos 42g", "prod_008": "InkaChips 40g", "prod_009": "Sublime 30g",
            "prod_010": "Pan de Molde Bimbo 500g",
            "prod_011": "Arroz Costeño 1kg", "prod_012": "Aceite Primor 1L", "prod_013": "Fideos Don Vittorio 500g",
            "prod_014": "Atún Florida 170g", "prod_015": "Detergente Ariel 500g"
        }

        ordenes_priorizadas = []
        for a in alertas_criticas:
            prod_id = a.get("producto_id", "Desconocido")
            prod_nombre = a.get("producto_nombre", mapa_nombres.get(prod_id, prod_id))
            if prod_nombre in mapa_nombres:
                prod_nombre = mapa_nombres[prod_nombre]
            elif prod_id in mapa_nombres:
                prod_nombre = mapa_nombres[prod_id]

            tipo = a.get("tipo", "GENERAL")
            prob_riesgo = a.get("probabilidad_bayesiana", 0.5)

            costo_g = round((1.0 - prob_riesgo) * 50.0 + (10.0 if tipo == "RIESGO_QUIEBRE" else 20.0), 2)
            heuristica_h = round(prob_riesgo * 100.0, 2)
            evaluacion_f = round(costo_g + heuristica_h, 2)

            ordenes_priorizadas.append({
                "producto_id": prod_nombre,
                "tipo_alerta": tipo,
                "probabilidad_riesgo": prob_riesgo,
                "costo_g": costo_g,
                "heuristica_h": heuristica_h,
                "evaluacion_f": evaluacion_f,
                "descripcion": a.get("descripcion", "")
            })

        ordenes_priorizadas.sort(key=lambda x: x["evaluacion_f"], reverse=True)
        return ordenes_priorizadas
