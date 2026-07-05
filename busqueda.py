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

    def __init__(self, dimension_almacen=(10, 10)):
        self.filas, self.columnas = dimension_almacen
        # Coordenadas predefinidas de los estantes de productos en el almacén (10x10)
        self.posiciones_estantes = {
            "Entrada / Recepcion": (0, 0),
            "Inca Kola 500ml": (1, 2),
            "Coca Cola 500ml": (1, 4),
            "Agua San Luis 625ml": (1, 7),
            "Yogurt Gloria 1L": (3, 2),
            "Leche Gloria 1L": (3, 5),
            "Lays 42g": (5, 1),
            "Doritos 42g": (5, 3),
            "InkaChips 40g": (5, 6),
            "Sublime 30g": (5, 8),
            "Pan de Molde Bimbo 500g": (7, 2),
            "Arroz Costeño 1kg": (8, 1),
            "Aceite Primor 1L": (8, 4),
            "Fideos Don Vittorio 500g": (8, 6),
            "Atún Florida 170g": (9, 3),
            "Detergente Ariel 500g": (9, 7),
            "Zona de Despacho": (9, 9)
        }

    def resolver_ruta_reabastecimiento(self, lista_productos_criticos):
        """
        Encuentra la secuencia y ruta óptima en el almacén para reabastecer/recolectar
        los productos en estado crítico utilizando A*.
        """
        if not lista_productos_criticos:
            lista_productos_criticos = ["Inca Kola 500ml", "Leche Gloria 1L", "Arroz Costeño 1kg"]

        origen = "Entrada / Recepcion"
        destino_final = "Zona de Despacho"

        puntos_a_visitar = [origen] + [p for p in lista_productos_criticos if p in self.posiciones_estantes] + [destino_final]

        camino_completo_coordenadas = []
        secuencia_nodos = []
        costo_g_acumulado = 0.0
        heuristica_h_acumulada = 0.0
        evaluacion_f_acumulada = 0.0

        for i in range(len(puntos_a_visitar) - 1):
            inicio_nombre = puntos_a_visitar[i]
            meta_nombre = puntos_a_visitar[i+1]
            pos_inicio = self.posiciones_estantes[inicio_nombre]
            pos_meta = self.posiciones_estantes[meta_nombre]

            sub_camino, g_sub, h_sub, f_sub = self._astar_punto_a_punto(pos_inicio, pos_meta)

            if camino_completo_coordenadas:
                camino_completo_coordenadas.extend(sub_camino[1:])
            else:
                camino_completo_coordenadas.extend(sub_camino)

            secuencia_nodos.append({
                "desde": inicio_nombre,
                "hacia": meta_nombre,
                "pos_desde": pos_inicio,
                "pos_hacia": pos_meta,
                "costo_g": g_sub,
                "heuristica_h": h_sub,
                "evaluacion_f": f_sub
            })

            costo_g_acumulado += g_sub
            heuristica_h_acumulada += h_sub
            evaluacion_f_acumulada += f_sub

        return {
            "secuencia_nodos": secuencia_nodos,
            "camino_coordenadas": camino_completo_coordenadas,
            "costo_g_total": costo_g_acumulado,
            "heuristica_h_total": heuristica_h_acumulada,
            "evaluacion_f_total": evaluacion_f_acumulada
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

        ordenes_priorizadas = []
        for a in alertas_criticas:
            prod_id = a.get("producto_id", "Desconocido")
            prod_nombre = a.get("producto_nombre", prod_id)
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
