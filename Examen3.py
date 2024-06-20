"""
Examen 3: Métodos y Modelos Computacionales

# Jose Daniel Quintana Fuentes

PLANEACIÓN DE MOVIMIENTO EN ROBÓTICA MULTI-AGENTE

Suponga que usted tiene un espacio de trabajo rectangular, dividido en pequeñas celdas cuadradas.
    - En dicho espacio de trabajo hay obstáculos (celdas cerradas), que no pueden ser visitados por los agentes.
    - En el espacio de trabajo, se distribuyen M tareas (círculos rojos), y N agentes (tortugas).
    - Tanto los obstáculos como las tareas se mueven aleatoriamente cada incierto tiempo.
    - Las tareas podrían tener, o no, orden de prioridad.

Desarrolle un algoritmo para planear el movimiento de los agentes tal que se alcancen las tareas, bien sea, en el orden
de prioridad si la tienen, o minimizando las distancias recorridas si las tareas no tienen prioridad.
    - Cree un informe detallado de su estrategia y metodología de solución.
    - Pruébelo con algunos casos etiquetados.
    - Recuerde contemplar los dos casos: 1) Cuando las tareas tienen prioridad, 2) cuando no la tienen.
    - Recuerde estructurar y formular formalmente los problemas de optimización necesarios, y evaluar las alternativas de solución.
    - En sus conclusiones incluya varias aplicaciones de la vida real en donde un sistema similar es aplicable.
"""

# Implementación un algoritmo para planificar el movimiento de los agentes para alcanzar las tareas asignadas en un espacio de trabajo rectangular con obstáculos.


# Importar librerías
from labyrinth import Labyrinth
import threading
from grafo import Grafo
from globales import candado
import random
import time
import math
import pulp


# Defnir una semilla para la generación de números aleatorios
random.seed(15)

# Macro expansions -sort of
ROWS = 10
COLUMNS = 20
# Definir si las tareas tienen prioridad o no
PRIORIDAD = False # False: Sin prioridad, True: Con prioridad


def create_labyrinth():
    """
    Función para crear espacio de trabajo rectangular dividido en pequeñas celdas cuadradas.
    :return: None
    """
    maze = Labyrinth(ROWS, COLUMNS)
    maze.start()

# Funciones auxiliar para verificar si una celda es válida y obtener el nodo correspondiente a una celda
def es_valido(x, y, rows, columns):
    return 0 <= x < rows and 0 <= y < columns # Verificar si la celda es válida, es decir, si está dentro de los límites del espacio de trabajo

# Función auxiliar para obtener el nodo correspondiente a una celd
def obtener_nodo(x, y, columns):
    return x * columns + y # Obtener el nodo correspondiente a la celda


# Función para generar celdas cerradas (componente conexa) en el espacio de trabajo rectangular
def generar_celda_cerrada(rows, columns, n):
    """
    Función para generar celdas cerradas (componente conexa) en el espacio de trabajo rectangular dividido en pequeñas celdas cuadradas.
    :param rows: Número de filas del espacio de trabajo.
    :param columns: Número de columnas del espacio de trabajo.
    :param n: Número de nodos que compone la componente conexa.
    :return: Lista de nodos que forman la componente conexa.
    """
    # Inicializar conjunto de nodos que forman la componente conexa
    componente_conexa = set()

    # Selección inicial de un nodo aleatorio
    start_x, start_y = random.randint(0, rows - 1), random.randint(0, columns - 1)
    componente_conexa.add(obtener_nodo(start_x, start_y, columns)) # Se añade el nodo inicial a la componente conexa

    # Mientras la componente conexa no tenga el número de nodos deseado n, se añaden nodos aleatorios a la componente conexa.
    while len(componente_conexa) < n:
        # Selección de un nodo aleatorio de la componente existente
        nodo_actual = random.choice(list(componente_conexa))
        x, y = nodo_actual // columns, nodo_actual % columns
        # Selección de un vecino aleatorio
        vecinos = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] if es_valido(x + dx, y + dy, rows, columns)]
        # Si hay vecinos, se selecciona uno aleatoriamente y se añade a la componente conexa
        if vecinos:
            vecino_x, vecino_y = random.choice(vecinos) # Selección de un vecino aleatorio
            nodo_vecino = obtener_nodo(vecino_x, vecino_y, columns) # Obtener el nodo correspondiente al vecino
            componente_conexa.add(nodo_vecino) # Se añade el vecino a la componente conexa

    return componente_conexa

# Función para generar un grafo con obstáculos (celdas cerradas o componentes conexas) en el espacio de trabajo rectangular
def generar_grafo_con_obstaculos(rows, columns, m):
    """
    Función para generar un grafo con obstáculos (celdas cerradas o componentes conexas) en el espacio de trabajo rectangular dividido en pequeñas celdas cuadradas.
    :param rows: Número de filas del espacio de trabajo.
    :param columns: Número de columnas del espacio de trabajo.
    :param m: Número de componentes conexas (celdas cerradas).
    :return: Grafo con obstáculos y nodos de las componentes conexas.
    """
    # Inicialización del grafo
    grafo = Grafo()

    # Generar varias celdas cerradas aleatoriamente en la cuadrícula.
    obstaculos = []
    for _ in range(m):
        n = random.randint(1, 6)  # Número de nodos que compone la componente conexa para cada componente conexa
        print("Números de nodos que componen la componente conexa: ", n)
        componente_conexa = generar_celda_cerrada(rows, columns, n)
        obstaculos.append(componente_conexa)

    # Añadir aristas entre nodos adyacentes que no estén en la componente conexa, permite crear el perímetro de celdas cerradas
    for x in range(rows):  # Recorrer las filas
        for y in range(columns):  # Recorrer las columnas
            nodo = obtener_nodo(x, y, columns)  # Obtener el nodo correspondiente a la celda
            if any(nodo in celda for celda in obstaculos):  # Si el nodo está en una celda cerrada
                continue  # Se continúa con la siguiente iteración

            # Añadir aristas entre nodos adyacentes
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Recorrer los vecinos de la celda (sin diagonales)
                nx, ny = x + dx, y + dy  # Obtener las coordenadas del vecino
                if es_valido(nx, ny, rows, columns) and all(obtener_nodo(nx, ny, columns) not in celda for celda in obstaculos):  # Verificar si el vecino es válido y no está en una celda cerrada
                    nodo_vecino = obtener_nodo(nx, ny, columns)  # Obtener el nodo correspondiente al vecino
                    # Definir diagonales
                    if dx != 0 and dy != 0:
                        grafo.add_edge(nodo, nodo_vecino, math.sqrt(2))
                    else:
                        grafo.add_edge(nodo, nodo_vecino, 1)

    # Añadir aristas entre nodos de componentes conexas
    for celda in obstaculos:
        for nodo in celda:  # Recorrer los nodos de la componente conexa
            x, y = nodo // columns, nodo % columns  # Obtener las coordenadas de la celda
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Recorrer los vecinos de la celda (sin diagonales)
                nx, ny = x + dx, y + dy  # Obtener las coordenadas del vecino
                if es_valido(nx, ny, rows, columns):
                    nodo_vecino = obtener_nodo(nx, ny, columns)  # Obtener el nodo correspondiente al vecino
                    if nodo_vecino in celda:  # Si el vecino está en la misma componente conexa
                        grafo.add_edge(nodo, nodo_vecino, 1) # Añadir arista con peso 1 (eliminar la pared)
                    else:  # Si el vecino no está en la misma componente conexa
                        grafo.add_edge(nodo, nodo_vecino, 0)  # Añadir arista con peso 0 (mantener la pared)
            #Definir diagonales
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Recorrer los vecinos de la celda (con diagonales)
                nx, ny = x + dx, y + dy  # Obtener las coordenadas del vecino
                if es_valido(nx, ny, rows, columns):
                    nodo_vecino = obtener_nodo(nx, ny, columns)
                    if nodo_vecino in celda:  # Si el vecino está en la misma componente conexa
                        grafo.add_edge(nodo, nodo_vecino, math.sqrt(2))  # Añadir arista con peso √2 (eliminar la pared)

    # Obtener los nodos de las componentes conexas
    nodos_obstaculos = [nodo for celda in obstaculos for nodo in celda]
    print("Nodos de las componentes conexas: ", nodos_obstaculos)

    return grafo, nodos_obstaculos # El grafo que retorna tiene la estructura de un diccionario donde las claves son los nodos y los valores son los nodos adyacentes, y los nodos de las componentes conexas

# Función para verificar si una arista diagonal es inválida, es decir, si los dos nodos intermedios entre la arista diagonal son nodos de la componente conexa.
def es_arista_diagonal_invalida(inicio, destino, nodos_obstaculos):
    x1, y1 = divmod(inicio, COLUMNS) # Obtener las coordenadas del nodo de inicio
    x2, y2 = divmod(destino, COLUMNS) # Obtener las coordenadas del nodo de destino

    if abs(x1 - x2) == 1 and abs(y1 - y2) == 1: # Si la arista es diagonal
        intermedio1 = (x1 * COLUMNS + y2) # Calcular el primer nodo intermedio
        intermedio2 = (x2 * COLUMNS + y1) # Calcular el segundo nodo intermedio
        if intermedio1 in nodos_obstaculos and intermedio2 in nodos_obstaculos: # Si los dos nodos intermedios están en la componente conexa
            return True # La arista diagonal es inválida
    return False # La arista diagonal es válida


# Implementación del Algoritmo de Floyd-Warshall para encontrar el camino más corto entre dos nodos en un grafo
def FloydWarshall(grafo, nodos_obstaculos):
    V = grafo.V
    E = grafo.E
    inf = float('inf')
    dist = {u: {v: inf for v in V} for u in V} # Inicializar la matriz de distancias con infinito
    next_node = {u: {v: None for v in V} for u in V} # Inicializar la matriz de nodos siguientes con None

    # Inicializar las distancias y los nodos siguientes con los valores de las aristas
    for u in V:
        dist[u][u] = 0
        for v in V[u]:
            if f"({u}, {v})" in E and not es_arista_diagonal_invalida(u, v, nodos_obstaculos):
                dist[u][v] = E[f"({u}, {v})"]
                next_node[u][v] = v
            if f"({v}, {u})" in E and not es_arista_diagonal_invalida(v, u, nodos_obstaculos):
                dist[v][u] = E[f"({v}, {u})"]
                next_node[v][u] = u
            if dist[u][v] != dist[v][u]:
                dist[u][v] = dist[v][u] = min(dist[u][v], dist[v][u])
                next_node[u][v] = v
                next_node[v][u] = u

    for k in V:
        if k in nodos_obstaculos:
            continue
        for i in V:
            if i in nodos_obstaculos:
                continue
            for j in V:
                if j in nodos_obstaculos:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node


#  Optimización de la asignación de tareas a agentes minimizando la distancia total recorrida por los agentes
#  para alcanzar las tareas asignadas mediante la programación lineal entera (ILP) con la librería PuLP.
def optimization(agentes, tareas, dist):
    """
    Función para optimizar la asignación de tareas a agentes minimizando la distancia total recorrida por los agentes.
    :param agentes: Agentes a los que se asignarán las tareas.
    :param tareas: Tareas que se asignarán a los agentes.
    :param dist: Distancias entre los agentes y las tareas.
    :return:
    """

    # Crear un problema de minimización
    prob = pulp.LpProblem("AssignmentProblem", pulp.LpMinimize)

    # Conjuntos de agentes y tareas
    num_agentes = len(agentes)
    num_tareas = len(tareas)

    # Variables de decisión
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(num_agentes) for j in range(num_tareas)), cat='Binary')

    # Función objetivo: minimizar la distancia total recorrida por los agentes
    prob += pulp.lpSum(dist[agentes[i]][tareas[j]] * x[i, j] for i in range(num_agentes) for j in range(num_tareas))

    # Restricción #1: cada tarea debe ser asignada a un solo agente
    for j in range(num_tareas):
        prob += pulp.lpSum(x[i, j] for i in range(num_agentes)) == 1

    # Restricción #2: cada agente puede ser asignado a más de una tarea, pero debe hacer al menos una tarea si hay más tareas que agentes
    if num_tareas > num_agentes:
        # Cada agente debe tener al menos una tarea
        for i in range(num_agentes):
            prob += pulp.lpSum(x[i, j] for j in range(num_tareas)) <= num_tareas
    else:
        # Cada agente puede tener como máximo una tarea
        for i in range(num_agentes):
            prob += pulp.lpSum(x[i, j] for j in range(num_tareas)) <= 1

    # Resolver el modelo usando CBC
    prob.solve(pulp.PULP_CBC_CMD())

    # Obtener las asignaciones
    assignments = [(i, j) for i in range(num_agentes) for j in range(num_tareas) if pulp.value(x[i, j]) == 1]

    return assignments # Retornar las asignaciones de tareas a agentes, donde cada asignación es una tupla (agente, tarea), donde la tupla agente está compuesta por el índice del agente y la tarea está compuesta por el índice de la tarea (índices, tareas)


# Función para reconstruir la ruta entre dos nodos
def reconstruir_ruta(next_node, inicio, destino):
    """
    Función para reconstruir la ruta entre dos nodos.
    :param next_node: Nodos siguientes en el camino más corto.
    :param inicio: Nodo de inicio.
    :param destino: Nodo de destino.
    :return: Ruta entre par de nodos.
    """
    if inicio is None or destino is None:
        return []
    if next_node[inicio][destino] is None and next_node[destino][inicio] is None:
        return []

    ruta = [inicio]
    while inicio != destino:
        siguiente = next_node[inicio][destino]
        if siguiente is None:
            break
        ruta.append(siguiente)
        inicio = siguiente

    return ruta


# Planificación de los movimientos de los agentes para alcanzar las tareas asignadas, cuando tiene prioridad o no
def planificar_movimientos(agentes, tareas, dist, next_node, nodos_obstaculos, prioridad=False, orden_prioridad=None):
    """
    Función para planificar los movimientos de los agentes para alcanzar las tareas asignadas.
    :param agentes:
    :param tareas:
    :param dist:
    :param next_node:
    :param nodos_obstaculos:
    :param prioridad:
    :param orden_prioridad:
    :return:
    """
    # Asignar tareas a agentes
    asignaciones = optimization(agentes, tareas, dist)
    print("Asignaciones de tareas a agentes:", asignaciones)

    # Agrupar tareas por agente
    tareas_por_agente = {agente: [] for agente in agentes}
    for a, t in asignaciones:
        agente = agentes[a]
        tarea = tareas[t]
        tareas_por_agente[agente].append(tarea)

    # Planificar rutas óptimas para cada agente
    rutas_totales = {}
    for agente in tareas_por_agente:
        tareas_agente = tareas_por_agente[agente]

        # Ordenar las tareas por prioridad si es necesario
        if prioridad:
            tareas_agente.sort(key=lambda t: orden_prioridad[t], reverse=True)

        ruta_optima = []
        # Seleccionar la tarea más cercana al agente minimizando la distancia total recorrida por el agente
        while len(tareas_agente) > 0:
            if len(tareas_agente) > 1:  # Si un agente tiene más de una tarea asignada, se selecciona la tarea más cercana
                if not prioridad:
                    # Si las tareas no tienen prioridad, se selecciona la tarea más cercana
                    siguiente_tarea = min(tareas_agente, key=lambda t: dist[agente][t])
                    ruta = reconstruir_ruta(next_node, agente, siguiente_tarea) # Reconstruir la ruta entre el agente y la tarea
                    ruta_optima.extend(ruta) # Añadir la ruta a la ruta óptima
                    agente = siguiente_tarea # Actualizar la posición del agente
                    tareas_agente.remove(siguiente_tarea) # Eliminar la tarea de la lista de tareas del agente
                else:
                    # Si las tareas tienen prioridad, se selecciona la primera tarea de la lista
                    ruta = reconstruir_ruta(next_node, agente, tareas_agente[0])
                    ruta_optima.extend(ruta)
                    agente = tareas_agente[0]
                    tareas_agente.pop(0)
            else:  # Si un agente tiene una sola tarea, se asigna la tarea al agente
                ruta = reconstruir_ruta(next_node, agente, tareas_agente[0])
                ruta_optima.extend(ruta)
                tareas_agente.pop(0)
                break

        rutas_totales[agente] = ruta_optima
        print(f"Ruta óptima para el agente Final {agente}: {ruta_optima}")

    print("Rutas totales de los agentes a las tareas asignadas:", rutas_totales)

    return rutas_totales

def solve_problem():
    """
    Función para resolver el problema de planificación de movimiento de los agentes.
    :return: None
    """

    def mover_agente(ruta, grafo, candado, tareas):
        if not ruta:  # Verificar que la ruta no esté vacía
            return
        for i in range(len(ruta) - 1):
            # Mover el agente a la siguiente posición
            grafo.turtle[ruta[i]] = ruta[i + 1]
            # Enviar el grafo actualizado a la cola (esto depende de tu implementación específica)
            with candado:
                grafo.send_graph()
            time.sleep(1/4)
            # Eliminar el agente de la posición anterior
            grafo.turtle.pop(ruta[i])
            # Actualizar color (verde) de la tarea si el agente llega a la tarea en la ruta
            if ruta[i + 1] in tareas:
                grafo.colors[ruta[i + 1]] = 'green'
                # Enviar el grafo actualizado a la cola (esto depende de tu implementación específica)
                with candado:
                    grafo.send_graph()
                #time.sleep(1/4)
        # Mover el agente a la posición de la tarea
        grafo.turtle[ruta[-1]] = ruta[-1]
        time.sleep(1/2)
        # Eliminar agente de la posición final
        grafo.turtle.pop(ruta[-1])
        # Actualizar la posición del agente en el grafo
        with candado:
            grafo.send_graph()


    while True:
        # Se crea un grafo con obstáculos (celdas cerradas o componentes conexas) y se añaden tareas y agentes aleatoriamente en el grafo.
        # Número de componentes conexas
        m = random.randint(1, 6) # Número de componentes conexas (celdas cerradas)
        print("Número de componentes conexas: ", m)
        # Generar grafo con obstáculos
        grafo, nodos_obstaculos = generar_grafo_con_obstaculos(ROWS, COLUMNS, m)
        print(grafo.get_graph())

        # Generar tareas y agentes aleatoriamente en el espacio de trabajo rectangular, evitando los nodos de las componentes conexas (celdas cerradas) como posiciones de tareas y agentes.
        # Agregar k tareas y n agentes
        k = random.randint(1, 4) # Número de tareas
        n = random.randint(1, 4) # Número de agentes

        # Obtener los nodos válidos (nodos que no son obstáculos)
        nodos_validos = [nodo for nodo in range(ROWS * COLUMNS) if nodo not in nodos_obstaculos]

        # Elegir aleatoriamente k nodos para las tareas
        tareas = random.sample(nodos_validos, k)
        # Elegir aleatoriamente n nodos para los agentes
        agentes = random.sample(nodos_validos, n)

        # Verificar que no se superpongan nodos de tareas y agentes
        for nodo_tarea in tareas:
            if nodo_tarea in agentes:
                agentes.remove(nodo_tarea)

        for nodo_agente in agentes:
            if nodo_agente in tareas:
                tareas.remove(nodo_agente)

        print("Tareas: ", tareas)
        # Se procede a añadir las tareas representadas como círculos rojos en el espacio de trabajo rectangular y los agentes representados como tortugas en el grafo.

        # Colocar los nodos de tareas y agentes en el grafo
        grafo.colors = {nodo_tarea: 'red' for nodo_tarea in tareas}
        grafo.turtle = {nodo_agente: nodo_agente - 1 for nodo_agente in agentes}
        #grafo.show(True) # Mostrar el grafo en la GUI
        # Actualizar el grafo en la cola
        with candado:
            grafo.send_graph()
        time.sleep(1/8)
        #
        # # Distancias de la función de Floyd-Warshall
        dist, next_node = FloydWarshall(grafo, nodos_obstaculos)

        # El orden de prioridas las tareas se define aleatoriamente de la siguiente manera: se asigna un número aleatorio entre 1 y 10 a cada tarea,
        # donde un número mayor indica una mayor prioridad.

        # Definir el orden de prioridad de las tareas
        orden_prioridad = {t: random.randint(1, 10) for t in tareas}
        print("Orden de prioridad de las tareas: ", orden_prioridad)


        # Planificar los movimientos de los agentes para alcanzar las tareas asignadas
        rutas_totales = planificar_movimientos(agentes, tareas, dist, next_node, nodos_obstaculos, prioridad=PRIORIDAD, orden_prioridad=orden_prioridad)

        # Crear hilos para mover cada agente
        hilos = []
        for agente, ruta in rutas_totales.items():
            hilo = threading.Thread(target=mover_agente, args=(ruta, grafo, candado, tareas))
            hilos.append(hilo)
            hilo.start()

        # Esperar a que todos los hilos terminen
        for hilo in hilos:
            hilo.join()


if __name__ == '__main__':
    # Create two threads to run the create_labyrinth and  functions concurrently
    # create_labyrinth function is executed in a separate thread
    hilo1 = threading.Thread(target=create_labyrinth)
    hilo1.start()  # start the thread

    # solve_problem function is executed in a separate thread
    hilo2 = threading.Thread(target=solve_problem)
    hilo2.start()

    # Wait for the threads to finish
    hilo2.join()
    hilo1.join()