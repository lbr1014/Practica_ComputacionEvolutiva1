"""
Autoras: Sara Abejón Peréz, Lydia Blanco Ruiz y Beatriz Llorente García
"""

# imports
import os
import time
import random
import json  
import numpy as np
from typing import Dict, Set, Tuple, List
import matplotlib.pyplot as plt
from collections import Counter

from deap import base, creator
from deap import tools
from deap import algorithms

#==================== LECTURA DE ARCHIVO =====================
def LeerArchivo(nombreFichero: str)-> Tuple[List[int], List[List[str]], Dict[int, int], List[int], int, Dict[int, List[int]]]:
    """
    Lee un fichero de entrada del problema Book Scanning (HashCode 2020).

    Devuelve:
    - puntuacion: lista con la puntuación de cada libro (indice = id libro)
    - diasProcesado: dict[ id_libreria ] =  dias_signup (int)
    - librosProcesadoAlDia: lista donde indice = id_libreria, valor = libros/dia
    - dias: numero total de días disponibles
    - idLibroEnLibreria: dict[ id_libreria ] = lista de ids de libros en esa libreria
    """
    with open(nombreFichero) as fichero:
        time.sleep(1)
        os.system('cls')
        linea = fichero.readline().strip()
        numeros = linea.split()
        
        numeroLibros = int(numeros[0])
        numeroLibrerias = int(numeros[1])
        dias = int(numeros[2])
                
        # Puntuaciones de libros
        puntuaciones= fichero.readline().strip()
        linea2 = puntuaciones.split()
        puntuacion:List[int]=[]
        for colum in linea2:
            puntuacion.append(int(colum))
            
        
        generos = fichero.readline().strip()
        linea3 = [g.strip() for g in generos.split(",") if g.strip()]
        contenido: List[List[str]] = []
        for colum in linea3:
            temas_libro = colum.split() 
            contenido.append(temas_libro) 
        
        # Estructuras de datos por librería        
        numeroLibrosLibreria: List[int] = []
        diasProcesado: Dict[int,int] = {}
        librosProcesadoAlDia: List[int] = []
        idLibroEnLibreria: Dict[int,List[int]] = {}
      
        for i in range(numeroLibrerias):
            # Primera línea: info de la librería
            librerias= fichero.readline().strip()
            informacionLibrosLibrerias = librerias.split()
            
            numeroLibrosLibreria.append(int(informacionLibrosLibrerias[0]))
            
            dias_procesado = int(informacionLibrosLibrerias[1])
                
            diasProcesado[i] = dias_procesado
            librosProcesadoAlDia.append(int(informacionLibrosLibrerias[2]))
            
            # Segunda línea: ids de libros de esa librería
            librerias= fichero.readline().strip()
            librerias = librerias.split()
            
            librosEnLibreria: List[int] = []
            for colum in librerias:
                librosEnLibreria.append(int(colum))
                
            idLibroEnLibreria[i] = librosEnLibreria
                
        print (f'NUMERO LIBROS: {numeroLibros}')
        print (f'NUMERO LIBRERIAS: {numeroLibrerias}')
        print (f'DÍAS: {dias}')
        print (f'PUNTUACIÓN: {puntuacion}')
        print (f'CONTENIDO: {contenido}')
        print (f'NUMERO LIBROS EN LIBRERIAS: {numeroLibrosLibreria}')
        print (f'NUMERO DÍAS PROCESADO: {diasProcesado}')
        print (f'LIBROS PROCESADO AL DÍA: {librosProcesadoAlDia}')
        print (f'ID LIBRO LIBRERIA: {idLibroEnLibreria}')

    return puntuacion, contenido, diasProcesado, librosProcesadoAlDia, dias, idLibroEnLibreria

#==================== EVALUACIÓN DEL INDIVIDUO =====================
def simular_individuo(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]]
) -> Tuple[int, Set[int]]:
    """
    Simula el comportamiento de un individuo (orden de librerías).
    Devuelve la puntuación total y el conjunto de libros escaneados.
    """
    dia_actual: int = 0
    libros_escaneados: Set[int] = set()
    puntuacion_total: int = 0

    for id_lib in individuo:
        dias_signup = diasProcesado[id_lib]

        if dia_actual + dias_signup >= dias:
            break

        dia_actual += dias_signup

        dias_restantes = dias - dia_actual
        capacidad = dias_restantes * librosProcesadoAlDia[id_lib]

        if capacidad <= 0:
            continue

        libros_candidatos = [
            b for b in librosLibrerias[id_lib]
            if b not in libros_escaneados
        ]

        libros_candidatos.sort(key=lambda b: puntuaciones[b], reverse=True)
        libros_seleccionados = libros_candidatos[:capacidad]

        for libro in libros_seleccionados:
            libros_escaneados.add(libro)
            puntuacion_total += puntuaciones[libro]

    return puntuacion_total, libros_escaneados


def calcular_balance(libros_escaneados: Set[int], contenido: List[List[str]]) -> float:
    """
    Calcula una métrica de balance de temas.
    balance = min_count / max_count 
    contenido[i] es el tema del libro i.
    """
    if not libros_escaneados:
        return 0.0

    temas: List[str] = []

    for i in libros_escaneados:
        if 0 <= i < len(contenido):
            temas.extend(contenido[i])    
    
    if not temas:
        return 0.0
    
    contador = Counter(temas)

    if not contador:
        return 0.0

    max_count = max(contador.values())
    min_count = min(contador.values())

    if max_count == 0:
        return 0.0

    return min_count / max_count


def evaluar_individuo(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]]
) -> Tuple[float]:
    """
    Maximiza la puntuación total.
    """
    puntuacion_total, _ = simular_individuo(
        individuo, puntuaciones, diasProcesado,
        librosProcesadoAlDia, dias, librosLibrerias
    )
    return (float(puntuacion_total),)

def evaluar_individuo_con_restriccion(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    contenido: List[List[str]],
    umbral_balance: float = 0.6,
    fitness_penalizado: float = 0.0
) -> Tuple[float]:
    """
    Evaluación con restricción de balance:
    - Si el balance es mayor o igual al umbral_balance entonces
    el fitness ess igual puntuación_total
    - Si no el fitness será igual al fitness_penalizado
    """
    puntuacion_total, libros_escaneados = simular_individuo(
        individuo, puntuaciones, diasProcesado,
        librosProcesadoAlDia, dias, librosLibrerias
    )

    balance = calcular_balance(libros_escaneados, contenido)

    if balance < umbral_balance:
        return (float(fitness_penalizado),)
    else:
        return (float(puntuacion_total),)

def evaluar_individuo_multi(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    contenido: List[List[str]]
) -> Tuple[float, float]:
    """
    Evaluación multiobjetivo:
    - Objetivo 1: maximizar puntuación_total
    - Objetivo 2: maximizar balance de temas
    """
    puntuacion_total, libros_escaneados = simular_individuo(
        individuo, puntuaciones, diasProcesado,
        librosProcesadoAlDia, dias, librosLibrerias
    )

    balance = calcular_balance(libros_escaneados, contenido)

    return (float(puntuacion_total), float(balance))

#==================== CONFIGURACIÓN DEL AG (DEAP) =====================
def configuracion(
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    tam_poblacion: int = 50,
    n_generaciones: int = 100,
    prob_cruce: float = 0.7,
    prob_mutacion: float = 0.8
):
    """
    Configura y ejecuta el algoritmo genético con DEAP.
    Devuelve el mejor individuo encontrado y su fitness.
    """
    num_librerias = len(librosLibrerias)
    
    # Evitar error si se ejecuta varias veces el script
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Atributo básico: una permutación de IDs de librería
    toolbox.register("indices", random.sample, range(num_librerias), num_librerias) # Permutación aleatoria de librerias
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices) # Crea un individuo a partir de la permutación anterior
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Lista de individuos

    # Registro de la función de evaluación
    toolbox.register(
        "evaluate",
        evaluar_individuo,
        puntuaciones=puntuaciones,
        diasProcesado=diasProcesado,
        librosProcesadoAlDia=librosProcesadoAlDia,
        dias=dias,
        librosLibrerias=librosLibrerias
    )
    
    # Operadores genéticos
    toolbox.register("mate", tools.cxPartialyMatched) # Cruce para permutaciones (PMX)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=prob_mutacion) # mutación que mezcla los índies dependiendo de la probabilidad de mutación
    toolbox.register("select", tools.selTournament, tournsize=3) # Selección por torneo de tamaño 3

    # ===== EJECUCIÓN DEL ALGORÍTMO GENÉTICO =====
    random.seed(42)

    poblacion = toolbox.population(n=tam_poblacion)

    # HallOfFame para guardar el mejor individuo
    halloffame = tools.HallOfFame(1)
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
        
    print("\n==> Comenzando algoritmo genético...\n")
    poblacion, logbook = algorithms.eaSimple(
        poblacion,
        toolbox,
        cxpb=prob_cruce,
        mutpb=prob_mutacion,
        ngen=n_generaciones,
        stats=stats,
        halloffame=halloffame,
        verbose=True
    )

    mejor_individuo = halloffame[0]
    mejor_fitness = mejor_individuo.fitness.values[0]
    
    print("\n==> Algoritmo genético terminado.")
    print(f"Mejor fitness encontrado: {mejor_fitness}")
    print(f"Mejor orden de librerías (mejor individuo): {list(mejor_individuo)}")

    # Representa la evolución del fitness
    try:
        gen = logbook.select("gen")
        maxs = logbook.select("max")
        avgs = logbook.select("avg")

        plt.figure()
        plt.plot(gen, maxs, label="max")
        plt.plot(gen, avgs, label="avg")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Evolución del fitness")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("No se ha podido representar la evolución del fitness:", e)

    return mejor_individuo, mejor_fitness

def configuracion_restriccion(
    puntuaciones: List[int],
    contenido: List[List[str]],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    tam_poblacion: int = 50,
    n_generaciones: int = 100,
    prob_cruce: float = 0.7,
    prob_mutacion: float = 0.8,
    umbral_balance: float = 0.6,
    fitness_penalizado: float = 0.0
):
    """
    Configuración del AG con restricción de balance de temas.
    """
    num_librerias = len(librosLibrerias)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(num_librerias), num_librerias)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluar_individuo_con_restriccion,
        puntuaciones=puntuaciones,
        diasProcesado=diasProcesado,
        librosProcesadoAlDia=librosProcesadoAlDia,
        dias=dias,
        librosLibrerias=librosLibrerias,
        contenido=contenido,
        umbral_balance=umbral_balance,
        fitness_penalizado=fitness_penalizado
    )

    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=prob_mutacion)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)
    poblacion = toolbox.population(n=tam_poblacion)
    halloffame = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("\n==> Comenzando AG con restricción de balance...\n")
    poblacion, logbook = algorithms.eaSimple(
        poblacion,
        toolbox,
        cxpb=prob_cruce,
        mutpb=prob_mutacion,
        ngen=n_generaciones,
        stats=stats,
        halloffame=halloffame,
        verbose=True
    )

    mejor_individuo = halloffame[0]
    mejor_fitness = mejor_individuo.fitness.values[0]

    print("\n==> AG con restricción terminado.")
    print(f"Mejor fitness (con restricción): {mejor_fitness}")
    print(f"Mejor orden de librerías: {list(mejor_individuo)}")

    # Gráfica de evolución
    try:
        gen = logbook.select("gen")
        maxs = logbook.select("max")
        avgs = logbook.select("avg")

        plt.figure()
        plt.plot(gen, maxs, label="max")
        plt.plot(gen, avgs, label="avg")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Evolución del fitness (restricción)")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("No se ha podido representar la evolución del fitness (restricción):", e)

    return mejor_individuo, mejor_fitness

def configuracion_multiobjetivo(
    puntuaciones: List[int],
    contenido: List[List[str]],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    tam_poblacion: int = 50,
    n_generaciones: int = 100,
    prob_cruce: float = 0.7,
    prob_mutacion: float = 0.8
):
    """
    Configuración del AG multiobjetivo (puntuación, balance) con NSGA-II.
    Devuelve el frente de Pareto (ParetoFront).
    """
    num_librerias = len(librosLibrerias)

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "IndividualMulti"):
        creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    toolbox.register("indices", random.sample, range(num_librerias), num_librerias)
    toolbox.register("individual", tools.initIterate, creator.IndividualMulti, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluar_individuo_multi,
        puntuaciones=puntuaciones,
        diasProcesado=diasProcesado,
        librosProcesadoAlDia=librosProcesadoAlDia,
        dias=dias,
        librosLibrerias=librosLibrerias,
        contenido=contenido
    )

    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=prob_mutacion)
    toolbox.register("select", tools.selNSGA2)

    random.seed(42)
    poblacion = toolbox.population(n=tam_poblacion)

    # Evaluar población inicial
    for ind in poblacion:
        ind.fitness.values = toolbox.evaluate(ind)

    # Aplicar NSGA-II para clasificar la población inicial
    poblacion = toolbox.select(poblacion, len(poblacion))

    pareto_front = tools.ParetoFront()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: np.mean(fits, axis=0))
    stats.register("min", lambda fits: np.min(fits, axis=0))
    stats.register("max", lambda fits: np.max(fits, axis=0))

    print("\n==> Comenzando AG multiobjetivo (NSGA-II)...\n")

    for gen in range(1, n_generaciones + 1):
        offspring = algorithms.varAnd(poblacion, toolbox, cxpb=prob_cruce, mutpb=prob_mutacion)

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        poblacion = toolbox.select(poblacion + offspring, tam_poblacion)
        pareto_front.update(poblacion)

        fits = [ind.fitness.values for ind in poblacion]
        record = stats.compile(poblacion)
        print(f"Gen {gen}: avg={record['avg']}, max={record['max']}")

    print("\n==> AG multiobjetivo terminado.")
    print(f"Tamaño del frente de Pareto: {len(pareto_front)}")

    # Gráfica del frente de Pareto (puntuación vs balance)
    try:
        xs = [ind.fitness.values[0] for ind in pareto_front]
        ys = [ind.fitness.values[1] for ind in pareto_front]

        plt.figure()
        plt.scatter(xs, ys)
        plt.xlabel("Puntuación total")
        plt.ylabel("Balance de temas")
        plt.title("Frente de Pareto (puntuación vs balance)")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("No se ha podido representar el frente de Pareto:", e)

    return pareto_front


#==================== CONSTRUIR UNA SOLUCIÓN HASHCODE =====================
def construir_salida_hashcode(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, int],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]]
) -> Tuple[int, Dict[int, List[int]]]:
    """
    A partir del mejor individuo, construye una solución con el siguiente formato:
    - número de librerías usadas
    - para cada librería: id_libreria, número de libros, lista de libros a escanear

    Devuelve:
    - A: número de librerías efectivamente usadas
    - plan: dict[ id_libreria ] = lista de libros a escanear en esa librería (en orden)
    """
    dia_actual: int = 0
    libros_escaneados: Set[int] = set()
    plan: Dict[int, List[int]] = {}

    for id_lib in individuo:
        dias_signup = diasProcesado[id_lib]
        if dia_actual + dias_signup >= dias:
            break

        dia_actual += dias_signup
        dias_restantes = dias - dia_actual
        capacidad = dias_restantes * librosProcesadoAlDia[id_lib]

        if capacidad <= 0:
            continue

        libros_candidatos = [
            b for b in librosLibrerias[id_lib]
            if b not in libros_escaneados
        ]

        libros_candidatos.sort(key=lambda b: puntuaciones[b], reverse=True)
        libros_seleccionados = libros_candidatos[:capacidad]

        if not libros_seleccionados:
            continue
        
        plan[id_lib] = libros_seleccionados
        for libro in libros_seleccionados:
            libros_escaneados.add(libro)

    A = len(plan)
    return A, plan

def imprimir_salida_hashcode(A: int, plan: Dict[int, List[int]]):
    """
    Imprime por pantalla el plan en el formato de salida HashCode.
    """
    print("\n===== SOLUCIÓN EN FORMATO HASHCODE =====")
    print(f'NÚMERO DE LIBRERÍAS QUE SE USAN EN LA SOLUCIÓN: {A}\n')
    for id_lib, libros in plan.items():
        print(f"ID LIBRERIA: {id_lib} \nNÚMERO DE LIBROS ESCANEADOS POR LA LIBRERÍA: {len(libros)}")
        print(f'IDs DE LOS LIBROS ESCANEADOS POR LA LIBRERÍA {id_lib}:')
        print(" ".join(str(l) for l in libros))
        print()

#==================== GUARDAR RESULTADOS EN JSON ====================
def guardar_resultados_json(
    nombre_json: str,
    nombre_entrada: str,
    mejor_individuo: List[int],
    mejor_fitness: float,
    A: int,
    plan: Dict[int, List[int]]
) -> None:
    """
    Guarda en un .json:
    - nombre del fichero de entrada
    - mejor individuo (orden de librerías)
    - mejor fitness
    - número de librerías usadas
    - plan HashCode (librería -> lista de libros)
    """
    datos = {
        "archivo": nombre_entrada,
        "mejor_individuo": list(mejor_individuo),
        "mejor_fitness": float(mejor_fitness),
        "número_librerias_usadas": A,
        "id_libros_leidos_por_cada_libreria": {str(lib): libros for lib, libros in plan.items()}
    }

    with open(nombre_json, "w", encoding="utf-8") as f:
        f.write(generar_json(datos, indent=0))

    print(f"\nResultados guardados en: {nombre_json}")
    
def generar_json(obj, indent=0) -> str:
    """
    Genera un JSON donde:
    - los diccionarios se indentan en varias líneas
    - las listas se quedan en una sola línea: [a, b, c]
    """
    espacio = " " * indent

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        lineas = []
        for k, v in obj.items():
            valor_str = generar_json(v, indent + 4)
            linea = f'{" " * (indent + 4)}{json.dumps(k, ensure_ascii=False)}: {valor_str}'
            lineas.append(linea)
        return "{\n" + ",\n".join(lineas) + "\n" + espacio + "}"
    
    elif isinstance(obj, list):
        # Las listas siempre en una sola línea
        elementos = ", ".join(generar_json(e, 0) for e in obj)
        return "[" + elementos + "]"

    else:
        # Tipos básicos
        return json.dumps(obj, ensure_ascii=False)
    
#==================== SELECCIÓN DE FICHERO ====================
def ArchivosDirectorio(directorio: str = ".") -> str | None:
    """Muestra un menú con todos los .txt del directorio y devuelve el elegido."""
    os.system('cls')
    ficheros = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    if not ficheros:
        print("No se encontraron ficheros .txt en el directorio actual.")
        return None
    
    while True:
        print("   ______________________________________")
        print("             SELECCIONA FICHERO           ")
        print("   -------------------------------------- ")
        for i, f in enumerate(ficheros, start=1):
            print(f"{i}) {f}")
        print("   ______________________________________\n")

        elec = input(f"Elige un fichero [1-{len(ficheros)}]: ").strip()
        if elec.isdigit() and 1 <= int(elec) <= len(ficheros):
            return ficheros[int(elec) - 1]
        print("Opción no válida, intenta de nuevo.")

########### MAIN ######################
if __name__ == "__main__":
    nombreFichero = ArchivosDirectorio(".")
    if nombreFichero is None:
        exit(1)

    puntuaciones, contenido, diasProcesado, librosProcesadoAlDia, dias, librosLibrerias = LeerArchivo(nombreFichero)

    print("\n¿Qué variante quieres ejecutar?")
    print("  1) AG simple (sin balance)")
    print("  2) AG con restricción de balance")
    print("  3) AG multiobjetivo (puntuación, balance)")
    opcion = input("Elige opción [1-3]: ").strip()

    if opcion == "2":
        mejor_ind, mejor_fit = configuracion_restriccion(
            puntuaciones,
            contenido,
            diasProcesado,
            librosProcesadoAlDia,
            dias,
            librosLibrerias,
            tam_poblacion=50,
            n_generaciones=100,
            umbral_balance=0.6,       
            fitness_penalizado=0.0   
        )
    elif opcion == "3":
        pareto_front = configuracion_multiobjetivo(
            puntuaciones,
            contenido,
            diasProcesado,
            librosProcesadoAlDia,
            dias,
            librosLibrerias,
            tam_poblacion=50,
            n_generaciones=100
        )
        # Para construir la salida HashCode, elegimos del frente la solución con mayor puntuación
        mejor_ind = max(pareto_front, key=lambda ind: ind.fitness.values[0])
        mejor_fit = mejor_ind.fitness.values[0]
        print(f"\nMejor individuo elegido del frente de Pareto (por puntuación): {list(mejor_ind)}")
        print(f"Puntuación: {mejor_fit}, Balance: {mejor_ind.fitness.values[1]}")
    else:
        # Versión simple (P04)
        mejor_ind, mejor_fit = configuracion(
            puntuaciones,
            diasProcesado,
            librosProcesadoAlDia,
            dias,
            librosLibrerias,
            tam_poblacion=50,
            n_generaciones=100
        )

    # Construir e imprimir una solución en formato HashCode a partir del individuo elegido
    A, plan = construir_salida_hashcode(
        mejor_ind,
        puntuaciones,
        diasProcesado,
        librosProcesadoAlDia,
        dias,
        librosLibrerias
    )

    imprimir_salida_hashcode(A, plan)

    # Guardar todo en JSON
    nombre_json = os.path.splitext(nombreFichero)[0] + "_resultado.json"
    guardar_resultados_json(nombre_json, nombreFichero, mejor_ind, mejor_fit, A, plan)
