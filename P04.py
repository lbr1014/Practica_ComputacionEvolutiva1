"""
Autoras: Sara Abejón Peréz, Lydia Blanco Ruiz y Beatriz Llorente García
"""

# imports
import os
import time
import random
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

from deap import base, creator
from deap import tools
from deap import algorithms

#==================== LECTURA DE ARCHIVO =====================
def LeerArchivo(nombreFichero: str)-> Tuple[List[int], Dict[int, List[int]], List[int], int, Dict[int, List[int]]]:
    """
    Lee un fichero de entrada del problema Book Scanning (HashCode 2020).

    Devuelve:
    - puntuacion: lista con la puntuación de cada libro (indice = id libro)
    - diasProcesado: dict[ id_libreria ] = [dias_signup]
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
        puntuaciones= fichero.readline()
        linea2 = puntuaciones.split()
        puntuacion:List[int]=[]
        for colum in linea2:
            puntuacion.append(int(colum))
        
        # Estructuras de datos por librería        
        numeroLibrosLibreria: List[int] = []
        diasProcesado: Dict[int,List[int]] = {}
        librosProcesadoAlDia: List[int] = []
        idLibroEnLibreria: Dict[int,List[int]] = {}
      
        for i in range(numeroLibrerias):
            # Primera línea: info de la librería
            librerias= fichero.readline()
            informacionLibrosLibrerias = librerias.split()
            
            numeroLibrosLibreria.append(int(informacionLibrosLibrerias[0]))
            
            libros: List[int] = []
            libros.append(int(informacionLibrosLibrerias[1]))
                
            diasProcesado[i] = libros
            librosProcesadoAlDia.append(int(informacionLibrosLibrerias[2]))
            
            # Segunda línea: ids de libros de esa librería
            librerias= fichero.readline()
            librerias = librerias.split()
            
            librosEnLibreria: List[int] = []
            for colum in librerias:
                librosEnLibreria.append(int(colum))
                
            idLibroEnLibreria[i] = librosEnLibreria
                
        print (f'NUMERO LIBROS: {numeroLibros}')
        print (f'NUMERO LIBRERIAS: {numeroLibrerias}')
        print (f'DÍAS: {dias}')
        print (f'PUNTUACIÓN: {puntuacion}')
        print (f'NUMERO LIBROS EN LIBRERIAS: {numeroLibrosLibreria}')
        print (f'NUMERO DÍAS PROCESADO: {diasProcesado}')
        print (f'LIBROS PROCESADO AL DÍA: {librosProcesadoAlDia}')
        print (f'ID LIBRO LIBRERIA: {idLibroEnLibreria}')

    return puntuacion, diasProcesado, librosProcesadoAlDia, dias, idLibroEnLibreria

#==================== FITNESS SOBRE LIBROS =====================
def Fitness_score(permutacion: List[int], puntuaciones:List[int]) -> int:
    """
    Calcula la puntuación total de un conjunto de libros, sin duplicar.
    """
    puntuacion: int = 0
    
    igual:bool = False
    for i in permutacion:
        libro = int(permutacion[i])
        for j in permutacion:
            if permutacion[j] == libro:
                if igual:
                    permutacion.pop(permutacion[j])
                igual = True
        igual = False
    
    for i in permutacion:
        libro = int(permutacion[i])
        puntuacion += int(puntuaciones[libro])
        
    print(f'PUNTUACIÓN: {puntuacion}')
    
    return puntuacion

#==================== EVALUACIÓN DEL INDIVIDUO =====================
def evaluar_individuo(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, List[int]],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]]
) -> Tuple[float]:
    """
    Un individuo es una permutación de IDs de librería.
    Se simula el proceso de signup y escaneo de libros y se devuelve
    la puntuación total obtenida.
    """

    dia_actual = 0
    libros_escaneados = set()
    puntuacion_total = 0

    # Recorremos las librerías en el orden dado por el individuo
    for id_lib in individuo:
        dias_signup = diasProcesado[id_lib][0]

        # Si al terminar el signup ya no quedan días para escanear, dejamos de procesar
        if dia_actual + dias_signup >= dias:
            break  # no tiene sentido seguir con más librerías detrás

        # Firmamos esta librería
        dia_actual += dias_signup

        dias_restantes = dias - dia_actual
        capacidad = dias_restantes * librosProcesadoAlDia[id_lib]

        if capacidad <= 0:
            continue

        # Libros que tiene la librería y que aún no se han escaneado
        libros_candidatos = [
            b for b in librosLibrerias[id_lib]
            if b not in libros_escaneados
        ]

        # Ordenamos por puntuación descendente
        libros_candidatos.sort(key=lambda b: puntuaciones[b], reverse=True)

        # Nos quedamos con tantos como permita la capacidad
        libros_seleccionados = libros_candidatos[:capacidad]

        # Actualizamos conjunto de libros escaneados y puntuación
        for libro in libros_seleccionados:
            libros_escaneados.add(libro)
            puntuacion_total += puntuaciones[libro]

    # DEAP espera una tupla de fitness
    return (float(puntuacion_total),)

def procesado(librosLibrerias: Dict[int, List[int]], diasProcesado: Dict[int, List[int]], librosProcesadoAlDia: List[int], dias):
    
    """"
    diasProcesado=sorted(diasProcesado.items(), reverse=True)
    print(f'DIAS PROCESADOS TRAS LA ORDENACIÓN: {diasProcesado}')
    mejor_valor= diasProcesado[0][1]
    mejor_valor_numero = int(mejor_valor[0])
    dias -= mejor_valor_numero
    print(f'DIAS: {dias}')
    libreria= diasProcesado[0][0]
    libros_procesados_dia: int = librosProcesadoAlDia[libreria]
    print(f'LIBROS PROCESADOS AL DÍA: {libros_procesados_dia}')
    libros_libreria = librosLibrerias[libreria]
    print(f'LIBROS EN LA LIBRERÍA: {libros_libreria}')
    """
    pass

#==================== CONFIGURACIÓN DEL AG (DEAP) =====================
def configuracion(
    puntuaciones: List[int],
    diasProcesado: Dict[int, List[int]],
    librosProcesadoAlDia: List[int],
    dias: int,
    librosLibrerias: Dict[int, List[int]],
    tam_poblacion: int = 50,
    n_generaciones: int = 100,
    prob_cruce: float = 0.7,
    prob_mutacion: float = 0.2
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
    
    #creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Atributo básico: una permutación de IDs de librería
    toolbox.register("indices", random.sample, range(num_librerias), num_librerias)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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
    toolbox.register("mate", tools.cxPartialyMatched)          # cruce para permutaciones
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ===== EJECUCIÓN DEL AG =====
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
    print(f"Mejor orden de librerías: {list(mejor_individuo)}")

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

#==================== CONSTRUIR UNA SOLUCIÓN HASHCODE =====================
def construir_salida_hashcode(
    individuo: List[int],
    puntuaciones: List[int],
    diasProcesado: Dict[int, List[int]],
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
    dia_actual = 0
    libros_escaneados = set()
    plan: Dict[int, List[int]] = {}

    for id_lib in individuo:
        dias_signup = diasProcesado[id_lib][0]
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
    print(A)
    for id_lib, libros in plan.items():
        print(f"{id_lib} {len(libros)}")
        print(" ".join(str(l) for l in libros))

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
            print(f"               {i}) {f}")
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
        
    puntuaciones, diasProcesado, librosProcesadoAlDia, dias, librosLibrerias = LeerArchivo(nombreFichero)
    
    # Ejecutar algoritmo genético
    mejor_ind, mejor_fit = configuracion(
        puntuaciones,
        diasProcesado,
        librosProcesadoAlDia,
        dias,
        librosLibrerias,
        tam_poblacion=50,
        n_generaciones=100
    )
    
    # Construir e imprimir una solución en formato HashCode a partir del mejor individuo
    A, plan = construir_salida_hashcode(
        mejor_ind,
        puntuaciones,
        diasProcesado,
        librosProcesadoAlDia,
        dias,
        librosLibrerias
    )
    
    imprimir_salida_hashcode(A, plan)