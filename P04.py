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
def LeerArchivo(nombreFichero: str)-> Tuple[int, Dict[int, List[int]], List[int], int, dict[int, List[int]]]:
    
    with open(nombreFichero) as fichero:
        time.sleep(1)
        os.system('cls')
        linea = fichero.readline()
        numeros = linea.split()
        
        numeroLibros = int(numeros[0])
        numeroLibrerias = int(numeros[1])
        dias = int(numeros[2])
                
        matriz = [[0 for _ in range(numeroLibrerias)] for _ in range(dias)]

        puntuaciones= fichero.readline()
        linea2 = puntuaciones.split()
        puntuacion:List[int]=[]

        for colum in linea2:
            puntuacion.append(int(colum))
                
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

def Fitness_score(permutacion: List[int], puntuaciones:List[int]) -> int:
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

def procesado(librosLibrerias: Dict[int, List[int]], diasProcesado: Dict[int, List[int]], librosProcesadoAlDia: List[int], dias):
    
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
    puntuaciones, diasProcesado, librosProcesadoAlDia, dias, librosLibrerias = LeerArchivo(nombreFichero)
    
    vector: List[int]=[2,1,3,0,4,4]
    Fitness_score(vector, puntuaciones)
    
    procesado(librosLibrerias, diasProcesado, librosProcesadoAlDia, dias)