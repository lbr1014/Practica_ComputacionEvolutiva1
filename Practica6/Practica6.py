"""
Práctica 6: Programación Genética (DEAP) - Breast Cancer
Autoras: Sara Abejón Peréz, Lydia Blanco Ruiz y Beatriz Llorente García
"""

import json
import math
import operator
import random
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from deap import base, creator, gp, tools

#==================== CONFIGURACIÓN FIJA =====================
INPUTS_XLSX = "cancerInputs.xlsx"
TARGETS_XLSX = "cancerTargets.xlsx"

TEST_RATIO = 0.30
SEED = 0

# Parámetros por defecto
POP = 300
CXPB = 0.8
MUTPB = 0.2
TOURN = 3
INIT_MAX_DEPTH = 2
MUT_MAX_DEPTH = 2
MAX_HEIGHT = 6
MAX_NODES = 80

# Mono-objetivo: parada por estancamiento
PATIENCE = 30
EPS = 1e-6

# Multi-objetivo: generaciones fijas
NGEN_MULTI = 100

# Carpeta de salida
RESULTS_ROOT = Path("resultados")


#==================== UTILIDADES DE GUARDADO =====================
def ensure_dir(path: Path) -> Path:
    """
        Crea un directorio si no existe.

        Parámetros:
            path: ruta del directorio a crear.

        Devuelve:
            path: la misma ruta, para encadenar llamadas.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """
        Guarda un diccionario en un fichero JSON.

        Parámetros:
            path: ruta del fichero JSON de salida.
            payload: diccionario con los resultados a serializar.
    """
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_fig(path: Path) -> None:
    """
        Guarda la figura actual de Matplotlib y la cierra.

        Parámetros:
            path: ruta del fichero de imagen.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


#==================== LECTURA DE DATOS (XLSX) =====================
def load_cancer_xlsx(inputs_path: str, targets_path: str):
    """
        Lee los ficheros XLSX del dataset Breast Cancer.

        Parámetros:
            inputs_path: ruta al XLSX con las entradas (9 x 699).
            targets_path: ruta al XLSX con los targets (2 x 699).

        Devuelve:
            X: np.ndarray (699 x 9) con las variables de entrada.
            y: np.ndarray (699) con etiquetas 0/1 (0=benigno, 1=maligno).
    """
    inputs = pd.read_excel(inputs_path, header=None).to_numpy(dtype=float)   
    targets = pd.read_excel(targets_path, header=None).to_numpy(dtype=int)   

    X = inputs.T                     
    Y = targets.T                     
    y = np.argmax(Y, axis=1).astype(int)  
    return X, y


def train_test_split(X, y, test_ratio=0.3, seed=0):
    """
        Divide X e y en train y test de forma aleatoria y reproducible.

        Parámetros:
            X: matriz de características.
            y: etiquetas.
            test_ratio: proporción destinada a test (por defecto 0.30).
            seed: semilla para la aleatoriedad.

        Devuelve:
            Xtr: matriz de características del conjunto de entrenamiento.
            Xte: matriz de características del conjunto de prueba.
            ytr: vector de etiquetas correspondiente al conjunto de entrenamiento.
            yte: vector de etiquetas correspondiente al conjunto de prueba.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


#==================== NODOS =====================
def protected_div(a, b):
    """
        División protegida para evitar divisiones por cero en GP.
        
        Devuelve:
            La división de a entre b o 1 si el valor absoluto de b es muy pequeño.
    """
    if abs(b) < 1e-12:
        return 1.0
    return a / b


def protected_log(a):
    """
        Protege los logaritmo sumandole al valor absoluto de a un valor por defecto (epsilon).
        Con ello evita log(0) y permite argumentos negativos.
    """
    return math.log(abs(a) + 1e-12)


def protected_sqrt(a):
    """
        Protege la raíz cuadrada haciendola siemrpe del valor absoluto de a.
        Con ello evita errores con valores negativos.
    """
    return math.sqrt(abs(a))


def protected_exp(a):
    """
        Protege los exponenciales incluyendo un recorte para evitar overflow.
        Limita el exponente al intervalo [-50, 50].
    """
    if a > 50:
        return math.exp(50)
    if a < -50:
        return math.exp(-50)
    return math.exp(a)


#==================== FITNESS =====================
def make_eval_binary(toolbox, X, y):
    """
        Este método evalua para clasificación binaria.

        La evaluación asigna una clase según el signo de la salida del árbol:
            pred = 1 si f(x) >= 0
            pred = 0 si f(x) < 0

        Fitness (a minimizar): error = 1 - accuracy

        Parámetros:
            toolbox: DEAP toolbox con el compilador registrado.
            X: entradas.
            y: etiquetas.

        Devuelve:
            eval_ind: llamada a la función que recibe un individuo y devuelve el error
    """
    def eval_ind(individual):
        try:
            func = toolbox.compile(expr=individual)
        except (SyntaxError, RecursionError, MemoryError):
            return (1.0,)  # castigo (peor error posible)

        correct = 0
        for row, target in zip(X, y):
            try:
                s = func(*row)
            except Exception:
                return (1.0,)
            pred = 1 if s >= 0.0 else 0
            correct += int(pred == target)

        acc = correct / len(y)
        return (1.0 - acc,)
    return eval_ind


#==================== MÉTRICAS =====================
def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
        Calcula métricas a partir de la matriz de confusión (similares a las que usabamos en la Práctica 1).

        Parámetros:
            y_true: etiquetas reales.
            y_pred: etiquetas predichas.

        Devuelve:
            dict con TP, TN, FP, FN y métricas de accuracy, recall, specificity, precision, f1.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(acc),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def evaluate_predictions(individual, toolbox, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
        Evalúa un individuo sobre un dataset y devuelve (accuracy, preds).
    """
    func = toolbox.compile(expr=individual)
    preds = []
    for row in X:
        s = func(*row)
        preds.append(1 if s >= 0.0 else 0)
    preds = np.array(preds, dtype=int)
    acc = float(np.mean(preds == y))
    return acc, preds


#==================== PARADA POR ESTANCAMIENTO =====================
@dataclass
class StagnationStop:
    patience: int = 30
    eps: float = 1e-6


def eaSimple_stagnation(population, toolbox, cxpb, mutpb, stop: StagnationStop,
                        stats=None, halloffame=None, verbose=True):
    """
    Variante de eaSimple cuya condición de parada consiste en seguir mientras 
    haya mejoras apreciables, es decir, minimización en un máximo de generaciones sin mejora.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluación inicial
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best = halloffame[0].fitness.values[0] if halloffame and len(halloffame) else min(ind.fitness.values[0] for ind in population)
    no_improve = 0
    gen = 0

    while no_improve < stop.patience:
        gen += 1

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        current_best = halloffame[0].fitness.values[0] if halloffame and len(halloffame) else min(ind.fitness.values[0] for ind in population)

        if (best - current_best) > stop.eps:  # minimizamos
            best = current_best
            no_improve = 0
        else:
            no_improve += 1

    return population, logbook


#==================== GRÁFICAS =====================
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    """
        Dibuja la matriz de confusión en una figura de Matplotlib.

        Parámetros:
            y_true: etiquetas reales.
            y_pred: etiquetas predichas.
            title: título de la figura.

        Devuelve:
            fig: la matriz de confusión generada.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benigno(0)", "Maligno(1)"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    return fig


def plot_pareto_front(population, title="Población final y frente de Pareto"):
    """
    Muestar la población final de Pareto, el azul se corresponde al frente de Pareto y el rojo a los que no son del frente.

    En multi-objetivo optimizamos simultáneamente:
        El error, minimizandolo (1 - accuracy)
        La complejidad la cual también se minimiza (profundidad / height del árbol)

    El Frente de Pareto es el conjunto de soluciones NO dominadas (ninguna otra solución es mejor o igual en ambos objetivos y estrictamente mejor en al menos uno.)
    """
    if not population:
        return None

    fronts = tools.sortNondominated(population, k=len(population), first_front_only=True)
    pareto_front = fronts[0] if fronts and fronts[0] else []

    pareto_ids = {id(ind) for ind in pareto_front}

    xs_front, ys_front, xs_rest, ys_rest = [], [], [], []
    for ind in population:
        err, depth = ind.fitness.values
        if id(ind) in pareto_ids:
            xs_front.append(err)
            ys_front.append(depth)
        else:
            xs_rest.append(err)
            ys_rest.append(depth)

    plt.figure()
    if xs_rest:
        plt.scatter(xs_rest, ys_rest, label="No Pareto", alpha=0.7, c="red")
    if xs_front:
        plt.scatter(xs_front, ys_front, label="Pareto", alpha=0.9, c="blue")
    plt.title(title)
    plt.xlabel("Error (1 - accuracy) [min]")
    plt.ylabel("Profundidad (height) [min]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    return pareto_front


def plot_multiobj_evolution(logbook, title="Evolución (multiobjetivo)"):
    """
    Evolución de (error, profundidad) a partir de logbook, ya que en DEAP,
    cuando se usa MultiStatistics, las estadísticas se guardan en
    logbook.chapters["fitness"] y logbook.chapters["size"]
    """
    if logbook is None or len(logbook) == 0:
        return

    # Generaciones (siempre están en el logbook principal)
    try:
        gens = logbook.select("gen")
    except Exception:
        gens = [rec.get("gen", i) for i, rec in enumerate(logbook)]

    # ---- Caso MultiStatistics (chapters) ----
    if hasattr(logbook, "chapters") and "fitness" in logbook.chapters:
        fit_ch = logbook.chapters["fitness"]

        # Estos nombres dependen de lo que registraste: "avg", "min", etc.
        avg = fit_ch.select("avg")  # lista de np.array([avg_err, avg_depth])
        mn  = fit_ch.select("min")  # lista de np.array([min_err, min_depth])

        # Extraemos componente 0 (error) y 1 (depth)
        avg_err   = [float(a[0]) for a in avg]
        avg_depth = [float(a[1]) for a in avg]
        min_err   = [float(a[0]) for a in mn]
        min_depth = [float(a[1]) for a in mn]

        # Ajuste por si gens tiene 1 elemento más/menos
        m = min(len(gens), len(avg_err))
        gens = gens[:m]
        avg_err, avg_depth, min_err, min_depth = avg_err[:m], avg_depth[:m], min_err[:m], min_depth[:m]

    else:
        # ---- Fallback: por si algún día guardas fitness dentro del record ----
        avg_err, min_err, avg_depth, min_depth = [], [], [], []
        for rec in logbook:
            if "fitness" not in rec or not isinstance(rec["fitness"], dict):
                continue
            if "avg" not in rec["fitness"] or "min" not in rec["fitness"]:
                continue
            a = rec["fitness"]["avg"]
            m_ = rec["fitness"]["min"]
            avg_err.append(float(a[0]));   avg_depth.append(float(a[1]))
            min_err.append(float(m_[0]));  min_depth.append(float(m_[1]))

        if len(avg_err) == 0:
            print("Aviso: no hay datos de fitness para dibujar.")
            return

        gens = gens[:len(avg_err)]

    # ---- Dibujo ----
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Error (1 - accuracy) [min]")
    ax1.plot(gens, avg_err, label="Media error")
    ax1.plot(gens, min_err, label="Mínimo error", linestyle="--")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Profundidad (height) [min]")
    ax2.plot(gens, avg_depth, label="Media profundidad")
    ax2.plot(gens, min_depth, label="Mínimo profundidad", linestyle="--")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

def plot_error_vs_depth(population, best=None, title="Error vs profundidad"):
    """
        Función para mostrar la dispersión para mono-objetivo.
    """
    if not population:
        return

    xs = [ind.height for ind in population]
    ys = [float(ind.fitness.values[0]) for ind in population]

    plt.figure()
    plt.scatter(xs, ys, alpha=0.7, label="Individuos")
    if best is not None:
        plt.scatter([best.height], [float(best.fitness.values[0])], marker="*", s=200, label="Mejor (HoF)")
    plt.title(title)
    plt.xlabel("Profundidad (height)")
    plt.ylabel("Error (1 - accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_mono_fitness_evolution(logbook, title="Evolución del fitness (mono-objetivo)"):
    """
        Representa la evolución del fitness, concretamente el min y la media a partir del logbook del mono-objetivo.

        Parámetros:
            logbook: tools.Logbook con los campos generaciones, mínimo y media.
            title: título de la gráfica.
    """
    if not logbook:
        return
    gens = [rec["gen"] for rec in logbook]
    avg = [float(rec["avg"]) for rec in logbook]
    min_ = [float(rec["min"]) for rec in logbook]

    plt.figure()
    plt.title(title)
    plt.plot(gens, min_, label="min")
    plt.plot(gens, avg, label="avg")
    plt.xlabel("Generación")
    plt.ylabel("Error (1 - accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend()


#==================== GP SETUP =====================
def build_pset(n_features: int):
    """
        Construye el conjunto de nodos (terminales y no terminales) para GP.
        Incluye operaciones aritméticas y funciones protegidas.

        Parámetros:
            n_features: número de variables de entrada.

        Devuelve:
            pset: gp.PrimitiveSet configurado.
    """
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)

    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_exp, 1)

    pset.addEphemeralConstant("rand", partial(random.uniform, -2.0, 2.0))

    for i in range(n_features):
        pset.renameArguments(**{f"ARG{i}": f"x{i}"})
    return pset


def ensure_creator(multiobj: bool):
    """
        En caso de que no existan crea las clases DEAP creator necesarias.
        Se comprueba antes de crearlos porque DEAP no permite redefinir creators en 
        la misma sesión de Python.

        Parámetros:
            multiobj: si True crea FitnessMulti/IndividualMulti para ejecutar la opción multi-objetivo;
            si False FitnessMin/IndividualMin para ejecutar la función mono-objetivo.

        Devuelve:
            clase Individual correspondiente.
    """
    if multiobj:
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if "IndividualMulti" not in creator.__dict__:
            creator.create("IndividualMulti", gp.PrimitiveTree, fitness=creator.FitnessMulti)
        return creator.IndividualMulti
    else:
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "IndividualMin" not in creator.__dict__:
            creator.create("IndividualMin", gp.PrimitiveTree, fitness=creator.FitnessMin)
        return creator.IndividualMin


def _apply_config(cfg: Dict[str, Any]) -> None:
    """
        Aplica una configuración a variables globales.
    """
    global POP, CXPB, MUTPB, TOURN, INIT_MAX_DEPTH, MUT_MAX_DEPTH, MAX_HEIGHT, MAX_NODES, PATIENCE, EPS, NGEN_MULTI, SEED, TEST_RATIO
    for k, v in cfg.items():
        if k in globals():
            globals()[k] = v


def pruebas_automaticas() -> List[Dict[str, Any]]:
    """
        Batería de 6 casos que se asemejan a los probados en la Práctica 1.
        En la Practica 1 se cambiaba el número de neuronas (5,10,20,22,13,3), aquí cambiamos el tamaño de la población
        y la altura y los nodos, manteniendo el resto estable, para observar tendencias.
    """
    return [
        {"case": "CASO_1_equiv_5",  "POP": 150, "MAX_HEIGHT": 5, "MAX_NODES": 70, "SEED": 0},
        {"case": "CASO_2_equiv_10", "POP": 250, "MAX_HEIGHT": 6, "MAX_NODES": 80, "SEED": 0},
        {"case": "CASO_3_equiv_20", "POP": 350, "MAX_HEIGHT": 6, "MAX_NODES": 90, "SEED": 0},
        {"case": "CASO_4_equiv_22", "POP": 450, "MAX_HEIGHT": 7, "MAX_NODES": 100, "SEED": 0},
        {"case": "CASO_5_equiv_13", "POP": 300, "MAX_HEIGHT": 7, "MAX_NODES": 90, "SEED": 0},
        {"case": "CASO_6_equiv_3",  "POP": 120, "MAX_HEIGHT": 4, "MAX_NODES": 60, "SEED": 0},
    ]


def _make_exploitation_vectors_from_dataset(X: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    """
        Automatiza las pruebas, incluyendo casos claros  de benigno y maligno, un caso mixto y uno aleatorio.

        Para ello hemos construido los siguientes vectores:
            benigno_claro: primera muestra de clase 0
            maligno_claro: primera muestra de clase 1
            mixto_frontera: promedio entre un benigno y un maligno (es decir, uno de la frontera)
            aleatorio_0_1: vector aleatorio uniforme en [0.1, 1.0] (como en la practica 1)
    """
    rng = np.random.default_rng(seed)

    idx_b = np.where(y == 0)[0][0]
    idx_m = np.where(y == 1)[0][0]

    benign = X[idx_b].copy()
    malign = X[idx_m].copy()
    mixed = 0.5 * benign + 0.5 * malign

    random_vec = rng.uniform(0.1, 1.0, size=X.shape[1])

    return {
        "benigno_claro": benign.tolist(),
        "maligno_claro": malign.tolist(),
        "mixto_frontera": mixed.tolist(),
        "aleatorio_0_1": random_vec.tolist(),
        "idx_benigno": int(idx_b),
        "idx_maligno": int(idx_m),
    }


def evaluate_vectors(individual, toolbox, vectors: Dict[str, Any]) -> Dict[str, Any]:
    """
        Evalúa el individuo en vectores de explotación y devuelve la puntuación real y la clase predicha.
    """
    func = toolbox.compile(expr=individual)
    out: Dict[str, Any] = {}
    for name, vec in vectors.items():
        if not isinstance(vec, list):
            continue
        s = float(func(*np.array(vec, dtype=float)))
        out[name] = {"score": s, "pred": int(1 if s >= 0 else 0)}
    return out


def run_gp(multiobj: bool, save_outputs: bool = True, run_label: str = "run") -> Dict[str, Any]:
    """
        Ejecuta el GP en modo mono-objetivo o multi-objetivo.
        
        Devuelve un dict con resultados y, si save_outputs=True, guarda el JSON y las gráficas (para la automatización de resultados).
    """
    random.seed(SEED)
    np.random.seed(SEED)

    X, y = load_cancer_xlsx(INPUTS_XLSX, TARGETS_XLSX)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_ratio=TEST_RATIO, seed=SEED)

    # Vectores para la automatización de pruebas
    exploit_vectors = _make_exploitation_vectors_from_dataset(X, y, seed=SEED)

    pset = build_pset(X.shape[1])
    Individual = ensure_creator(multiobj)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=INIT_MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    eval_core = make_eval_binary(toolbox, Xtr, ytr)

    def evaluate(ind):
        err = eval_core(ind)[0]
        if multiobj:
            return (err, ind.height)
        return (err,)

    toolbox.register("evaluate", evaluate)

    # Operadores
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=MUT_MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Control de nodos y altura
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_NODES))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_NODES))

    # Selección
    if multiobj:
        toolbox.register("select", tools.selNSGA2)
    else:
        toolbox.register("select", tools.selTournament, tournsize=TOURN)

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    # Valores estadísticos
    if multiobj:
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", lambda x: np.mean(x, axis=0))
        mstats.register("std", lambda x: np.std(x, axis=0))
        mstats.register("min", lambda x: np.min(x, axis=0))
        mstats.register("max", lambda x: np.max(x, axis=0))

    else:
        mstats = None
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)

    logbook = tools.Logbook()


    #==================== EVOLUCIÓN =====================
    if multiobj:
        # Evalua la población inicial
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, f in zip(invalid, fits):
            ind.fitness.values = f

        # Asigna un ranking (crowding) usando NSGA-II
        pop = toolbox.select(pop, len(pop))
        hof.update(pop)

        logbook.header = ["gen", "nevals"] + (mstats.fields if mstats else [])
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=0, nevals=len(invalid), **record)
        print(logbook.stream)

        for gen in range(1, NGEN_MULTI + 1):
            # Selección DCD (k debe ser divisible por 4)
            k = (POP // 4) * 4  # usamos el mayor múltiplo de 4
            if k < 4:
                k = 4  # usamos 4 como mínimo razonable
            parents = tools.selTournamentDCD(pop, k)
            if POP % 2 == 1:
                parents.append(random.choice(pop))
            offspring = [toolbox.clone(ind) for ind in parents]

            # Cruce y mutación
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mut in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            # Evaluar descendencia
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid)
            for ind, f in zip(invalid, fits):
                ind.fitness.values = f

            # Reemplazo NSGA-II
            pop = toolbox.select(pop + offspring, POP)
            hof.update(pop)

            record = mstats.compile(pop) if mstats else {}
            logbook.record(gen=gen, nevals=len(invalid), **record)
            print(logbook.stream)

    else:
        stop = StagnationStop(patience=PATIENCE, eps=EPS)
        pop, logbook = eaSimple_stagnation(
            pop, toolbox,
            cxpb=CXPB, mutpb=MUTPB,
            stop=stop,
            stats=stats, halloffame=hof, verbose=True
        )

    best = hof[0]
    print("\n=== Mejor individuo (train) ===")
    print(best)
    print("fitness:", best.fitness.values)


    #==================== EVALUACIÓN EN TRAIN Y TEST =====================
    try:
        train_acc, train_preds = evaluate_predictions(best, toolbox, Xtr, ytr)
        test_acc, test_preds = evaluate_predictions(best, toolbox, Xte, yte)
    except (SyntaxError, RecursionError, MemoryError, OverflowError) as e:
        print("El mejor individuo no compila o falla en evaluación. Ajusta MAX_HEIGHT/MAX_NODES.", e)
        return {"error": "best_individual_failed", "exception": repr(e)}

    train_metrics = confusion_metrics(ytr, train_preds)
    test_metrics = confusion_metrics(yte, test_preds)

    print("\n=== Métricas (TRAIN) ===")
    print(train_metrics)
    print("\n=== Métricas (TEST) ===")
    print(test_metrics)

    #==================== MULTI-OBJETIVO (PARETO EXPLICITO) ====================
    pareto_info = None
    pareto_front = None
    if multiobj:
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        pareto_info = {
            "definition": "Frente de Pareto = conjunto de individuos NO dominados al minimizar (error, height).",
            "objectives": {"obj1": "error = 1-accuracy (min)", "obj2": "height (min)"},
            "n_pareto": int(len(pareto_front)),
            "examples": [
                {"expr": str(ind), "fitness": [float(ind.fitness.values[0]), float(ind.fitness.values[1])], "height": int(ind.height), "nodes": int(len(ind))}
                for ind in pareto_front[:10]
            ],
        }


    #==================== AUTOMATIZACIÓN PRUEBAS =====================
    exploit_results = evaluate_vectors(best, toolbox, exploit_vectors)


    #==================== JSON =====================
    cfg_snapshot = {
        "SEED": SEED,
        "TEST_RATIO": TEST_RATIO,
        "POP": POP,
        "CXPB": CXPB,
        "MUTPB": MUTPB,
        "TOURN": TOURN,
        "INIT_MAX_DEPTH": INIT_MAX_DEPTH,
        "MUT_MAX_DEPTH": MUT_MAX_DEPTH,
        "MAX_HEIGHT": MAX_HEIGHT,
        "MAX_NODES": MAX_NODES,
        "PATIENCE": PATIENCE,
        "EPS": EPS,
        "NGEN_MULTI": NGEN_MULTI,
    }

    # Listas de dicts serializable
    log_serializable: List[Dict[str, Any]] = []
    for rec in logbook:
        r = dict(rec)
        for k, v in list(r.items()):
            if isinstance(v, np.generic):
                r[k] = v.item()
            if isinstance(v, np.ndarray):
                r[k] = v.tolist()
            if isinstance(v, dict):
                vv = {}
                for kk, vv0 in v.items():
                    if isinstance(vv0, np.ndarray):
                        vv[kk] = vv0.tolist()
                    elif isinstance(vv0, np.generic):
                        vv[kk] = vv0.item()
                    else:
                        try:
                            vv[kk] = vv0.tolist()
                        except Exception:
                            vv[kk] = vv0
                r[k] = vv
        log_serializable.append(r)

    payload: Dict[str, Any] = {
        "run_label": run_label,
        "mode": "multi-objetivo" if multiobj else "mono-objetivo",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": cfg_snapshot,
        "best_individual": {
            "expr": str(best),
            "fitness": [float(x) for x in best.fitness.values],
            "height": int(best.height),
            "nodes": int(len(best)),
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "pareto": pareto_info,
        "exploitation": {
            "vectors": exploit_vectors,
            "results": exploit_results,
        },
        "logbook": log_serializable,
    }

    #==================== GUARDADO DE GRÁFICAS Y JSON =====================
    if save_outputs:
        mode_dir = RESULTS_ROOT / ("multi-objetivo" if multiobj else "mono-objetivo")
        ensure_dir(mode_dir)
        run_dir = ensure_dir(mode_dir / run_label)

        # JSON
        save_json(run_dir / "resultados.json", payload)

        # Matrices de confusión
        plot_confusion(ytr, train_preds, title="Matriz de confusión (TRAIN)")
        save_fig(run_dir / "confusion_train.png")

        plot_confusion(yte, test_preds, title="Matriz de confusión (TEST)")
        save_fig(run_dir / "confusion_test.png")

        # Gráficas evolución y Pareto
        if multiobj:
            plot_pareto_front(pop, title="Población final y frente de Pareto (multiobjetivo)")
            save_fig(run_dir / "pareto_front.png")

            plot_multiobj_evolution(logbook, title="Evolución de error y profundidad (multiobjetivo)")
            save_fig(run_dir / "evolucion_multi.png")
        else:
            plot_error_vs_depth(pop, best=best, title="Error vs profundidad (mono-objetivo)")
            save_fig(run_dir / "error_vs_depth.png")

            plot_mono_fitness_evolution(logbook, title="Evolución del fitness (mono-objetivo)")
            save_fig(run_dir / "evolucion_mono.png")

    return payload


def run_test_suite(multiobj: bool) -> List[Dict[str, Any]]:
    """
        Ejecuta automáticamente la batería de pruebas, es decir, los 6 casos estilo Práctica 1.
        Guarda cada caso en su subcarpeta, generando un resumen.
    """
    base_cfg = {
        "CXPB": CXPB, "MUTPB": MUTPB, "TOURN": TOURN,
        "INIT_MAX_DEPTH": INIT_MAX_DEPTH, "MUT_MAX_DEPTH": MUT_MAX_DEPTH,
        "PATIENCE": PATIENCE, "EPS": EPS,
        "NGEN_MULTI": NGEN_MULTI,
        "TEST_RATIO": TEST_RATIO,
    }

    suite = pruebas_automaticas()
    results: List[Dict[str, Any]] = []

    for case_cfg in suite:
        # Se aplica la configuración
        cfg = dict(base_cfg)
        cfg.update(case_cfg)
        _apply_config(cfg)

        print("\n" + "=" * 80)
        print(f"EJECUTANDO EL {case_cfg['case']}  |  MODO: {'MULTI-OBJETIVO' if multiobj else 'MONO-OBJETIVO'}")
        print("=" * 80)

        out = run_gp(multiobj=multiobj, save_outputs=True, run_label=case_cfg["case"])
        results.append(out)

    # Guarda resumen en un JSON
    mode_dir = RESULTS_ROOT / ("multi-objetivo" if multiobj else "mono-objetivo")
    ensure_dir(mode_dir)
    summary_path = mode_dir / "summary.json"

    summary = {
        "mode": "multi-objetivo" if multiobj else "mono-objetivo",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases": [
            {
                "run_label": r.get("run_label"),
                "config": r.get("config", {}),
                "test_metrics": r.get("metrics", {}).get("test", {}),
                "best": r.get("best_individual", {}),
                "pareto_n": (r.get("pareto") or {}).get("n_pareto"),
            }
            for r in results
            if isinstance(r, dict) and "metrics" in r
        ],
    }
    save_json(summary_path, summary)
    print(f"\nResumen guardado en: {summary_path}")

    return results


#==================== MENU =====================
def menu():
    print("\nPráctica 6 - Programación Genética (Breast Cancer)")
    print(f"Dataset: {INPUTS_XLSX} + {TARGETS_XLSX}")
    print("\nElige modo:")
    print("  1) Mono-objetivo")
    print("  2) Multi-objetivo")
    print("  3) Salir")

    while True:
        op = input("\nOpción: ").strip()
        if op in {"1", "2", "3"}:
            return op
        print("Opción inválida.")


def submenu_run():
    print("\nQué quieres ejecutar:")
    print("  1) Una ejecución")
    print("  2) Pruebas automáticas (similares a las de la Practica 1)")
    print("  3) Volver")
    while True:
        op = input("\nOpción: ").strip()
        if op in {"1", "2", "3"}:
            return op
        print("Opción inválida.")


if __name__ == "__main__":
    while True:
        op = menu()

        if op == "3":
            print("Saliendo del programa.")
            break

        multi = (op == "2")

        sub = submenu_run()
        if sub == "3":
            continue
        if sub == "1":
            run_gp(multiobj=multi, save_outputs=True, run_label="manual")
        elif sub == "2":
            run_test_suite(multiobj=multi)
