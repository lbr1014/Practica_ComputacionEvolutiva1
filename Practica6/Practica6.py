"""
Autoras: Sara Abejón Peréz, Lydia Blanco Ruiz y Beatriz Llorente García
Práctica 6 - Programación Genética (DEAP) - Breast Cancer (Inputs/Targets XLSX)
"""

import math
import operator
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

from deap import base, creator, gp, tools, algorithms


# =============================
# CONFIG FIJA (sin parámetros)
# =============================
INPUTS_XLSX = "cancerInputs.xlsx"
TARGETS_XLSX = "cancerTargets.xlsx"

TEST_RATIO = 0.30
SEED = 0

# Parámetros por defecto (puedes tocar aquí o vía menú)
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


# -----------------------------
#  Lectura de datos (XLSX)
# -----------------------------
def load_cancer_xlsx(inputs_path: str, targets_path: str):
    inputs = pd.read_excel(inputs_path, header=None).to_numpy(dtype=float)   # 9 x 699
    targets = pd.read_excel(targets_path, header=None).to_numpy(dtype=int)   # 2 x 699

    X = inputs.T                      # 699 x 9
    Y = targets.T                     # 699 x 2
    y = np.argmax(Y, axis=1).astype(int)  # 699, etiquetas 0/1
    return X, y


def train_test_split(X, y, test_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


# -----------------------------
#  Primitivas "protegidas"
# -----------------------------
def protected_div(a, b):
    if abs(b) < 1e-12:
        return 1.0
    return a / b

def protected_log(a):
    return math.log(abs(a) + 1e-12)

def protected_sqrt(a):
    return math.sqrt(abs(a))

def protected_exp(a):
    if a > 50:
        return math.exp(50)
    if a < -50:
        return math.exp(-50)
    return math.exp(a)


# -----------------------------
#  Fitness (binario 0/1)
# -----------------------------
def make_eval_binary(toolbox, X, y):
    # minimiza error = 1 - accuracy
    def eval_ind(individual):
        try:
            func = toolbox.compile(expr=individual)
        except (SyntaxError, RecursionError, MemoryError):
            return (1.0,)  # castigo: peor error posible

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



# -----------------------------
#  Parada por estancamiento
# -----------------------------
@dataclass
class StagnationStop:
    patience: int = 30
    eps: float = 1e-6

def eaSimple_stagnation(population, toolbox, cxpb, mutpb, stop: StagnationStop,
                        stats=None, halloffame=None, verbose=True):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Eval inicial
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

# -----------------------------
#  Gráficas (Matplotlib)
# -----------------------------
def plot_pareto_front(population, title="Frente de Pareto"):
    """Scatter de población final: azul = frente de Pareto, rojo = no frente.
    Objetivos (minimizar): error y profundidad (height).
    """
    if not population:
        return

    fronts = tools.sortNondominated(population, k=len(population), first_front_only=True)
    pareto_front = fronts[0] if fronts and fronts[0] else []

    # Usamos ids para poder hacer membership rápido sin set(individuos)
    pareto_ids = {id(ind) for ind in pareto_front}

    xs_front, ys_front, xs_rest, ys_rest = [], [], [], []
    for ind in population:
        err, depth = ind.fitness.values
        if id(ind) in pareto_ids:
            xs_front.append(err); ys_front.append(depth)
        else:
            xs_rest.append(err); ys_rest.append(depth)

    plt.figure()
    if xs_rest:
        plt.scatter(xs_rest, ys_rest, label="No óptimos de Pareto", alpha=0.7, c="red")
    if xs_front:
        plt.scatter(xs_front, ys_front, label="Óptimos de Pareto", alpha=0.9, c="blue")
    plt.title(title)
    plt.xlabel("Error (1 - accuracy) [min]")
    plt.ylabel("Profundidad (height) [min]")
    plt.grid(True, alpha=0.3)
    plt.legend()



def plot_multiobj_evolution(logbook, title="Evolución de objetivos (multiobjetivo)"):
    """Evolución de (error, profundidad).
    Soporta dos formatos de logbook:
      1) MultiStatistics (DEAP): rec['fitness']['avg'] -> array([avg_err, avg_depth])
      2) Statistics simple: rec['avg'] / rec['min'] -> tupla/lista/np.array de 2 objetivos
    """
    if logbook is None or len(logbook) == 0:
        return

    gens = [rec.get("gen", i) for i, rec in enumerate(logbook)]

    def _get_2(obj, idx):
        # extrae componente idx de tupla/lista/np.array
        try:
            return float(obj[idx])
        except Exception:
            return float(obj)

    avg_err = []
    min_err = []
    avg_depth = []
    min_depth = []

    for rec in logbook:
        if "fitness" in rec:
            avg = rec["fitness"].get("avg", None)
            mn = rec["fitness"].get("min", None)
            if avg is None or mn is None:
                continue
            avg_err.append(_get_2(avg, 0))
            min_err.append(_get_2(mn, 0))
            avg_depth.append(_get_2(avg, 1))
            min_depth.append(_get_2(mn, 1))
        else:
            # formato alternativo: rec['avg'] y rec['min'] ya contienen 2 objetivos
            if "avg" not in rec or "min" not in rec:
                continue
            avg = rec["avg"]; mn = rec["min"]
            avg_err.append(_get_2(avg, 0))
            min_err.append(_get_2(mn, 0))
            avg_depth.append(_get_2(avg, 1))
            min_depth.append(_get_2(mn, 1))

    # Ajustar gens al número real de puntos que hemos podido extraer
    gens = gens[:len(avg_err)]

    if len(gens) == 0:
        print("Aviso: el logbook no contiene estadísticas multi-objetivo ('fitness' ni ('avg','min')).")
        return

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


def plot_error_vs_depth(population, best=None, title="Error vs profundidad (población final)"):
    """Scatter para mono-objetivo (y también útil en multi)."""
    if not population:
        return

    xs = [ind.height for ind in population]
    ys = [float(ind.fitness.values[0]) for ind in population]

    plt.figure()
    plt.scatter(xs, ys, alpha=0.7, label="Individuos")
    if best is not None:
        plt.scatter([best.height], [float(best.fitness.values[0])], marker="*", s=200, label="Mejor")
    plt.title(title)
    plt.xlabel("Profundidad (height)")
    plt.ylabel("Error (1 - accuracy)")
    plt.grid(True, alpha=0.3)
    plt.legend()


def plot_mono_fitness_evolution(logbook, title="Evolución del fitness (mono-objetivo)"):
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



# -----------------------------
#  GP setup
# -----------------------------
def build_pset(n_features: int):
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
    # DEAP no permite redefinir creators: usamos nombres distintos
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


def run_gp(multiobj: bool):
    random.seed(SEED)
    np.random.seed(SEED)

    X, y = load_cancer_xlsx(INPUTS_XLSX, TARGETS_XLSX)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_ratio=TEST_RATIO, seed=SEED)

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

    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT))

    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_NODES))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_NODES))

    # Selección
    if multiobj:
        toolbox.register("select", tools.selNSGA2)
        select_parents = lambda pop: tools.selTournamentDCD(pop, len(pop))
    else:
        toolbox.register("select", tools.selTournament, tournsize=TOURN)
        select_parents = None

    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Evolución
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (mstats.fields if mstats else [])

    if multiobj:
        # --- 1) Evaluar población inicial ---
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, f in zip(invalid, fits):
            ind.fitness.values = f

        # --- 2) Asignar rank/crowding (NSGA-II) ---
        pop = toolbox.select(pop, len(pop))
        hof.update(pop)

        # --- 3) Estadísticas gen 0 ---
        record = mstats.compile(pop) if mstats else {}
        logbook.record(gen=0, nevals=len(invalid), **record)
        print(logbook.stream)

        for gen in range(1, NGEN_MULTI + 1):
            # --- 4) Selección de padres (DCD) ---
            k = POP if POP % 2 == 0 else POP - 1
            parents = tools.selTournamentDCD(pop, k)
            if POP % 2 == 1:
                parents.append(random.choice(pop))
            offspring = [toolbox.clone(ind) for ind in parents]

            # --- 5) Cruce / mutación ---
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values

            for mut in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mut)
                    del mut.fitness.values

            # --- 6) Evaluar descendencia ---
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid)
            for ind, f in zip(invalid, fits):
                ind.fitness.values = f

            # --- 7) Reemplazo NSGA-II ---
            pop = toolbox.select(pop + offspring, POP)
            hof.update(pop)

            # --- 8) Estadísticas ---
            record = mstats.compile(pop) if mstats else {}
            logbook.record(gen=gen, nevals=len(invalid), **record)
            print(logbook.stream)

    else:

        stop = StagnationStop(patience=PATIENCE, eps=EPS)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)

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

    # Test binario directo
    try:
        func = toolbox.compile(expr=best)
    except (SyntaxError, RecursionError, MemoryError):
        print("El mejor individuo no compila (bloat). Prueba bajando MAX_HEIGHT/MAX_NODES.")
        return
    preds = []
    for row in Xte:
        s = func(*row)
        preds.append(1 if s >= 0 else 0)
    preds = np.array(preds, dtype=int)

    acc = float(np.mean(preds == yte))
    print("\n=== Test ===")
    print("Accuracy:", acc)

    # -----------------------------
    #  Gráficas solicitadas
    # -----------------------------
    if multiobj:
        plot_pareto_front(pop, title="Población final y frente de Pareto (multiobjetivo)")
        plot_multiobj_evolution(logbook, title="Evolución de error y profundidad (multiobjetivo)")
    else:
        # Informe: error vs profundidad (población final)
        plot_error_vs_depth(pop, best=best, title="Error vs profundidad (mono-objetivo)")
        # (Opcional pero útil) evolución del error
        plot_mono_fitness_evolution(logbook, title="Evolución del fitness (mono-objetivo)")

    plt.show()


# -----------------------------
#  Menú
# -----------------------------
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


if __name__ == "__main__":
    while True:
        op = menu()

        if op == "1":
            run_gp(multiobj=False)
        elif op == "2":
            run_gp(multiobj=True)
        elif op == "3":
            print("Saliendo del programa.")
            break
