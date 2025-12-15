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
    def _safe_stat(func, x):
        # x puede ser lista de escalares o tuplas (p.ej. fitness multi-objetivo)
        if len(x) == 0:
            return float("nan")
        arr = np.asarray(list(x), dtype=float)
        return func(arr, axis=0)

    mstats.register("avg", lambda x: _safe_stat(np.mean, x))
    mstats.register("std", lambda x: _safe_stat(np.std, x))
    mstats.register("min", lambda x: _safe_stat(np.min, x))
    mstats.register("max", lambda x: _safe_stat(np.max, x))

    # Evolución
    if multiobj:
        # Requisitos de selTournamentDCD: si k == len(pop), k debe ser múltiplo de 4
        if POP % 2 == 1:
            raise ValueError("POP debe ser par para este modo multi-objetivo (NSGA-II).")
        if POP % 4 != 0:
            raise ValueError("POP debe ser múltiplo de 4 para selTournamentDCD (p.ej. 200, 300, 400...).")

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + mstats.fields

        # --- 1) Evaluar población inicial ---
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for ind, f in zip(invalid, fits):
            ind.fitness.values = f

        # --- 2) Asignar rank/crowding (NSGA-II) ---
        pop = toolbox.select(pop, len(pop))
        hof.update(pop)

        record = mstats.compile(pop) if len(pop) else {}
        logbook.record(gen=0, nevals=len(invalid), **record)
        print(logbook.stream)

        # --- 3) Bucle evolutivo ---
        for gen in range(1, NGEN_MULTI + 1):
            # Selección de padres (DCD)
            parents = tools.selTournamentDCD(pop, len(pop))
            parents = list(map(toolbox.clone, parents))

            # Variación (cruce + mutación)
            offspring = algorithms.varAnd(parents, toolbox, cxpb=CXPB, mutpb=MUTPB)

            # Evaluar descendencia
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = toolbox.map(toolbox.evaluate, invalid)
            for ind, f in zip(invalid, fits):
                ind.fitness.values = f

            # Reemplazo NSGA-II (mu+lambda)
            pop = toolbox.select(pop + offspring, POP)
            hof.update(pop)

            record = mstats.compile(pop) if len(pop) else {}
            logbook.record(gen=gen, nevals=len(invalid), **record)
            print(logbook.stream)



    else:
        stop = StagnationStop(patience=PATIENCE, eps=EPS)
        pop, _ = eaSimple_stagnation(
            pop, toolbox,
            cxpb=CXPB, mutpb=MUTPB,
            stop=stop,
            stats=mstats, halloffame=hof, verbose=True
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
        try:
            s = func(*row)
        except Exception:
            s = 0.0
        preds.append(1 if s >= 0 else 0)
    preds = np.array(preds, dtype=int)

    acc = float(np.mean(preds == yte))
    print("\n=== Test ===")
    print("Accuracy:", acc)


# -----------------------------
#  Menú
# -----------------------------
def menu():
    print("\nPráctica 6 - Programación Genética (Breast Cancer)")
    print(f"Dataset fijo: {INPUTS_XLSX} + {TARGETS_XLSX}")
    print("\nElige modo:")
    print("  1) Mono-objetivo")
    print("  2) Multi-objetivo")
    print("  0) Salir")

    while True:
        op = input("\nOpción: ").strip()
        if op in {"0", "1", "2"}:
            return op
        print("Opción inválida.")


if __name__ == "__main__":
    op = menu()
    if op == "0":
        raise SystemExit
    elif op == "1":
        run_gp(multiobj=False)
    elif op == "2":
        run_gp(multiobj=True)
