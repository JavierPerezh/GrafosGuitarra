from typing import Iterable, List, Sequence, Tuple
import numpy as np

from src.graph.cost_vector import CostVector

DEFAULT_EPSILON = 1e-9


def dominates(
    vec_a: CostVector,
    vec_b: CostVector,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    """Retorna True si vec_a domina estrictamente en Pareto a vec_b (vec_a < vec_b)."""
    not_worse = bool(np.all(vec_a <= vec_b + eps))
    strictly_better = bool(np.any(vec_a < vec_b - eps))
    return not_worse and strictly_better


def weakly_dominates(
    vec_a: CostVector,
    vec_b: CostVector,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    """Retorna True si vec_a es mejor o igual que vec_b en todas las dimensiones."""
    return bool(np.all(vec_a <= vec_b + eps))


def is_dominated_by_set(
    candidate: CostVector,
    front: Iterable[CostVector],
    allow_equal: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    """Chequeo temprano de dominancia contra un conjunto de vectores existentes."""
    check_fn = weakly_dominates if allow_equal else dominates
    return any(check_fn(existing, candidate, eps=eps) for existing in front)


def filter_non_dominated(
    vectors: Sequence[CostVector],
    eps: float = DEFAULT_EPSILON,
) -> List[CostVector]:
    """Extrae el subconjunto de vectores no dominados (Frontera de Pareto)."""
    frontier: List[CostVector] = []
    for candidate in vectors:
        if is_dominated_by_set(candidate, frontier, allow_equal=True, eps=eps):
            continue
        frontier = [
            existing
            for existing in frontier
            if not dominates(candidate, existing, eps=eps)
        ]
        frontier.append(candidate)
    return frontier


def update_pareto_front(
    front: List[CostVector],
    candidate: CostVector,
    eps: float = DEFAULT_EPSILON,
) -> Tuple[bool, List[CostVector]]:
    """Inserta candidate si no es dominado y poda vectores ahora dominados."""
    if is_dominated_by_set(candidate, front, allow_equal=True, eps=eps):
        return False, front

    pruned = [
        existing
        for existing in front
        if not dominates(candidate, existing, eps=eps)
    ]
    pruned.append(candidate)
    return True, pruned


def lexicographic_key(
    vec: CostVector,
    precision: int = 9,
) -> Tuple[float, ...]:
    """Clave lexicográfica determinista para ordenar etiquetas en OPEN."""
    return tuple(round(float(val), precision) for val in vec)