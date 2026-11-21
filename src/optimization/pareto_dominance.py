from typing import Any, Iterable, List, Sequence, Tuple

from config.instrument_settings import DEFAULT_SETTINGS

CostVector = Tuple[float, float, float]

DEFAULT_EPSILON = DEFAULT_SETTINGS.solver.exact_eps
DEFAULT_MERGE_EPSILON = DEFAULT_SETTINGS.solver.merge_eps


class _CostTuple(tuple):
    def __new__(cls, values: Tuple[float, float, float]):
        return super().__new__(cls, values)

    def __add__(self, other: Any) -> "_CostTuple":
        if isinstance(other, (int, float)):
            return _CostTuple((self[0] + other, self[1] + other, self[2] + other))
        other_t = tuple(other)
        return _CostTuple((self[0] + other_t[0], self[1] + other_t[1], self[2] + other_t[2]))

    def __sub__(self, other: Any) -> "_CostTuple":
        if isinstance(other, (int, float)):
            return _CostTuple((self[0] - other, self[1] - other, self[2] - other))
        other_t = tuple(other)
        return _CostTuple((self[0] - other_t[0], self[1] - other_t[1], self[2] - other_t[2]))

    def __ge__(self, other: Any) -> Tuple[bool, ...]:
        if isinstance(other, (int, float)):
            return (self[0] >= other, self[1] >= other, self[2] >= other)
        other_t = tuple(other)
        return (self[0] >= other_t[0], self[1] >= other_t[1], self[2] >= other_t[2])

    def __le__(self, other: Any) -> Tuple[bool, ...]:
        if isinstance(other, (int, float)):
            return (self[0] <= other, self[1] <= other, self[2] <= other)
        other_t = tuple(other)
        return (self[0] <= other_t[0], self[1] <= other_t[1], self[2] <= other_t[2])


def to_cost_tuple(vec: Any) -> CostVector:
    if isinstance(vec, _CostTuple):
        return vec
    values = tuple(float(v) for v in vec)
    if len(values) != 3:
        raise ValueError("El vector de costo debe tener tres componentes.")
    return _CostTuple((values[0], values[1], values[2]))


def dominates(
    vec_a: Any,
    vec_b: Any,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    """Retorna True si vec_a domina en sentido de Pareto a vec_b."""
    a = to_cost_tuple(vec_a)
    b = to_cost_tuple(vec_b)
    return (
        a[0] <= b[0] + eps
        and a[1] <= b[1] + eps
        and a[2] <= b[2] + eps
        and (a[0] < b[0] - eps or a[1] < b[1] - eps or a[2] < b[2] - eps)
    )


def weakly_dominates(
    vec_a: Any,
    vec_b: Any,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    a = to_cost_tuple(vec_a)
    b = to_cost_tuple(vec_b)
    return (
        a[0] <= b[0] + eps
        and a[1] <= b[1] + eps
        and a[2] <= b[2] + eps
    )


def is_dominated_by_set(
    candidate: Any,
    front: Iterable[Any],
    allow_equal: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> bool:
    cand = to_cost_tuple(candidate)
    if allow_equal:
        return any(
            existing[0] <= cand[0] + eps
            and existing[1] <= cand[1] + eps
            and existing[2] <= cand[2] + eps
            for existing in front
        )
    return any(dominates(existing, cand, eps=eps) for existing in front)


def filter_non_dominated(
    vectors: Sequence[Any],
    eps: float = DEFAULT_EPSILON,
) -> List[CostVector]:
    frontier: List[CostVector] = []
    for candidate in vectors:
        candidate_tuple = to_cost_tuple(candidate)
        if is_dominated_by_set(candidate_tuple, frontier, allow_equal=True, eps=eps):
            continue
        frontier = [
            existing
            for existing in frontier
            if not dominates(candidate_tuple, existing, eps=eps)
        ]
        frontier.append(candidate_tuple)
    return frontier


def update_pareto_front(
    front: List[Any],
    candidate: Any,
    eps: float = DEFAULT_EPSILON,
) -> Tuple[bool, List[CostVector]]:
    candidate_tuple = to_cost_tuple(candidate)
    if is_dominated_by_set(candidate_tuple, front, allow_equal=True, eps=eps):
        return False, [to_cost_tuple(existing) for existing in front]

    pruned = [
        to_cost_tuple(existing)
        for existing in front
        if not dominates(candidate_tuple, existing, eps=eps)
    ]
    pruned.append(candidate_tuple)
    return True, pruned


def lexicographic_key(
    vec: Any,
    precision: int = 9,
) -> Tuple[float, ...]:
    return (
        round(float(vec[0]), precision),
        round(float(vec[1]), precision),
        round(float(vec[2]), precision),
    )