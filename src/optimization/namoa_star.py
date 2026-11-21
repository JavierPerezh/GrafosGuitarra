from dataclasses import dataclass, field
import heapq
import math
from typing import Dict, List, Optional, Tuple

from config.instrument_settings import DEFAULT_SETTINGS
from src.graph.base_dag import PostureNode
from src.graph.dual_graph import CompoundGraph
from src.optimization.pareto_dominance import (
    CostVector,
    DEFAULT_EPSILON,
    DEFAULT_MERGE_EPSILON,
    _CostTuple,
    dominates,
    is_dominated_by_set,
    lexicographic_key,
    to_cost_tuple,
    update_pareto_front,
)


def _add(a: CostVector, b: CostVector) -> CostVector:
    return _CostTuple((a[0] + b[0], a[1] + b[1], a[2] + b[2]))


def _componentwise_min(a: CostVector, b: CostVector) -> CostVector:
    return _CostTuple((min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2])))


def _extract_cnode_posture(cnode: object) -> Optional[PostureNode]:
    for attr in ("curr_node", "to_node", "target_node", "v", "second", "posture"):
        val = getattr(cnode, attr, None)
        if isinstance(val, PostureNode):
            return val
    return None


def _open_bridge_cost(
    last_fretted: Optional[PostureNode],
    curr_posture: Optional[PostureNode],
    next_posture: Optional[PostureNode],
) -> CostVector:
    """Evalúa desplazamiento y continuidad melódica cuando hubo cuerdas al aire intermedias."""
    if (
        last_fretted is None
        or curr_posture is None
        or next_posture is None
        or curr_posture.hand_position != 0
        or next_posture.hand_position == 0
    ):
        return _CostTuple((0.0, 0.0, 0.0))

    shift = abs(next_posture.hand_position - last_fretted.hand_position)
    bio_extra = 1.5 * (shift ** 1.6) if shift > 0 else 0.0

    prev_fretted = [a for a in last_fretted.assignments if not a.is_open]
    next_fretted = [a for a in next_posture.assignments if not a.is_open]
    timbre_extra = 0.0
    fluency_extra = 0.0

    if len(prev_fretted) == 1 and len(next_fretted) == 1:
        s_diff = abs(next_fretted[0].string - prev_fretted[0].string)
        f_diff = abs(next_fretted[0].fret - prev_fretted[0].fret)
        if s_diff > 0 and f_diff >= 3:
            bio_extra += 0.85 * f_diff
            timbre_extra += 1.65 * s_diff + 0.25 * f_diff
        if shift > 0 or s_diff > 0:
            fluency_extra += 0.45

    return _CostTuple((bio_extra, timbre_extra, fluency_extra))


@dataclass(order=True)
class _QueueEntry:
    f_key: Tuple[float, ...]
    label_id: int
    label: "SearchLabel" = field(compare=False)


@dataclass
class SearchLabel:
    label_id: int
    node_id: str
    g_cost: CostVector
    f_cost: CostVector
    parent: Optional["SearchLabel"] = None
    active: bool = True
    last_fretted_node: Optional[PostureNode] = None


@dataclass(frozen=True)
class ParetoSolution:
    solution_id: int
    total_cost: CostVector
    compound_path: Tuple[str, ...]
    posture_sequence: Tuple[PostureNode, ...]


@dataclass(frozen=True)
class NamoaStarResult:
    solutions: Tuple[ParetoSolution, ...]
    expanded_labels: int
    generated_labels: int
    pruned_labels: int


def _tab_geometry_signature(
    solution: ParetoSolution,
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return tuple(
        tuple((a.string, a.fret) for a in posture.assignments)
        for posture in solution.posture_sequence
    )


def deduplicate_by_tab_geometry(
    solutions: Tuple[ParetoSolution, ...] | List[ParetoSolution],
) -> Tuple[ParetoSolution, ...]:
    """Colapsa soluciones con idéntica tablatura (cuerda, traste) conservando la de menor esfuerzo."""
    if len(solutions) <= 1:
        return tuple(solutions)

    best_by_tab: Dict[Tuple[Tuple[Tuple[int, int], ...], ...], ParetoSolution] = {}
    for sol in solutions:
        sig = _tab_geometry_signature(sol)
        existing = best_by_tab.get(sig)
        if existing is None:
            best_by_tab[sig] = sol
            continue
        curr_rank = (sol.total_cost[0], sol.total_cost[2], sol.total_cost[1])
        prev_rank = (existing.total_cost[0], existing.total_cost[2], existing.total_cost[1])
        if curr_rank < prev_rank:
            best_by_tab[sig] = sol

    deduped = sorted(best_by_tab.values(), key=lambda s: lexicographic_key(s.total_cost))
    return tuple(deduped)


def merge_near_duplicate_solutions(
    solutions: Tuple[ParetoSolution, ...] | List[ParetoSolution],
    merge_eps: float = DEFAULT_MERGE_EPSILON,
) -> Tuple[ParetoSolution, ...]:
    """Agrupa soluciones casi idénticas preservando siempre los 3 extremos del frente."""
    if len(solutions) <= 1 or merge_eps <= 0.0:
        return tuple(solutions)

    unique = list(solutions)
    extremal_ids = {
        min(unique, key=lambda s: (s.total_cost[0], s.total_cost[1], s.total_cost[2])).solution_id,
        min(unique, key=lambda s: (s.total_cost[1], s.total_cost[0], s.total_cost[2])).solution_id,
        min(unique, key=lambda s: (s.total_cost[2], s.total_cost[0], s.total_cost[1])).solution_id,
    }

    ordered = sorted(
        unique,
        key=lambda s: (0 if s.solution_id in extremal_ids else 1, lexicographic_key(s.total_cost)),
    )
    kept: List[ParetoSolution] = []
    for candidate in ordered:
        if candidate.solution_id in extremal_ids:
            kept.append(candidate)
            continue
        if is_dominated_by_set(
            candidate.total_cost,
            (sol.total_cost for sol in kept),
            allow_equal=True,
            eps=merge_eps,
        ):
            continue
        kept.append(candidate)

    kept.sort(key=lambda s: lexicographic_key(s.total_cost))
    return tuple(kept)


def select_diverse_solutions(
    solutions: Tuple[ParetoSolution, ...] | List[ParetoSolution],
    k: Optional[int],
) -> Tuple[ParetoSolution, ...]:
    if not solutions:
        return ()
    if k is not None and k <= 0:
        return ()

    unique = list(solutions)
    if len(unique) == 1:
        return (unique[0],)

    extremal = [
        min(unique, key=lambda sol: (sol.total_cost[0], sol.total_cost[1], sol.total_cost[2])),
        min(unique, key=lambda sol: (sol.total_cost[1], sol.total_cost[0], sol.total_cost[2])),
        min(unique, key=lambda sol: (sol.total_cost[2], sol.total_cost[0], sol.total_cost[1])),
    ]

    values = [sol.total_cost for sol in unique]
    mins = tuple(min(v[d] for v in values) for d in range(3))
    maxs = tuple(max(v[d] for v in values) for d in range(3))
    spans = tuple(max(maxs[d] - mins[d], 1e-9) for d in range(3))

    def normalized_distance(sol: ParetoSolution) -> float:
        normalized = tuple(
            (sol.total_cost[d] - mins[d]) / spans[d]
            for d in range(3)
        )
        return math.sqrt(sum(v * v for v in normalized))

    ordered_knees = sorted(
        unique,
        key=lambda sol: (normalized_distance(sol), lexicographic_key(sol.total_cost)),
    )

    if k is not None and k < 3:
        priority_pool = [ordered_knees[0], *extremal, *ordered_knees[1:]]
        selected: List[ParetoSolution] = []
        selected_ids = set()
        for sol in priority_pool:
            if len(selected) >= k:
                break
            if sol.solution_id not in selected_ids:
                selected.append(sol)
                selected_ids.add(sol.solution_id)
        return tuple(sorted(selected, key=lambda sol: lexicographic_key(sol.total_cost)))

    selected = []
    selected_ids = set()
    for sol in extremal:
        if sol.solution_id not in selected_ids:
            selected.append(sol)
            selected_ids.add(sol.solution_id)

    for sol in ordered_knees:
        if k is not None and len(selected) >= k:
            break
        if sol.solution_id not in selected_ids:
            selected.append(sol)
            selected_ids.add(sol.solution_id)

    return tuple(sorted(selected, key=lambda sol: lexicographic_key(sol.total_cost)))


class NamoaStarSolver:
    def __init__(
        self,
        eps: float = DEFAULT_EPSILON,
        merge_eps: float = DEFAULT_MERGE_EPSILON,
        max_solutions: Optional[int] = DEFAULT_SETTINGS.solver.max_solutions,
    ) -> None:
        self.eps = eps
        self.merge_eps = merge_eps
        self.max_solutions = max_solutions

    def compute_backward_heuristic(
        self, graph: CompoundGraph
    ) -> Dict[str, CostVector]:
        """Calcula h(n) admisible por componente mediante programación dinámica inversa."""
        inf = float("inf")
        inf_tuple = _CostTuple((inf, inf, inf))
        heuristic: Dict[str, CostVector] = {
            node_id: inf_tuple for node_id in graph.nodes_by_id
        }
        heuristic[graph.source_id] = inf_tuple
        heuristic[graph.sink_id] = _CostTuple((0.0, 0.0, 0.0))

        for edge in graph.reverse_adjacency.get(graph.sink_id, ()):
            cost = to_cost_tuple(edge.cost_vector)
            heuristic[edge.source_id] = _componentwise_min(
                heuristic[edge.source_id], cost
            )

        for layer in reversed(graph.layers):
            for cnode in layer:
                h_curr = heuristic[cnode.node_id]
                if not all(math.isfinite(v) for v in h_curr):
                    continue
                for edge in graph.reverse_adjacency.get(cnode.node_id, ()):
                    cost = to_cost_tuple(edge.cost_vector)
                    candidate = _add(cost, h_curr)
                    heuristic[edge.source_id] = _componentwise_min(
                        heuristic[edge.source_id], candidate
                    )

        return heuristic

    def solve(
        self,
        graph: CompoundGraph,
        initial_fretted_posture: Optional[PostureNode] = None,
    ) -> NamoaStarResult:
        heuristic = self.compute_backward_heuristic(graph)
        if not all(math.isfinite(v) for v in heuristic[graph.source_id]):
            return NamoaStarResult((), 0, 0, 0)

        finite_heuristic: Dict[str, CostVector] = {
            nid: h_vec
            for nid, h_vec in heuristic.items()
            if all(math.isfinite(v) for v in h_vec)
        }
        posture_by_cnode: Dict[str, Optional[PostureNode]] = {
            nid: _extract_cnode_posture(cnode)
            for nid, cnode in graph.nodes_by_id.items()
        }
        edge_costs: Dict[str, Tuple[Tuple[str, CostVector], ...]] = {
            nid: tuple(
                (edge.target_id, to_cost_tuple(edge.cost_vector))
                for edge in edges
                if edge.target_id in finite_heuristic
            )
            for nid, edges in graph.adjacency.items()
        }

        g_open: Dict[str, List[SearchLabel]] = {nid: [] for nid in finite_heuristic}
        g_closed: Dict[str, List[CostVector]] = {nid: [] for nid in finite_heuristic}

        open_heap: List[_QueueEntry] = []
        label_counter = 0
        expanded = 0
        generated = 1
        pruned = 0

        start_g: CostVector = _CostTuple((0.0, 0.0, 0.0))
        start_f = finite_heuristic[graph.source_id]
        start_label = SearchLabel(
            label_counter,
            graph.source_id,
            start_g,
            start_f,
            None,
            True,
            initial_fretted_posture,
        )
        g_open[graph.source_id].append(start_label)
        heapq.heappush(
            open_heap,
            _QueueEntry(lexicographic_key(start_f), label_counter, start_label),
        )

        goal_front: List[CostVector] = []
        goal_labels: List[SearchLabel] = []
        eps = self.eps
        sink_id = graph.sink_id

        while open_heap:
            entry = heapq.heappop(open_heap)
            curr = entry.label
            if not curr.active:
                continue

            curr.active = False
            nid = curr.node_id
            g_open[nid] = [lbl for lbl in g_open[nid] if lbl.active]

            if is_dominated_by_set(
                curr.f_cost, goal_front, allow_equal=False, eps=eps
            ):
                pruned += 1
                continue

            if nid == sink_id:
                accepted, updated_front = update_pareto_front(
                    goal_front, curr.g_cost, eps=eps
                )
                if accepted:
                    goal_front = updated_front
                    goal_labels = [
                        gl for gl in goal_labels
                        if not dominates(curr.g_cost, gl.g_cost, eps=eps)
                    ]
                    goal_labels.append(curr)
                    g_closed[nid] = list(goal_front)
                else:
                    pruned += 1
                continue

            g_closed[nid].append(curr.g_cost)
            expanded += 1
            curr_posture = posture_by_cnode.get(nid)

            for next_id, edge_cost in edge_costs.get(nid, ()):
                h_next = finite_heuristic[next_id]
                next_posture = posture_by_cnode.get(next_id)
                bridge_cost = _open_bridge_cost(
                    curr.last_fretted_node, curr_posture, next_posture
                )
                new_g = _add(_add(curr.g_cost, edge_cost), bridge_cost)
                new_f = _add(new_g, h_next)

                open_vectors = [lbl.g_cost for lbl in g_open[next_id] if lbl.active]
                if is_dominated_by_set(
                    new_g, open_vectors, allow_equal=True, eps=eps
                ) or is_dominated_by_set(
                    new_g, g_closed[next_id], allow_equal=True, eps=eps
                ):
                    pruned += 1
                    continue

                if is_dominated_by_set(
                    new_f, goal_front, allow_equal=False, eps=eps
                ):
                    pruned += 1
                    continue

                for lbl in g_open[next_id]:
                    if lbl.active and dominates(new_g, lbl.g_cost, eps=eps):
                        lbl.active = False

                g_closed[next_id] = [
                    closed_vec
                    for closed_vec in g_closed[next_id]
                    if not dominates(new_g, closed_vec, eps=eps)
                ]

                next_last_fretted = (
                    next_posture
                    if (next_posture is not None and next_posture.hand_position > 0)
                    else curr.last_fretted_node
                )

                label_counter += 1
                generated += 1
                next_label = SearchLabel(
                    label_counter,
                    next_id,
                    new_g,
                    new_f,
                    curr,
                    True,
                    next_last_fretted,
                )
                g_open[next_id].append(next_label)
                heapq.heappush(
                    open_heap,
                    _QueueEntry(lexicographic_key(new_f), label_counter, next_label),
                )

        goal_labels.sort(key=lambda lbl: lexicographic_key(lbl.g_cost))
        raw_solutions: List[ParetoSolution] = []
        for idx, gl in enumerate(goal_labels):
            compound_path = self._reconstruct_compound_path(gl)
            postures = graph.extract_posture_path(compound_path)
            raw_solutions.append(
                ParetoSolution(idx + 1, gl.g_cost, compound_path, postures)
            )

        unique_tabs = deduplicate_by_tab_geometry(raw_solutions)
        num_events = len(unique_tabs[0].posture_sequence) if unique_tabs else 1
        adaptive_merge_eps = max(self.merge_eps, 0.025 * num_events)
        merged = merge_near_duplicate_solutions(unique_tabs, adaptive_merge_eps)
        effective_k = (
            self.max_solutions
            if self.max_solutions is not None
            else (DEFAULT_SETTINGS.solver.max_solutions if len(merged) > 10 else None)
        )
        selected = select_diverse_solutions(merged, effective_k)
        reindexed = tuple(
            ParetoSolution(idx + 1, sol.total_cost, sol.compound_path, sol.posture_sequence)
            for idx, sol in enumerate(selected)
        )
        return NamoaStarResult(reindexed, expanded, generated, pruned)

    @staticmethod
    def _reconstruct_compound_path(goal_label: SearchLabel) -> Tuple[str, ...]:
        path: List[str] = []
        curr: Optional[SearchLabel] = goal_label
        while curr is not None:
            path.append(curr.node_id)
            curr = curr.parent
        path.reverse()
        return tuple(path)
