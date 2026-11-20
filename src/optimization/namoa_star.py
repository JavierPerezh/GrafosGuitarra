from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.graph.base_dag import PostureNode
from src.graph.cost_vector import CostVector, zero_cost_vector
from src.graph.dual_graph import CompoundGraph
from src.optimization.pareto_dominance import (
    DEFAULT_EPSILON,
    dominates,
    is_dominated_by_set,
    lexicographic_key,
    update_pareto_front,
)


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


class NamoaStarSolver:
    def __init__(
        self,
        eps: float = DEFAULT_EPSILON,
        max_solutions: Optional[int] = None,
    ) -> None:
        self.eps = eps
        self.max_solutions = max_solutions

    def compute_backward_heuristic(
        self, graph: CompoundGraph
    ) -> Dict[str, CostVector]:
        """Calcula h(n) admisible por componente mediante programación dinámica inversa."""
        inf_vec = np.full(3, np.inf, dtype=np.float64)
        heuristic: Dict[str, CostVector] = {
            node_id: inf_vec.copy() for node_id in graph.nodes_by_id
        }
        heuristic[graph.source_id] = inf_vec.copy()
        heuristic[graph.sink_id] = zero_cost_vector()

        for edge in graph.reverse_adjacency.get(graph.sink_id, ()):
            heuristic[edge.source_id] = np.minimum(
                heuristic[edge.source_id], edge.cost_vector
            )

        for layer in reversed(graph.layers):
            for cnode in layer:
                h_curr = heuristic[cnode.node_id]
                if not np.all(np.isfinite(h_curr)):
                    continue
                for edge in graph.reverse_adjacency.get(cnode.node_id, ()):
                    candidate = edge.cost_vector + h_curr
                    heuristic[edge.source_id] = np.minimum(
                        heuristic[edge.source_id], candidate
                    )

        return heuristic

    def solve(self, graph: CompoundGraph) -> NamoaStarResult:
        heuristic = self.compute_backward_heuristic(graph)
        if not np.all(np.isfinite(heuristic[graph.source_id])):
            return NamoaStarResult(
                solutions=(),
                expanded_labels=0,
                generated_labels=0,
                pruned_labels=0,
            )

        g_open: Dict[str, List[CostVector]] = {
            nid: [] for nid in graph.nodes_by_id
        }
        g_open[graph.source_id] = []
        g_open[graph.sink_id] = []

        g_closed: Dict[str, List[CostVector]] = {
            nid: [] for nid in graph.nodes_by_id
        }
        g_closed[graph.source_id] = []
        g_closed[graph.sink_id] = []

        open_heap: List[_QueueEntry] = []
        label_counter = 0
        expanded = 0
        generated = 1
        pruned = 0

        start_g = zero_cost_vector()
        start_f = start_g + heuristic[graph.source_id]
        start_label = SearchLabel(
            label_id=label_counter,
            node_id=graph.source_id,
            g_cost=start_g,
            f_cost=start_f,
            parent=None,
        )
        g_open[graph.source_id].append(start_g)
        heapq.heappush(
            open_heap,
            _QueueEntry(
                f_key=lexicographic_key(start_f),
                label_id=label_counter,
                label=start_label,
            ),
        )

        goal_front: List[CostVector] = []
        goal_labels: List[SearchLabel] = []

        while open_heap:
            entry = heapq.heappop(open_heap)
            curr = entry.label
            nid = curr.node_id

            if not self._contains_vector(g_open[nid], curr.g_cost):
                pruned += 1
                continue

            self._remove_vector(g_open[nid], curr.g_cost)

            if is_dominated_by_set(
                curr.f_cost, goal_front, allow_equal=False, eps=self.eps
            ):
                pruned += 1
                continue

            if nid == graph.sink_id:
                accepted, updated_front = update_pareto_front(
                    goal_front, curr.g_cost, eps=self.eps
                )
                if accepted:
                    goal_front = updated_front
                    goal_labels = [
                        gl
                        for gl in goal_labels
                        if not dominates(curr.g_cost, gl.g_cost, eps=self.eps)
                    ]
                    goal_labels.append(curr)
                    g_closed[nid] = list(goal_front)
                    if (
                        self.max_solutions is not None
                        and len(goal_labels) >= self.max_solutions
                    ):
                        break
                else:
                    pruned += 1
                continue

            g_closed[nid].append(curr.g_cost)
            expanded += 1

            for edge in graph.adjacency.get(nid, ()):
                next_id = edge.target_id
                h_next = heuristic.get(next_id)
                if h_next is None or not np.all(np.isfinite(h_next)):
                    pruned += 1
                    continue

                new_g = curr.g_cost + edge.cost_vector
                new_f = new_g + h_next

                # Early Dominance Checking contra estados abiertos, cerrados y meta
                if is_dominated_by_set(
                    new_g, g_open[next_id], allow_equal=True, eps=self.eps
                ) or is_dominated_by_set(
                    new_g, g_closed[next_id], allow_equal=True, eps=self.eps
                ):
                    pruned += 1
                    continue

                if is_dominated_by_set(
                    new_f, goal_front, allow_equal=False, eps=self.eps
                ):
                    pruned += 1
                    continue

                g_open[next_id] = [
                    v
                    for v in g_open[next_id]
                    if not dominates(new_g, v, eps=self.eps)
                ]
                g_closed[next_id] = [
                    v
                    for v in g_closed[next_id]
                    if not dominates(new_g, v, eps=self.eps)
                ]

                g_open[next_id].append(new_g)
                label_counter += 1
                generated += 1

                next_label = SearchLabel(
                    label_id=label_counter,
                    node_id=next_id,
                    g_cost=new_g,
                    f_cost=new_f,
                    parent=curr,
                )
                heapq.heappush(
                    open_heap,
                    _QueueEntry(
                        f_key=lexicographic_key(new_f),
                        label_id=label_counter,
                        label=next_label,
                    ),
                )

        goal_labels.sort(key=lambda lbl: lexicographic_key(lbl.g_cost))
        solutions: List[ParetoSolution] = []
        for idx, gl in enumerate(goal_labels):
            c_path = self._reconstruct_compound_path(gl)
            postures = graph.extract_posture_path(c_path)
            solutions.append(
                ParetoSolution(
                    solution_id=idx + 1,
                    total_cost=gl.g_cost,
                    compound_path=c_path,
                    posture_sequence=postures,
                )
            )

        return NamoaStarResult(
            solutions=tuple(solutions),
            expanded_labels=expanded,
            generated_labels=generated,
            pruned_labels=pruned,
        )

    def _contains_vector(
        self, vec_list: List[CostVector], target: CostVector
    ) -> bool:
        return any(np.allclose(v, target, atol=self.eps, rtol=0.0) for v in vec_list)

    def _remove_vector(
        self, vec_list: List[CostVector], target: CostVector
    ) -> None:
        for idx, v in enumerate(vec_list):
            if np.allclose(v, target, atol=self.eps, rtol=0.0):
                vec_list.pop(idx)
                return

    @staticmethod
    def _reconstruct_compound_path(goal_label: SearchLabel) -> Tuple[str, ...]:
        path: List[str] = []
        curr: Optional[SearchLabel] = goal_label
        while curr is not None:
            path.append(curr.node_id)
            curr = curr.parent
        path.reverse()
        return tuple(path)