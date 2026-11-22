from dataclasses import dataclass, replace
import math
from typing import Dict, List, Optional, Tuple

from config.instrument_settings import DEFAULT_SETTINGS, InstrumentSettings
from src.domain.fretboard_map import FretboardMap
from src.domain.musical_event import MusicalEvent, MusicalSequence
from src.graph.base_dag import BaseDagBuilder, PostureNode
from src.graph.cost_vector import CostVectorEvaluator
from src.graph.dual_graph import DualGraphBuilder
from src.optimization.namoa_star import (
    NamoaStarResult,
    NamoaStarSolver,
    ParetoSolution,
    _open_bridge_cost,
    deduplicate_by_tab_geometry,
    select_diverse_solutions,
)
from src.optimization.pareto_dominance import (
    _CostTuple,
    dominates,
    lexicographic_key,
    to_cost_tuple,
)


@dataclass(frozen=True)
class MeasureSegment:
    segment_index: int
    start_bar: int
    end_bar: int
    sequence: MusicalSequence


@dataclass
class _ProfileTrack:
    profile: str
    total_cost: Tuple[float, float, float]
    compound_path: List[str]
    postures: List[PostureNode]
    last_posture: Optional[PostureNode]
    last_fretted_posture: Optional[PostureNode]


class MeasureAwareSegmenter:
    """Subdivide una secuencia musical respetando barras de compás y un máximo de N eventos."""

    def __init__(self, settings: InstrumentSettings = DEFAULT_SETTINGS) -> None:
        self.settings = settings

    def segment(self, sequence: MusicalSequence) -> List[MeasureSegment]:
        max_events = max(2, self.settings.solver.max_segment_events)
        min_rest = self.settings.solver.min_rest_ioi_sec

        if len(sequence.events) <= max_events:
            return [
                MeasureSegment(
                    segment_index=0,
                    start_bar=1,
                    end_bar=1,
                    sequence=sequence,
                )
            ]

        num, den = sequence.metadata.get(
            "time_signature", self.settings.solver.default_time_signature
        )
        sec_per_quarter = 60.0 / max(sequence.tempo_bpm, 1.0)
        beat_sec = sec_per_quarter * (4.0 / max(den, 1))
        bar_sec = max(beat_sec * max(num, 1), 0.1)

        bars_map: Dict[int, List[MusicalEvent]] = {}
        for ev in sequence.events:
            bar_idx = int((ev.onset_sec + 1e-4) // bar_sec)
            bars_map.setdefault(bar_idx, []).append(ev)

        atomic_chunks: List[Tuple[int, List[MusicalEvent]]] = []
        for bar_idx in sorted(bars_map):
            bar_events = bars_map[bar_idx]
            if len(bar_events) <= max_events:
                atomic_chunks.append((bar_idx, bar_events))
                continue

            sub_chunk: List[MusicalEvent] = []
            current_beat = int((bar_events[0].onset_sec + 1e-4) // beat_sec)
            for ev in bar_events:
                ev_beat = int((ev.onset_sec + 1e-4) // beat_sec)
                if (
                    sub_chunk
                    and len(sub_chunk) + 1 > max_events
                    and ev_beat != current_beat
                ):
                    atomic_chunks.append((bar_idx, sub_chunk))
                    sub_chunk = [ev]
                    current_beat = ev_beat
                else:
                    if len(sub_chunk) >= max_events:
                        atomic_chunks.append((bar_idx, sub_chunk))
                        sub_chunk = [ev]
                    else:
                        sub_chunk.append(ev)
                    current_beat = ev_beat
            if sub_chunk:
                atomic_chunks.append((bar_idx, sub_chunk))

        merged_groups: List[Tuple[int, int, List[MusicalEvent]]] = []
        curr_start_bar, curr_events = atomic_chunks[0]
        curr_end_bar = curr_start_bar

        for bar_idx, chunk_events in atomic_chunks[1:]:
            last_ev = curr_events[-1]
            has_phrase_rest = (
                last_ev.ioi_sec is not None and last_ev.ioi_sec >= min_rest
            )
            if (
                not has_phrase_rest
                and len(curr_events) + len(chunk_events) <= max_events
            ):
                curr_events.extend(chunk_events)
                curr_end_bar = bar_idx
            else:
                merged_groups.append((curr_start_bar, curr_end_bar, curr_events))
                curr_start_bar = bar_idx
                curr_end_bar = bar_idx
                curr_events = list(chunk_events)

        merged_groups.append((curr_start_bar, curr_end_bar, curr_events))

        segments: List[MeasureSegment] = []
        for seg_idx, (b_start, b_end, ev_list) in enumerate(merged_groups):
            base_onset = ev_list[0].onset_sec
            reindexed_events = tuple(
                replace(
                    ev,
                    index=i,
                    onset_sec=round(ev.onset_sec - base_onset, 6),
                )
                for i, ev in enumerate(ev_list)
            )
            sub_seq = MusicalSequence(
                events=reindexed_events,
                title=f"{sequence.title}_seg{seg_idx + 1}",
                tempo_bpm=sequence.tempo_bpm,
                metadata=dict(sequence.metadata),
            )
            if ev_list[-1].ioi_sec is not None:
                fixed_last = replace(sub_seq.events[-1], ioi_sec=ev_list[-1].ioi_sec)
                object.__setattr__(
                    sub_seq, "events", (*sub_seq.events[:-1], fixed_last)
                )

            segments.append(
                MeasureSegment(
                    segment_index=seg_idx,
                    start_bar=b_start + 1,
                    end_bar=b_end + 1,
                    sequence=sub_seq,
                )
            )

        return segments


class SegmentedNamoaSolver:
    """Orquesta NAMOA* por compases propagando múltiples perfiles de Pareto."""

    def __init__(
        self,
        settings: InstrumentSettings = DEFAULT_SETTINGS,
        max_solutions: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.max_solutions = (
            max_solutions
            if max_solutions is not None
            else settings.solver.max_solutions
        )
        self.fretboard = FretboardMap(settings.physics)
        self.segmenter = MeasureAwareSegmenter(settings)

    def solve(self, sequence: MusicalSequence) -> NamoaStarResult:
        segments = self.segmenter.segment(sequence)
        if len(segments) == 1:
            return self._solve_single_segment(
                segments[0].sequence,
                initial_posture=None,
                initial_fretted_posture=None,
                max_k=self.max_solutions,
            )

        total_expanded = 0
        total_generated = 0
        total_pruned = 0

        # Segmento 0: calcula el frente exacto inicial y arranca pistas por perfil
        first_res = self._solve_single_segment(
            segments[0].sequence,
            initial_posture=None,
            initial_fretted_posture=None,
            max_k=max(self.max_solutions or 5, 5),
        )
        total_expanded += first_res.expanded_labels
        total_generated += first_res.generated_labels
        total_pruned += first_res.pruned_labels

        if not first_res.solutions:
            return NamoaStarResult((), total_expanded, total_generated, total_pruned)

        target_tracks = min(max(self.max_solutions or 4, 4), 5)
        tracks = self._init_tracks(first_res.solutions)[:target_tracks]

        # Segmentos posteriores: encadena cada perfil respetando su criterio objetivo
        for seg in segments[1:]:
            for track in tracks:
                seg_res = self._solve_profile_segment(
                    seg.sequence,
                    profile=track.profile,
                    initial_posture=track.last_posture,
                    initial_fretted_posture=track.last_fretted_posture,
                )
                total_expanded += seg_res.expanded_labels
                total_generated += seg_res.generated_labels
                total_pruned += seg_res.pruned_labels

                if not seg_res.solutions:
                    continue

                chosen = seg_res.solutions[0]
                track.total_cost = (
                    track.total_cost[0] + chosen.total_cost[0],
                    track.total_cost[1] + chosen.total_cost[1],
                    track.total_cost[2] + chosen.total_cost[2],
                )
                track.compound_path.extend(
                    f"s{seg.segment_index}:{nid}" for nid in chosen.compound_path
                )
                track.postures.extend(chosen.posture_sequence)
                track.last_posture = chosen.posture_sequence[-1]
                for p in chosen.posture_sequence:
                    if p.hand_position > 0:
                        track.last_fretted_posture = p

        raw_global: List[ParetoSolution] = []
        for idx, tr in enumerate(tracks):
            if len(tr.postures) == len(sequence.events):
                raw_global.append(
                    ParetoSolution(
                        solution_id=idx + 1,
                        total_cost=_CostTuple(tr.total_cost),
                        compound_path=tuple(tr.compound_path),
                        posture_sequence=tuple(tr.postures),
                    )
                )

        deduped = deduplicate_by_tab_geometry(raw_global)
        non_dominated = self._filter_pareto_nondominated(deduped)
        selected = select_diverse_solutions(non_dominated, self.max_solutions)
        reindexed = tuple(
            ParetoSolution(
                solution_id=idx + 1,
                total_cost=sol.total_cost,
                compound_path=sol.compound_path,
                posture_sequence=sol.posture_sequence,
            )
            for idx, sol in enumerate(selected)
        )
        return NamoaStarResult(reindexed, total_expanded, total_generated, total_pruned)

    def _solve_single_segment(
        self,
        sub_sequence: MusicalSequence,
        initial_posture: Optional[PostureNode],
        initial_fretted_posture: Optional[PostureNode],
        max_k: Optional[int],
    ) -> NamoaStarResult:
        dag_builder = BaseDagBuilder(
            self.fretboard,
            self.settings.biomechanics,
            kinematic_pruning=True,
        )
        base_dag = dag_builder.build(sub_sequence)

        evaluator = CostVectorEvaluator(
            fretboard=self.fretboard,
            weights=self.settings.cost_weights,
            biomechanics=self.settings.biomechanics,
            initial_posture=initial_posture,
            initial_fretted_posture=initial_fretted_posture,
        )

        if len(sub_sequence.events) == 1:
            candidates = base_dag.layers[0]
            single_sols: List[ParetoSolution] = []
            for idx, node in enumerate(candidates):
                if initial_posture is not None:
                    c_vec = evaluator.evaluate_first_order(
                        initial_posture,
                        node,
                        sub_sequence.events[0],
                        sub_sequence.events[0],
                    )
                else:
                    bio, tim = evaluator._compute_initial_posture_cost(node)
                    c_vec = [bio, tim, 0.0]
                single_sols.append(
                    ParetoSolution(
                        solution_id=idx + 1,
                        total_cost=to_cost_tuple(c_vec),
                        compound_path=(node.node_id,),
                        posture_sequence=(node,),
                    )
                )
            deduped = deduplicate_by_tab_geometry(single_sols)
            non_dom = self._filter_pareto_nondominated(deduped)
            sel = select_diverse_solutions(non_dom, max_k)
            return NamoaStarResult(sel, len(candidates), len(candidates), 0)

        dual_builder = DualGraphBuilder(evaluator)
        dual_graph = dual_builder.build(base_dag, sub_sequence)
        solver = NamoaStarSolver(
            eps=self.settings.solver.exact_eps,
            merge_eps=self.settings.solver.merge_eps,
            max_solutions=max_k,
        )
        return solver.solve(dual_graph, initial_fretted_posture=initial_fretted_posture)

    @staticmethod
    def _init_tracks(solutions: Tuple[ParetoSolution, ...]) -> List[_ProfileTrack]:
        profiles = ["balanced", "ergonomic", "timbral", "fluency", "balanced_alt"]
        tracks: List[_ProfileTrack] = []
        used_ids = set()

        for prof in profiles:
            chosen = SegmentedNamoaSolver._pick_for_profile(solutions, prof)
            if chosen.solution_id in used_ids and len(solutions) > len(used_ids):
                for alt in solutions:
                    if alt.solution_id not in used_ids:
                        chosen = alt
                        break
            used_ids.add(chosen.solution_id)

            last_fretted: Optional[PostureNode] = None
            for p in chosen.posture_sequence:
                if p.hand_position > 0:
                    last_fretted = p

            tracks.append(
                _ProfileTrack(
                    profile=prof,
                    total_cost=(
                        chosen.total_cost[0],
                        chosen.total_cost[1],
                        chosen.total_cost[2],
                    ),
                    compound_path=[f"s0:{nid}" for nid in chosen.compound_path],
                    postures=list(chosen.posture_sequence),
                    last_posture=chosen.posture_sequence[-1],
                    last_fretted_posture=last_fretted,
                )
            )
        return tracks

    @staticmethod
    def _pick_for_profile(
        solutions: Tuple[ParetoSolution, ...],
        profile: str,
    ) -> ParetoSolution:
        if len(solutions) == 1:
            return solutions[0]
        if profile == "ergonomic":
            return min(solutions, key=lambda s: (s.total_cost[0], s.total_cost[2], s.total_cost[1]))
        if profile == "timbral":
            return min(solutions, key=lambda s: (s.total_cost[1], s.total_cost[0], s.total_cost[2]))
        if profile == "fluency":
            return min(solutions, key=lambda s: (s.total_cost[2], s.total_cost[0], s.total_cost[1]))

        mins = tuple(min(s.total_cost[d] for s in solutions) for d in range(3))
        maxs = tuple(max(s.total_cost[d] for s in solutions) for d in range(3))
        spans = tuple(max(maxs[d] - mins[d], 1e-9) for d in range(3))

        def norm_dist(sol: ParetoSolution) -> float:
            return math.sqrt(
                sum(((sol.total_cost[d] - mins[d]) / spans[d]) ** 2 for d in range(3))
            )

        ranked = sorted(
            solutions, key=lambda s: (norm_dist(s), lexicographic_key(s.total_cost))
        )
        if profile == "balanced_alt" and len(ranked) > 1:
            return ranked[1]
        return ranked[0]

    @staticmethod
    def _filter_pareto_nondominated(
        solutions: Tuple[ParetoSolution, ...],
    ) -> Tuple[ParetoSolution, ...]:
        kept: List[ParetoSolution] = []
        for cand in solutions:
            if any(
                dominates(other.total_cost, cand.total_cost)
                for other in solutions
                if other.solution_id != cand.solution_id
            ):
                continue
            kept.append(cand)
        return tuple(kept) if kept else solutions

    @staticmethod
    def _profile_weights(profile: str) -> Tuple[float, float, float]:
        if profile == "ergonomic":
            return (1.0, 0.15, 0.25)
        if profile == "timbral":
            return (0.25, 1.0, 0.25)
        if profile == "fluency":
            return (0.30, 0.20, 1.0)
        if profile == "balanced_alt":
            return (0.65, 0.55, 0.50)
        return (0.55, 0.45, 0.45)

    def _solve_profile_segment(
        self,
        sub_sequence: MusicalSequence,
        profile: str,
        initial_posture: Optional[PostureNode],
        initial_fretted_posture: Optional[PostureNode],
    ) -> NamoaStarResult:
        dag_builder = BaseDagBuilder(
            self.fretboard,
            self.settings.biomechanics,
            kinematic_pruning=True,
        )
        base_dag = dag_builder.build(sub_sequence)
        evaluator = CostVectorEvaluator(
            fretboard=self.fretboard,
            weights=self.settings.cost_weights,
            biomechanics=self.settings.biomechanics,
            initial_posture=initial_posture,
            initial_fretted_posture=initial_fretted_posture,
        )

        w_bio_k, w_tim_k, w_flu_k = self._profile_weights(profile)
        selected: List[PostureNode] = []
        total = [0.0, 0.0, 0.0]
        last_fretted: Optional[PostureNode] = initial_fretted_posture

        for idx, event in enumerate(sub_sequence.events):
            candidates = base_dag.layers[idx]
            scored: List[Tuple[float, Tuple[float, float, float], PostureNode]] = []

            for node in candidates:
                if idx == 0:
                    if initial_posture is None:
                        bio, timbre = evaluator._compute_initial_posture_cost(node)
                        cost_vec = (bio, timbre, 0.0)
                    else:
                        raw = evaluator.evaluate_first_order(
                            initial_posture,
                            node,
                            sub_sequence.events[0],
                            event,
                        )
                        bridge = _open_bridge_cost(last_fretted, initial_posture, node)
                        cost_vec = (
                            float(raw[0]) + bridge[0],
                            float(raw[1]) + bridge[1],
                            float(raw[2]) + bridge[2],
                        )
                else:
                    prev_node = selected[-1]
                    prev_event = sub_sequence.events[idx - 1]
                    anchor = prev_node if prev_node.hand_position > 0 else last_fretted
                    if anchor is None:
                        raw = evaluator.evaluate_first_order(
                            prev_node,
                            node,
                            prev_event,
                            event,
                        )
                    else:
                        raw = evaluator.evaluate_second_order(
                            anchor,
                            prev_node,
                            node,
                            prev_event,
                            event,
                        )
                    bridge = _open_bridge_cost(last_fretted, prev_node, node)
                    cost_vec = (
                        float(raw[0]) + bridge[0],
                        float(raw[1]) + bridge[1],
                        float(raw[2]) + bridge[2],
                    )

                scalar_score = (
                    w_bio_k * cost_vec[0]
                    + w_tim_k * cost_vec[1]
                    + w_flu_k * cost_vec[2]
                )
                scored.append((scalar_score, cost_vec, node))

            _, best_cost, chosen = min(
                scored,
                key=lambda item: (item[0], item[2].node_id),
            )

            for dim in range(3):
                total[dim] += best_cost[dim]
            selected.append(chosen)
            if chosen.hand_position > 0:
                last_fretted = chosen

        solution = ParetoSolution(
            solution_id=1,
            total_cost=_CostTuple((total[0], total[1], total[2])),
            compound_path=tuple(f"P_{profile}_{idx}:{node.node_id}" for idx, node in enumerate(selected)),
            posture_sequence=tuple(selected),
        )
        return NamoaStarResult((solution,), len(selected), len(selected), 0)