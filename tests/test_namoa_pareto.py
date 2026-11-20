import json
from pathlib import Path
import numpy as np

from config.instrument_settings import DEFAULT_SETTINGS
from main import main
from src.domain.fretboard_map import FretboardMap
from src.domain.musical_event import (
    ArticulationType,
    MusicalEvent,
    MusicalSequence,
    NoteEvent,
)
from src.graph.base_dag import BaseDagBuilder
from src.graph.cost_vector import CostVectorEvaluator
from src.graph.dual_graph import DualGraphBuilder
from src.optimization.namoa_star import NamoaStarSolver
from src.optimization.pareto_dominance import (
    dominates,
    filter_non_dominated,
    update_pareto_front,
    weakly_dominates,
)


def _build_and_solve(sequence: MusicalSequence, max_solutions: int | None = None):
    fretboard = FretboardMap(DEFAULT_SETTINGS.physics)
    base_dag = BaseDagBuilder(fretboard, DEFAULT_SETTINGS.biomechanics).build(sequence)
    evaluator = CostVectorEvaluator(
        fretboard=fretboard,
        weights=DEFAULT_SETTINGS.cost_weights,
        biomechanics=DEFAULT_SETTINGS.biomechanics,
    )
    dual_graph = DualGraphBuilder(evaluator).build(base_dag, sequence)
    solver = NamoaStarSolver(max_solutions=max_solutions)
    return dual_graph, solver, solver.solve(dual_graph)


class TestParetoDominanceAndNamoa:
    def test_pareto_dominance_algebra(self) -> None:
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.5, 3.0])
        v3 = np.array([2.0, 1.5, 3.0])

        assert dominates(v1, v2)
        assert not dominates(v2, v1)
        assert not dominates(v1, v1)
        assert weakly_dominates(v1, v1)
        assert not dominates(v1, v3)
        assert not dominates(v3, v1)

        front = filter_non_dominated([v2, v1, v3, v1.copy()])
        assert len(front) == 2
        assert any(np.allclose(vec, v1) for vec in front)
        assert any(np.allclose(vec, v3) for vec in front)

        accepted, updated = update_pareto_front(front, np.array([0.8, 1.4, 2.9]))
        assert accepted
        assert len(updated) == 1

    def test_backward_heuristic_is_admissible(self) -> None:
        seq = MusicalSequence.from_events(
            [
                MusicalEvent(0, 0.0, (NoteEvent(52, 0.25),)),
                MusicalEvent(1, 0.25, (NoteEvent(55, 0.25),)),
                MusicalEvent(2, 0.50, (NoteEvent(59, 0.25),)),
                MusicalEvent(3, 0.75, (NoteEvent(64, 0.25),)),
            ]
        )
        dual_graph, solver, result = _build_and_solve(seq)
        heuristic = solver.compute_backward_heuristic(dual_graph)

        h_source = heuristic[dual_graph.source_id]
        assert np.all(np.isfinite(h_source))
        assert np.all(h_source >= 0.0)
        assert len(result.solutions) >= 1

        # Admisibilidad: h(source) nunca sobrestima el costo real de ninguna solución Pareto
        for sol in result.solutions:
            assert np.all(h_source <= sol.total_cost + 1e-9)

    def test_pareto_front_solutions_are_mutually_non_dominated(self) -> None:
        # Pasaje ascendente que contrapone estabilidad posicional (W_bio) vs continuidad de cuerda (W_timbre)
        seq = MusicalSequence.from_events(
            [
                MusicalEvent(0, 0.00, (NoteEvent(52, 0.2),)),
                MusicalEvent(1, 0.20, (NoteEvent(54, 0.2),)),
                MusicalEvent(2, 0.40, (NoteEvent(56, 0.2),)),
                MusicalEvent(3, 0.60, (NoteEvent(57, 0.2),)),
            ]
        )
        _, _, result = _build_and_solve(seq)

        assert len(result.solutions) >= 2
        for i, sol_a in enumerate(result.solutions):
            assert len(sol_a.posture_sequence) == len(seq)
            for j, sol_b in enumerate(result.solutions):
                if i == j:
                    continue
                assert not dominates(sol_a.total_cost, sol_b.total_cost)

    def test_legato_articulation_produces_same_string_candidate(self) -> None:
        seq = MusicalSequence.from_events(
            [
                MusicalEvent(0, 0.0, (NoteEvent(55, 0.2, articulation=ArticulationType.NORMAL),)),
                MusicalEvent(1, 0.2, (NoteEvent(57, 0.2, articulation=ArticulationType.LEGATO),)),
            ]
        )
        _, _, result = _build_and_solve(seq)

        assert len(result.solutions) >= 1
        has_same_string_hammer_on = any(
            sol.posture_sequence[0].assignments[0].string
            == sol.posture_sequence[1].assignments[0].string
            and sol.posture_sequence[0].assignments[0].fret
            < sol.posture_sequence[1].assignments[0].fret
            for sol in result.solutions
        )
        assert has_same_string_hammer_on

    def test_end_to_end_cli_pipeline(self, tmp_path: Path) -> None:
        sample_score = {
            "title": "Test Riff G08",
            "tempo_bpm": 120.0,
            "events": [
                {"onset_sec": 0.0, "pitches": [40, 47], "duration_sec": 0.5},
                {"onset_sec": 0.5, "pitches": [43], "duration_sec": 0.25},
                {
                    "onset_sec": 0.75,
                    "pitches": [45],
                    "duration_sec": 0.25,
                    "articulation": "legato",
                },
            ],
        }
        input_json = tmp_path / "riff.json"
        output_dir = tmp_path / "out"
        input_json.write_text(json.dumps(sample_score), encoding="utf-8")

        exit_code = main([str(input_json), "-o", str(output_dir), "-q"])
        assert exit_code == 0

        txt_out = output_dir / "riff_tabs.txt"
        json_out = output_dir / "riff_pareto.json"
        assert txt_out.exists()
        assert json_out.exists()

        report = json.loads(json_out.read_text(encoding="utf-8"))
        assert report["title"] == "Test Riff G08"
        assert report["search_stats"]["pareto_solutions"] >= 1
        first_sol = report["solutions"][0]
        assert set(first_sol["cost_vector"].keys()) == {"W_bio", "W_timbre", "W_fluidez"}