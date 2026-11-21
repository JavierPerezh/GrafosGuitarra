import argparse
from pathlib import Path
import sys
import time
from typing import Optional, Sequence, Tuple

from config.instrument_settings import DEFAULT_SETTINGS, InstrumentSettings
from src.domain.fretboard_map import FretboardMap
from src.graph.base_dag import BaseDagBuilder, GraphConstructionError
from src.graph.cost_vector import CostVectorEvaluator
from src.graph.dual_graph import DualGraphBuilder
from src.ingestion.midi_xml_parser import MidiXmlScoreParser
from src.ingestion.parser_interface import ScoreParsingError
from src.optimization.namoa_star import (
    NamoaStarResult,
    NamoaStarSolver,
    ParetoSolution,
    select_diverse_solutions,
)
from src.optimization.pareto_dominance import dominates
from src.optimization.segmented_solver import SegmentedNamoaSolver
from src.presentation.tab_renderer import TabRenderer


def run_pipeline(
    input_path: Path,
    output_dir: Path = Path("data/output"),
    settings: InstrumentSettings = DEFAULT_SETTINGS,
    max_solutions: Optional[int] = None,
    line_width: int = 80,
    show_articulations: bool = True,
    verbose: bool = True,
) -> Tuple[NamoaStarResult, Path, Path]:
    t0 = time.perf_counter()

    # 1. Ingestión musical
    parser = MidiXmlScoreParser()
    sequence = parser.parse(input_path)

    if len(sequence) > 20:
        seg_solver = SegmentedNamoaSolver(
            settings=settings,
            max_solutions=max_solutions,
        )
        result = seg_solver.solve(sequence)
        base_dag = None
        dual_graph = None
    else:
        # 2. Construcción del DAG base y filtrado anatómico intra-capa
        fretboard = FretboardMap(settings.physics)
        base_builder = BaseDagBuilder(
            fretboard,
            settings.biomechanics,
            kinematic_pruning=len(sequence) > 12,
        )
        base_dag = base_builder.build(sequence)

        # 3. Transformación al Grafo de Estados Compuestos (memoria inercial)
        cost_evaluator = CostVectorEvaluator(
            fretboard=fretboard,
            weights=settings.cost_weights,
            biomechanics=settings.biomechanics,
        )
        dual_builder = DualGraphBuilder(cost_evaluator)
        dual_graph = dual_builder.build(base_dag, sequence)

        # 4. Búsqueda multiobjetivo NAMOA*
        solver = NamoaStarSolver(max_solutions=max_solutions)
        result = solver.solve(dual_graph)

    # 5. Renderizado y exportación de reportes
    renderer = TabRenderer(
        physics=settings.physics,
        line_width=line_width,
        show_articulations=show_articulations,
    )
    txt_file, json_file = renderer.save_report(
        result=result,
        sequence=sequence,
        output_dir=output_dir,
        base_filename=input_path.stem,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if verbose:
        if base_dag is not None and dual_graph is not None:
            print(
                f"[G08] Pieza: '{sequence.title}' | Capas: {base_dag.num_layers} | "
                f"DAG base: ({base_dag.num_posture_nodes} nodos, {base_dag.num_edges} aristas) | "
                f"Grafo dual: ({dual_graph.num_compound_nodes} nodos, {dual_graph.num_edges} aristas)"
            )
        else:
            print(
                f"[G08] Pieza: '{sequence.title}' | Eventos: {len(sequence)} | "
                "Ejecución: segmentación por compases"
            )
        print(
            f"[G08] Soluciones Pareto: {len(result.solutions)} | "
            f"Tiempo total: {elapsed_ms:.2f} ms"
        )
        print(f"[G08] Reportes guardados en:\n  - {txt_file}\n  - {json_file}")
        print(renderer.render_all_ascii(result, sequence))

    return result, txt_file, json_file


def _filter_segment_solutions(solutions: list[ParetoSolution]) -> list[ParetoSolution]:
    frontier: list[ParetoSolution] = []
    for candidate in solutions:
        if any(
            dominates(existing.total_cost, candidate.total_cost)
            for existing in frontier
        ):
            continue
        frontier = [
            existing
            for existing in frontier
            if not dominates(candidate.total_cost, existing.total_cost)
        ]
        frontier.append(candidate)
    return frontier


def _solve_segmented_pipeline(
    sequence,
    parser: MidiXmlScoreParser,
    settings: InstrumentSettings,
    max_solutions: Optional[int],
) -> Tuple[NamoaStarResult, object, object]:
    fretboard = FretboardMap(settings.physics)
    cost_evaluator = CostVectorEvaluator(
        fretboard=fretboard,
        weights=settings.cost_weights,
        biomechanics=settings.biomechanics,
    )
    segments = parser.segment_sequence(sequence, settings)
    combined: list[ParetoSolution] = []
    expanded = generated = pruned = 0

    for segment_idx, segment in enumerate(segments):
        base_builder = BaseDagBuilder(
            fretboard,
            settings.biomechanics,
            kinematic_pruning=len(segment) > 12,
        )
        base_dag = base_builder.build(segment)
        dual_builder = DualGraphBuilder(cost_evaluator)
        dual_graph = dual_builder.build(base_dag, segment)
        segment_result = NamoaStarSolver(max_solutions=None).solve(dual_graph)
        expanded += segment_result.expanded_labels
        generated += segment_result.generated_labels
        pruned += segment_result.pruned_labels

        if not segment_result.solutions:
            continue

        if segment_idx == 0:
            combined = [
                ParetoSolution(
                    solution_id=idx + 1,
                    total_cost=sol.total_cost,
                    compound_path=sol.compound_path,
                    posture_sequence=sol.posture_sequence,
                )
                for idx, sol in enumerate(segment_result.solutions)
            ]
            continue

        previous_segment = segments[segment_idx - 1]
        chained: list[ParetoSolution] = []
        for previous in combined:
            previous_posture = previous.posture_sequence[-1]
            previous_event = previous_segment.events[-1]
            for sol in segment_result.solutions:
                first_posture = sol.posture_sequence[0]
                first_event = segment.events[0]
                boundary_cost = cost_evaluator.evaluate_first_order(
                    curr_node=previous_posture,
                    next_node=first_posture,
                    curr_event=previous_event,
                    next_event=first_event,
                )
                total = tuple(
                    previous.total_cost[i] + float(boundary_cost[i]) + float(sol.total_cost[i])
                    for i in range(3)
                )
                chained.append(
                    ParetoSolution(
                        solution_id=len(chained) + 1,
                        total_cost=(total[0], total[1], total[2]),
                        compound_path=previous.compound_path + sol.compound_path,
                        posture_sequence=previous.posture_sequence + sol.posture_sequence,
                    )
                )
        combined = _filter_segment_solutions(chained)

    selected = select_diverse_solutions(tuple(combined), max_solutions)
    reindexed = tuple(
        ParetoSolution(idx + 1, sol.total_cost, sol.compound_path, sol.posture_sequence)
        for idx, sol in enumerate(selected)
    )
    return NamoaStarResult(reindexed, expanded, generated, pruned), base_dag, dual_graph


def _build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        prog="GrafosGuitarra",
        description="Generador algorítmico de tablaturas óptimas mediante DAGs duales y NAMOA*.",
    )
    arg_parser.add_argument(
        "input_file",
        type=Path,
        help="Ruta al archivo musical de entrada (.mid, .midi, .musicxml, .xml, .mxl, .json).",
    )
    arg_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Directorio de salida para las tablaturas y reportes JSON (por defecto: data/output).",
    )
    arg_parser.add_argument(
        "-k",
        "--max-solutions",
        type=int,
        default=None,
        help="Límite máximo de soluciones Pareto-óptimas a extraer.",
    )
    arg_parser.add_argument(
        "-w",
        "--line-width",
        type=int,
        default=80,
        help="Ancho máximo de línea para la tablatura ASCII (por defecto: 80).",
    )
    arg_parser.add_argument(
        "--no-articulations",
        action="store_true",
        help="Oculta símbolos de articulación (h, p, /, \\) en la tablatura ASCII.",
    )
    arg_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suprime la salida detallada en consola.",
    )
    return arg_parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        run_pipeline(
            input_path=args.input_file,
            output_dir=args.output_dir,
            max_solutions=args.max_solutions,
            line_width=args.line_width,
            show_articulations=not args.no_articulations,
            verbose=not args.quiet,
        )
        return 0
    except (FileNotFoundError, ScoreParsingError, GraphConstructionError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())