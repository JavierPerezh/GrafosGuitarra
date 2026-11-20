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
from src.optimization.namoa_star import NamoaStarResult, NamoaStarSolver
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

    # 2. Construcción del DAG base y filtrado anatómico intra-capa
    fretboard = FretboardMap(settings.physics)
    base_builder = BaseDagBuilder(fretboard, settings.biomechanics)
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
        print(
            f"[G08] Pieza: '{sequence.title}' | Capas: {base_dag.num_layers} | "
            f"DAG base: ({base_dag.num_posture_nodes} nodos, {base_dag.num_edges} aristas) | "
            f"Grafo dual: ({dual_graph.num_compound_nodes} nodos, {dual_graph.num_edges} aristas)"
        )
        print(
            f"[G08] Soluciones Pareto: {len(result.solutions)} | "
            f"Tiempo total: {elapsed_ms:.2f} ms"
        )
        print(f"[G08] Reportes guardados en:\n  - {txt_file}\n  - {json_file}")
        print(renderer.render_all_ascii(result, sequence))

    return result, txt_file, json_file


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