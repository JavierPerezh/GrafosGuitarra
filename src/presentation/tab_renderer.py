import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from config.instrument_settings import DEFAULT_SETTINGS, GuitarPhysicsConfig
from src.domain.musical_event import ArticulationType, MusicalEvent, MusicalSequence
from src.graph.base_dag import PostureNode
from src.optimization.namoa_star import NamoaStarResult, ParetoSolution

FrontBounds = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


class TabRenderer:
    def __init__(
        self,
        physics: GuitarPhysicsConfig = DEFAULT_SETTINGS.physics,
        line_width: int = 80,
        show_articulations: bool = True,
    ) -> None:
        self.physics = physics
        self.line_width = max(40, line_width)
        self.show_articulations = show_articulations

    def _string_labels(self) -> List[str]:
        labels: List[str] = []
        for idx, name in enumerate(self.physics.string_names):
            note_letter = name[0]
            labels.append(note_letter.lower() if idx == 0 else note_letter.upper())
        return labels

    @staticmethod
    def _compute_front_bounds(
        solutions: Sequence[ParetoSolution],
    ) -> Optional[FrontBounds]:
        if not solutions:
            return None
        mins = (
            min(float(sol.total_cost[0]) for sol in solutions),
            min(float(sol.total_cost[1]) for sol in solutions),
            min(float(sol.total_cost[2]) for sol in solutions),
        )
        maxs = (
            max(float(sol.total_cost[0]) for sol in solutions),
            max(float(sol.total_cost[1]) for sol in solutions),
            max(float(sol.total_cost[2]) for sol in solutions),
        )
        return mins, maxs

    def render_ascii(
        self,
        solution: ParetoSolution,
        sequence: MusicalSequence,
        front_bounds: Optional[FrontBounds] = None,
    ) -> str:
        if len(solution.posture_sequence) != len(sequence):
            raise ValueError(
                "La longitud de la secuencia de posturas no coincide con los eventos."
            )

        labels = self._string_labels()
        num_strings = self.physics.num_strings
        columns: List[List[str]] = []

        for idx, posture in enumerate(solution.posture_sequence):
            prev_posture = solution.posture_sequence[idx - 1] if idx > 0 else None
            event = sequence.events[idx]
            col_tokens = self._format_column(posture, prev_posture, event, num_strings)
            columns.append(col_tokens)

        w_bio, w_timbre, w_fluidez = solution.total_cost
        profile = getattr(solution, "profile", None)
        if profile is None:
            profile = self._profile_label(solution, front_bounds=front_bounds)

        header = (
            f"=== Opción Pareto #{solution.solution_id} {profile} | {sequence.title} ===\n"
            f"Costo [W_bio={w_bio:.3f}, W_timbre={w_timbre:.3f}, W_fluidez={w_fluidez:.3f}]"
        )

        systems = self._wrap_columns_into_systems(labels, columns)
        return f"{header}\n\n" + "\n\n".join(systems)

    def render_all_ascii(
        self,
        result: NamoaStarResult,
        sequence: MusicalSequence,
    ) -> str:
        summary = (
            f"Pieza: {sequence.title} | Eventos: {len(sequence)} | Tempo: {sequence.tempo_bpm:.1f} BPM\n"
            f"Frontera de Pareto: {len(result.solutions)} soluciones | "
            f"Etiquetas (expandidas={result.expanded_labels}, "
            f"generadas={result.generated_labels}, podadas={result.pruned_labels})"
        )
        if not result.solutions:
            return f"{summary}\n\nNo se encontraron trayectorias viables."

        front_bounds = self._compute_front_bounds(result.solutions)
        blocks = [summary]
        for sol in result.solutions:
            blocks.append(self.render_ascii(sol, sequence, front_bounds=front_bounds))
        return "\n\n" + ("\n" + "-" * self.line_width + "\n\n").join(blocks)

    def to_dict(
        self,
        result: NamoaStarResult,
        sequence: MusicalSequence,
    ) -> Dict[str, Any]:
        front_bounds = self._compute_front_bounds(result.solutions)
        solutions_payload: List[Dict[str, Any]] = []

        for sol in result.solutions:
            w_bio, w_timbre, w_fluidez = sol.total_cost
            steps: List[Dict[str, Any]] = []
            for idx, posture in enumerate(sol.posture_sequence):
                event = sequence.events[idx]
                steps.append(
                    {
                        "event_index": idx,
                        "onset_sec": event.onset_sec,
                        "hand_position": posture.hand_position,
                        "assignments": [
                            {
                                "string": a.string,
                                "fret": a.fret,
                                "finger": a.finger,
                                "midi_pitch": a.midi_pitch,
                            }
                            for a in posture.assignments
                        ],
                    }
                )

            solutions_payload.append(
                {
                    "solution_id": sol.solution_id,
                    "profile": self._profile_label(sol, front_bounds=front_bounds),
                    "cost_vector": {
                        "W_bio": round(float(w_bio), 6),
                        "W_timbre": round(float(w_timbre), 6),
                        "W_fluidez": round(float(w_fluidez), 6),
                    },
                    "ascii_tab": self.render_ascii(sol, sequence, front_bounds=front_bounds),
                    "steps": steps,
                }
            )

        return {
            "title": sequence.title,
            "tempo_bpm": sequence.tempo_bpm,
            "num_events": len(sequence),
            "search_stats": {
                "pareto_solutions": len(result.solutions),
                "expanded_labels": result.expanded_labels,
                "generated_labels": result.generated_labels,
                "pruned_labels": result.pruned_labels,
            },
            "solutions": solutions_payload,
        }

    def render_json(
        self,
        result: NamoaStarResult,
        sequence: MusicalSequence,
        indent: int = 2,
    ) -> str:
        return json.dumps(self.to_dict(result, sequence), indent=indent, ensure_ascii=False)

    def save_report(
        self,
        result: NamoaStarResult,
        sequence: MusicalSequence,
        output_dir: Union[str, Path] = "data/output",
        base_filename: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stem = base_filename or self._sanitize_filename(sequence.title)
        txt_file = out_path / f"{stem}_tabs.txt"
        json_file = out_path / f"{stem}_pareto.json"

        txt_file.write_text(self.render_all_ascii(result, sequence), encoding="utf-8")
        json_file.write_text(self.render_json(result, sequence), encoding="utf-8")
        return txt_file, json_file

    def _format_column(
        self,
        posture: PostureNode,
        prev_posture: Optional[PostureNode],
        event: MusicalEvent,
        num_strings: int,
    ) -> List[str]:
        prev_by_string = (
            {a.string: a for a in prev_posture.assignments} if prev_posture else {}
        )
        curr_by_string = {a.string: a for a in posture.assignments}
        note_art_by_pitch = {n.midi_pitch: n.articulation for n in event.notes}

        raw_tokens: List[str] = []
        for s in range(1, num_strings + 1):
            if s not in curr_by_string:
                raw_tokens.append("-")
                continue

            curr_a = curr_by_string[s]
            prefix = ""
            if self.show_articulations and s in prev_by_string:
                prev_a = prev_by_string[s]
                art = note_art_by_pitch.get(curr_a.midi_pitch, ArticulationType.NORMAL)
                if art == ArticulationType.LEGATO and prev_a.fret != curr_a.fret:
                    prefix = "h" if curr_a.fret > prev_a.fret else "p"
                elif art == ArticulationType.SLIDE and prev_a.fret != curr_a.fret:
                    prefix = "/" if curr_a.fret > prev_a.fret else "\\"

            raw_tokens.append(f"{prefix}{curr_a.fret}")

        col_width = max(len(tok) for tok in raw_tokens) + 1
        return [tok.ljust(col_width, "-") for tok in raw_tokens]

    def _wrap_columns_into_systems(
        self,
        labels: List[str],
        columns: List[List[str]],
    ) -> List[str]:
        num_strings = len(labels)
        prefix_width = max(len(lbl) for lbl in labels) + 2
        available_width = max(20, self.line_width - prefix_width - 1)

        systems: List[str] = []
        current_lines = [f"{lbl.rjust(prefix_width - 2)}|--" for lbl in labels]
        used_width = 2

        for col in columns:
            col_len = len(col[0])
            if used_width + col_len > available_width and used_width > 2:
                for s in range(num_strings):
                    current_lines[s] += "|"
                systems.append("\n".join(current_lines))
                current_lines = [f"{lbl.rjust(prefix_width - 2)}|--" for lbl in labels]
                used_width = 2

            for s in range(num_strings):
                current_lines[s] += col[s]
            used_width += col_len

        for s in range(num_strings):
            current_lines[s] += "|"
        systems.append("\n".join(current_lines))
        return systems

    @staticmethod
    def _profile_label(
        solution: ParetoSolution,
        front_bounds: Optional[FrontBounds] = None,
    ) -> str:
        if front_bounds is None:
            return "[Equilibrada]"

        mins, maxs = front_bounds
        spans = tuple(maxs[d] - mins[d] for d in range(3))
        active_dims = [d for d in range(3) if spans[d] > 1e-6]
        if not active_dims:
            return "[Equilibrada]"

        values = tuple(float(v) for v in solution.total_cost)
        normalized = [
            (values[d] - mins[d]) / spans[d] if spans[d] > 1e-6 else 0.5
            for d in range(3)
        ]

        best_dim = min(active_dims, key=lambda d: normalized[d])
        best_score = normalized[best_dim]
        other_scores = [normalized[d] for d in active_dims if d != best_dim]

        if best_score <= 0.15 and (not other_scores or min(other_scores) - best_score >= 0.15):
            return (
                "[Ergonómica]"
                if best_dim == 0
                else "[Tímbrica]"
                if best_dim == 1
                else "[Fluidez]"
            )

        return "[Equilibrada]"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        return clean.strip("_") or "tablatura"
