import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from music21 import chord, converter, meter, note, spanner, tempo

from config.instrument_settings import DEFAULT_SETTINGS, GuitarPhysicsConfig
from src.domain.musical_event import (
    ArticulationType,
    MusicalEvent,
    MusicalSequence,
    NoteEvent,
)


class ScoreParsingError(Exception):
    """Error al leer o normalizar una partitura musical."""


class MidiXmlScoreParser:
    SUPPORTED_EXTENSIONS = {".mid", ".midi", ".xml", ".musicxml", ".mxl", ".json"}

    def __init__(
        self,
        physics: GuitarPhysicsConfig = DEFAULT_SETTINGS.physics,
        onset_quantize_sec: float = 0.035,
    ) -> None:
        self.physics = physics
        self.onset_quantize_sec = onset_quantize_sec

    def parse(self, file_path: Union[str, Path]) -> MusicalSequence:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo musical: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ScoreParsingError(
                f"Formato no soportado '{ext}'. Extensiones válidas: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".json":
            return self._parse_json(path)
        return self._parse_music21(path)

    def _parse_json(self, path: Path) -> MusicalSequence:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScoreParsingError(f"JSON inválido en {path}: {exc}") from exc

        title = str(payload.get("title", path.stem))
        tempo_bpm = float(payload.get("tempo_bpm", 120.0))
        raw_ts = payload.get("time_signature", list(DEFAULT_SETTINGS.solver.default_time_signature))
        time_sig = (int(raw_ts[0]), int(raw_ts[1])) if len(raw_ts) >= 2 else (4, 4)

        raw_events = payload.get("events", [])
        if not raw_events:
            raise ScoreParsingError(f"El archivo {path} no contiene eventos musicales.")

        events: List[MusicalEvent] = []
        for idx, ev_data in enumerate(raw_events):
            onset_sec = float(ev_data["onset_sec"])
            notes_list: List[NoteEvent] = []
            for n_data in ev_data.get("notes", []):
                art_str = str(n_data.get("articulation", "normal")).lower()
                articulation = (
                    ArticulationType(art_str)
                    if art_str in {a.value for a in ArticulationType}
                    else ArticulationType.NORMAL
                )
                notes_list.append(
                    NoteEvent(
                        midi_pitch=int(n_data["midi"]),
                        duration_sec=max(float(n_data.get("duration_sec", 0.25)), 0.02),
                        velocity=int(n_data.get("velocity", 80)),
                        articulation=articulation,
                    )
                )
            if notes_list:
                events.append(
                    MusicalEvent(
                        index=idx,
                        onset_sec=onset_sec,
                        notes=tuple(notes_list),
                    )
                )

        if not events:
            raise ScoreParsingError(f"No se extrajeron notas válidas desde {path}.")

        return MusicalSequence.from_events(
            events=events,
            title=title,
            tempo_bpm=tempo_bpm,
            metadata={"source": str(path), "time_signature": time_sig},
        )

    def _parse_music21(self, path: Path) -> MusicalSequence:
        try:
            score = converter.parse(str(path))
        except Exception as exc:
            raise ScoreParsingError(f"Error de music21 al leer {path}: {exc}") from exc

        tempo_bpm = self._extract_tempo(score)
        time_sig = self._extract_time_signature(score)
        sec_per_quarter = 60.0 / tempo_bpm

        slurred_ids = set()
        for sp in score.recurse().getElementsByClass(spanner.Slur):
            spanned = list(sp.getSpannedElements())
            for elem in spanned[1:]:
                slurred_ids.add(id(elem))

        raw_entries: List[Tuple[float, NoteEvent]] = []
        flat_notes = score.flatten().notes

        for elem in flat_notes:
            onset_sec = float(elem.offset) * sec_per_quarter
            dur_sec = max(float(elem.quarterLength) * sec_per_quarter, 0.03)
            articulation = (
                ArticulationType.LEGATO
                if id(elem) in slurred_ids
                else ArticulationType.NORMAL
            )

            if isinstance(elem, note.Note):
                if elem.tie is not None and elem.tie.type in ("stop", "continue"):
                    continue
                vel = int(elem.volume.velocity or 80)
                raw_entries.append(
                    (
                        onset_sec,
                        NoteEvent(
                            midi_pitch=int(elem.pitch.midi),
                            duration_sec=dur_sec,
                            velocity=vel,
                            articulation=articulation,
                        ),
                    )
                )
            elif isinstance(elem, chord.Chord):
                vel = int(elem.volume.velocity or 80)
                for p in elem.pitches:
                    raw_entries.append(
                        (
                            onset_sec,
                            NoteEvent(
                                midi_pitch=int(p.midi),
                                duration_sec=dur_sec,
                                velocity=vel,
                                articulation=articulation,
                            ),
                        )
                    )

        if not raw_entries:
            raise ScoreParsingError(f"No se encontraron notas ejecutables en {path}.")

        raw_entries.sort(key=lambda item: (item[0], item[1].midi_pitch))

        # Agrupación temporal (clustering de rasgueos dentro de onset_quantize_sec)
        clusters: List[Tuple[float, List[NoteEvent]]] = []
        active_end_times: Dict[int, float] = {}

        for onset_sec, n_ev in raw_entries:
            prev_end = active_end_times.get(n_ev.midi_pitch, -1.0)
            if onset_sec < prev_end - 0.015 and n_ev.velocity < 35:
                continue
            active_end_times[n_ev.midi_pitch] = onset_sec + n_ev.duration_sec

            if not clusters:
                clusters.append((onset_sec, [n_ev]))
                continue

            cluster_onset, cluster_notes = clusters[-1]
            if (onset_sec - cluster_onset) <= self.onset_quantize_sec:
                cluster_notes.append(n_ev)
            else:
                clusters.append((onset_sec, [n_ev]))

        events: List[MusicalEvent] = []
        for idx, (onset_sec, notes_group) in enumerate(clusters):
            # Limita a máximo 6 notas simultáneas (cuerdas de la guitarra)
            dedup: Dict[int, NoteEvent] = {}
            for n in sorted(notes_group, key=lambda x: x.midi_pitch):
                dedup[n.midi_pitch] = n
            selected_notes = tuple(list(dedup.values())[: self.physics.num_strings])
            events.append(
                MusicalEvent(
                    index=idx,
                    onset_sec=round(onset_sec, 6),
                    notes=selected_notes,
                )
            )

        title = path.stem
        if score.metadata is not None and score.metadata.title:
            title = str(score.metadata.title)

        return MusicalSequence.from_events(
            events=events,
            title=title,
            tempo_bpm=tempo_bpm,
            metadata={"source": str(path), "time_signature": time_sig},
        )

    @staticmethod
    def _extract_tempo(score: object) -> float:
        marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))  # type: ignore[attr-defined]
        for mm in marks:
            if mm.number is not None and mm.number > 0:
                return float(mm.number)
        return 120.0

    @staticmethod
    def _extract_time_signature(score: object) -> Tuple[int, int]:
        sigs = list(score.recurse().getElementsByClass(meter.TimeSignature))  # type: ignore[attr-defined]
        for ts in sigs:
            if ts.numerator and ts.denominator:
                return (int(ts.numerator), int(ts.denominator))
        return DEFAULT_SETTINGS.solver.default_time_signature

    def segment_sequence(
        self,
        sequence: MusicalSequence,
        segmentation_settings: Optional[Any] = None,
    ) -> List[MusicalSequence]:
        """Divide una secuencia por compases y un máximo configurable de eventos."""
        from dataclasses import replace

        from config.instrument_settings import InstrumentSettings, SolverConfig
        from src.optimization.segmented_solver import MeasureAwareSegmenter

        settings = segmentation_settings
        if isinstance(settings, SolverConfig):
            settings = replace(DEFAULT_SETTINGS, solver=settings)
        elif not isinstance(settings, InstrumentSettings):
            settings = DEFAULT_SETTINGS

        segmenter = MeasureAwareSegmenter(settings)
        return [segment.sequence for segment in segmenter.segment(sequence)]
