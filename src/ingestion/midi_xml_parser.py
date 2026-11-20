import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from music21 import articulations, chord, converter, note, spanner, tempo

from src.domain.musical_event import (
    ArticulationType,
    MusicalEvent,
    MusicalSequence,
    NoteEvent,
)
from src.ingestion.parser_interface import BaseScoreParser, ScoreParsingError


class MidiXmlScoreParser(BaseScoreParser):
    supported_extensions: Tuple[str, ...] = (
        ".mid",
        ".midi",
        ".musicxml",
        ".xml",
        ".mxl",
        ".json",
    )

    def __init__(self, onset_quantize_sec: float = 0.005) -> None:
        self.onset_quantize_sec = onset_quantize_sec

    def parse(self, file_path: Union[str, Path]) -> MusicalSequence:
        path = self._validate_file(file_path)
        if path.suffix.lower() == ".json":
            return self._parse_json(path)
        return self._parse_music21(path)

    def _parse_music21(self, path: Path) -> MusicalSequence:
        try:
            score = converter.parse(str(path))
        except Exception as exc:
            raise ScoreParsingError(f"Error al leer '{path.name}': {exc}") from exc

        flat_score = score.flatten()
        title = (
            score.metadata.title
            if score.metadata and score.metadata.title
            else path.stem
        )
        marks = list(flat_score.getElementsByClass(tempo.MetronomeMark))
        bpm = float(marks[0].number) if marks and marks[0].number else 120.0

        grouped_notes: Dict[float, List[NoteEvent]] = {}
        for entry in flat_score.secondsMap:
            element = entry["element"]
            if not isinstance(element, (note.Note, chord.Chord)):
                continue

            onset = round(
                round(float(entry["offsetSeconds"]) / self.onset_quantize_sec)
                * self.onset_quantize_sec,
                6,
            )
            duration = max(float(entry["durationSeconds"]), 1e-3)
            extracted = self._extract_note_events(element, duration)
            if extracted:
                grouped_notes.setdefault(onset, []).extend(extracted)

        if not grouped_notes:
            raise ScoreParsingError(f"No se encontraron notas válidas en '{path.name}'.")

        events: List[MusicalEvent] = []
        active_end_times: Dict[int, float] = {}

        for onset in sorted(grouped_notes.keys()):
            raw_notes = grouped_notes[onset]
            has_new_attack = any(not n.is_tied_continuation for n in raw_notes)

            # Omite eventos donde solo continúan ligaduras sin un nuevo ataque
            if not has_new_attack:
                for n in raw_notes:
                    if n.midi_pitch in active_end_times:
                        active_end_times[n.midi_pitch] = max(
                            active_end_times[n.midi_pitch], onset + n.duration_sec
                        )
                continue

            event_notes: Dict[int, NoteEvent] = {}
            for n in raw_notes:
                event_notes[n.midi_pitch] = n
                active_end_times[n.midi_pitch] = onset + n.duration_sec

            events.append(
                MusicalEvent(
                    index=len(events),
                    onset_sec=onset,
                    notes=tuple(event_notes.values()),
                )
            )

        if not events:
            raise ScoreParsingError(f"No se generaron eventos atacados en '{path.name}'.")

        return MusicalSequence.from_events(
            events=events,
            title=title,
            tempo_bpm=bpm,
            metadata={"source_file": str(path), "format": path.suffix.lower()},
        )

    def _extract_note_events(
        self,
        element: Union[note.Note, chord.Chord],
        duration_sec: float,
    ) -> List[NoteEvent]:
        if isinstance(element, note.Note):
            return [self._build_single_note(element, element, duration_sec)]

        return [
            self._build_single_note(sub_note, element, duration_sec)
            for sub_note in element.notes
        ]

    def _build_single_note(
        self,
        target_note: note.Note,
        owner: Union[note.Note, chord.Chord],
        duration_sec: float,
    ) -> NoteEvent:
        tie_obj = target_note.tie or owner.tie
        is_continuation = bool(tie_obj and tie_obj.type in ("stop", "continue"))
        articulation = self._resolve_articulation(target_note, owner, is_continuation)

        velocity = 80
        if target_note.volume and target_note.volume.velocity is not None:
            velocity = int(target_note.volume.velocity)
        elif owner.volume and owner.volume.velocity is not None:
            velocity = int(owner.volume.velocity)

        return NoteEvent(
            midi_pitch=int(target_note.pitch.midi),
            duration_sec=duration_sec,
            velocity=max(1, min(127, velocity)),
            articulation=articulation,
            is_tied_continuation=is_continuation,
        )

    def _resolve_articulation(
        self,
        target_note: note.Note,
        owner: Union[note.Note, chord.Chord],
        is_continuation: bool,
    ) -> ArticulationType:
        if is_continuation:
            return ArticulationType.TIE

        for sp in list(target_note.getSpannerSites()) + list(owner.getSpannerSites()):
            if isinstance(sp, spanner.Glissando):
                return ArticulationType.SLIDE
            if isinstance(sp, spanner.Slur) and not (
                sp.isFirst(target_note) or sp.isFirst(owner)
            ):
                return ArticulationType.LEGATO

        for art in list(target_note.articulations) + list(owner.articulations):
            if isinstance(art, articulations.Staccato):
                return ArticulationType.STACCATO
            if isinstance(art, (articulations.HammerOn, articulations.PullOff)):
                return ArticulationType.LEGATO

        return ArticulationType.NORMAL

    def _parse_json(self, path: Path) -> MusicalSequence:
        try:
            payload: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ScoreParsingError(f"JSON inválido en '{path.name}': {exc}") from exc

        raw_events = payload.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ScoreParsingError("El JSON debe contener una lista no vacía 'events'.")

        events: List[MusicalEvent] = []
        for idx, item in enumerate(raw_events):
            onset = float(item["onset_sec"])
            notes: List[NoteEvent] = []

            if "notes" in item:
                for n in item["notes"]:
                    notes.append(
                        NoteEvent(
                            midi_pitch=int(n["midi"]),
                            duration_sec=float(n.get("duration_sec", 0.5)),
                            velocity=int(n.get("velocity", 80)),
                            articulation=ArticulationType(
                                n.get("articulation", ArticulationType.NORMAL.value)
                            ),
                            is_tied_continuation=bool(n.get("is_tied_continuation", False)),
                        )
                    )
            elif "pitches" in item:
                dur = float(item.get("duration_sec", 0.5))
                art = ArticulationType(
                    item.get("articulation", ArticulationType.NORMAL.value)
                )
                for pitch in item["pitches"]:
                    notes.append(
                        NoteEvent(
                            midi_pitch=int(pitch),
                            duration_sec=dur,
                            velocity=int(item.get("velocity", 80)),
                            articulation=art,
                        )
                    )
            else:
                raise ScoreParsingError(f"Evento {idx} sin claves 'notes' ni 'pitches'.")

            events.append(
                MusicalEvent(index=idx, onset_sec=onset, notes=tuple(notes))
            )

        return MusicalSequence.from_events(
            events=events,
            title=str(payload.get("title", path.stem)),
            tempo_bpm=float(payload.get("tempo_bpm", 120.0)),
            metadata={"source_file": str(path), "format": ".json"},
        )