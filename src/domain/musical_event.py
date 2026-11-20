from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, List, Optional, Tuple


class ArticulationType(str, Enum):
    NORMAL = "normal"
    LEGATO = "legato"      # Hammer-on / Pull-off
    SLIDE = "slide"
    STACCATO = "staccato"
    TIE = "tie"


@dataclass(frozen=True)
class NoteEvent:
    midi_pitch: int
    duration_sec: float
    velocity: int = 80
    articulation: ArticulationType = ArticulationType.NORMAL
    is_tied_continuation: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError(f"Altura MIDI fuera de rango [0, 127]: {self.midi_pitch}")
        if self.duration_sec <= 0.0:
            raise ValueError(f"La duración debe ser positiva: {self.duration_sec}")

    @property
    def frequency_hz(self) -> float:
        return 440.0 * (2.0 ** ((self.midi_pitch - 69) / 12.0))


@dataclass(frozen=True)
class MusicalEvent:
    index: int
    onset_sec: float
    notes: Tuple[NoteEvent, ...]
    ioi_sec: Optional[float] = None  # Tiempo hasta el siguiente evento t_{i+1}

    def __post_init__(self) -> None:
        if not self.notes:
            raise ValueError(f"El evento en índice {self.index} no contiene notas.")
        if self.onset_sec < 0.0:
            raise ValueError(f"El tiempo de inicio no puede ser negativo: {self.onset_sec}")

        # Ordena notas de grave a agudo y elimina alturas MIDI duplicadas simultáneas
        unique_by_pitch = {}
        for note in sorted(self.notes, key=lambda n: n.midi_pitch):
            if note.midi_pitch not in unique_by_pitch:
                unique_by_pitch[note.midi_pitch] = note
        object.__setattr__(self, "notes", tuple(unique_by_pitch.values()))

    @property
    def polyphony(self) -> int:
        return len(self.notes)

    @property
    def is_chord(self) -> bool:
        return len(self.notes) > 1

    @property
    def midi_pitches(self) -> Tuple[int, ...]:
        return tuple(n.midi_pitch for n in self.notes)

    @property
    def max_duration_sec(self) -> float:
        return max(n.duration_sec for n in self.notes)

    @property
    def has_legato(self) -> bool:
        return any(
            n.articulation in (ArticulationType.LEGATO, ArticulationType.SLIDE)
            for n in self.notes
        )


@dataclass(frozen=True)
class MusicalSequence:
    events: Tuple[MusicalEvent, ...]
    title: str = "Untitled"
    tempo_bpm: float = 120.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("La secuencia musical no puede estar vacía.")

        ordered = sorted(self.events, key=lambda e: e.onset_sec)
        normalized: List[MusicalEvent] = []

        for idx, event in enumerate(ordered):
            if idx > 0 and event.onset_sec <= ordered[idx - 1].onset_sec:
                raise ValueError(
                    f"Los eventos deben tener tiempos de inicio estrictamente crecientes: "
                    f"{ordered[idx - 1].onset_sec} >= {event.onset_sec}"
                )
            computed_ioi = (
                ordered[idx + 1].onset_sec - event.onset_sec
                if idx + 1 < len(ordered)
                else event.max_duration_sec
            )
            normalized.append(
                replace(event, index=idx, ioi_sec=computed_ioi)
            )

        object.__setattr__(self, "events", tuple(normalized))

    @classmethod
    def from_events(
        cls,
        events: Iterable[MusicalEvent],
        title: str = "Untitled",
        tempo_bpm: float = 120.0,
        metadata: Optional[dict] = None,
    ) -> "MusicalSequence":
        return cls(
            events=tuple(events),
            title=title,
            tempo_bpm=tempo_bpm,
            metadata=metadata or {},
        )

    def __len__(self) -> int:
        return len(self.events)