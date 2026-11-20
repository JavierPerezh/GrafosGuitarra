from pathlib import Path
from typing import Sequence, Tuple, Union
from music21 import chord, note, stream, tempo

PitchSpec = Union[int, Sequence[int]]


def _write_midi(
    events: Sequence[Tuple[PitchSpec, float]],
    output_path: Path,
    bpm: float = 110.0,
) -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=bpm))

    for pitches, q_len in events:
        if isinstance(pitches, int):
            elem = note.Note(pitches)
        else:
            elem = chord.Chord(list(pitches))
        elem.quarterLength = q_len
        part.append(elem)

    score.insert(0, part)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output_path))


def generate_all(base_dir: Path = Path("data/input")) -> None:
    # 1. Frase melódica ascendente (dilema timbre vs. estabilidad de posición)
    scale_events: list[Tuple[PitchSpec, float]] = [
        (64, 0.5), (66, 0.5), (68, 0.5), (69, 0.5), (71, 1.0)
    ]
    _write_midi(scale_events, base_dir / "01_escala_pareto.mid", bpm=120.0)

    # 2. Progresión de acordes (abiertos + cejilla en F mayor)
    chord_events: list[Tuple[PitchSpec, float]] = [
        ([45, 52, 57, 60, 64], 1.0),  # Am
        ([41, 48, 53, 57, 60, 65], 1.0),  # F (cejilla)
        ([48, 52, 55, 60, 64], 1.0),  # C
        ([40, 47, 52, 56, 59, 64], 1.0),  # E7
    ]
    _write_midi(chord_events, base_dir / "02_acordes_cejilla.mid", bpm=90.0)

    # 3. Romance Anónimo (primeros 5 compases, arpegio en tresillos de corchea)
    t = 1.0 / 3.0
    romance_events: list[Tuple[PitchSpec, float]] = [
        # Compás 1 (Em, melodía en B4)
        ([40, 71], t), (55, t), (59, t),
        (71, t), (55, t), (59, t),
        (71, t), (55, t), (59, t),
        # Compás 2 (B4 -> A4 -> G4)
        ([40, 71], t), (55, t), (59, t),
        (69, t), (55, t), (59, t),
        (67, t), (55, t), (59, t),
        # Compás 3 (G4 -> F#4 -> E4)
        ([40, 67], t), (55, t), (59, t),
        (66, t), (55, t), (59, t),
        (64, t), (55, t), (59, t),
        # Compás 4 (E4 -> G4 -> B4)
        ([40, 64], t), (55, t), (59, t),
        (67, t), (55, t), (59, t),
        (71, t), (55, t), (59, t),
        # Compás 5 (clímax en E5, traste 12)
        ([40, 76], t), (55, t), (59, t),
        (76, t), (55, t), (59, t),
        (76, t), (55, t), (59, t),
    ]
    _write_midi(romance_events, base_dir / "03_romance_anonimo_intro.mid", bpm=100.0)


if __name__ == "__main__":
    generate_all()