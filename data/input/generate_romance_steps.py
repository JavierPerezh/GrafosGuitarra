from pathlib import Path
from typing import Sequence, Tuple, Union
from music21 import chord, note, stream, tempo

PitchSpec = Union[int, Sequence[int]]


def _write_midi(
    events: Sequence[Tuple[PitchSpec, float]],
    output_path: Path,
    bpm: float = 100.0,
) -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=bpm))

    for pitches, q_len in events:
        elem = note.Note(pitches) if isinstance(pitches, int) else chord.Chord(list(pitches))
        elem.quarterLength = q_len
        part.append(elem)

    score.insert(0, part)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output_path))


def generate_romance_progression(base_dir: Path = Path("data/input")) -> None:
    t = 1.0 / 3.0

    # Compás 1 (9 eventos): arpegio estático en Em sobre B4 (traste 7 cuerda 1 o al aire cuerda 2)
    compas_1: list[Tuple[PitchSpec, float]] = [
        ([40, 71], t), (59, t), (55, t),
        (71, t), (59, t), (55, t),
        (71, t), (59, t), (55, t),
    ]

    # Compás 2 (9 eventos): descenso melódico B4 -> A4 -> G4
    compas_2: list[Tuple[PitchSpec, float]] = [
        ([40, 71], t), (59, t), (55, t),
        (69, t), (59, t), (55, t),
        (67, t), (59, t), (55, t),
    ]

    # Compás 3 (9 eventos): llegada a tónica G4 -> F#4 -> E4
    compas_3: list[Tuple[PitchSpec, float]] = [
        ([40, 67], t), (55, t), (59, t),
        (66, t), (59, t), (55, t),
        (64, t), (59, t), (55, t),
    ]

    _write_midi(compas_1, base_dir / "romance_step1_1compas.mid")
    _write_midi(compas_1 + compas_2, base_dir / "romance_step2_2compases.mid")
    _write_midi(compas_1 + compas_2 + compas_3, base_dir / "romance_step3_3compases.mid")


if __name__ == "__main__":
    generate_romance_progression()