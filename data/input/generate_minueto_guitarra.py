from pathlib import Path
from typing import Sequence, Tuple, Union
from music21 import chord, meter, note, stream, tempo

PitchSpec = Union[int, Sequence[int]]


def generate_minueto_guitarra(
    output_path: Path = Path("data/input/cancion_03_minueto_bach_guitarra.mid"),
    bpm: float = 110.0,
) -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(meter.TimeSignature("3/4"))
    part.append(tempo.MetronomeMark(number=bpm))

    # Primeros 8 compases de BWV Anh. 114 en registro estándar de guitarra clásica
    events: list[Tuple[PitchSpec, float]] = [
        # Compás 1: G3 + D5 -> G4 A4 B4 C5
        ([55, 74], 1.0), (67, 0.5), (69, 0.5), (71, 0.5), (72, 0.5),
        # Compás 2: G3 + D5 -> G4 -> G4
        ([55, 74], 1.0), (67, 1.0), (67, 1.0),
        # Compás 3: C3 + E5 -> C5 D5 E5 F#5
        ([48, 76], 1.0), (72, 0.5), (74, 0.5), (76, 0.5), (78, 0.5),
        # Compás 4: B3 + G5 -> G4 -> G4
        ([59, 79], 1.0), (67, 1.0), (67, 1.0),
        # Compás 5: A3 + C5 -> D5 C5 B4 A4
        ([57, 72], 1.0), (74, 0.5), (72, 0.5), (71, 0.5), (69, 0.5),
        # Compás 6: B3 + B4 -> C5 B4 A4 G4
        ([59, 71], 1.0), (72, 0.5), (71, 0.5), (69, 0.5), (67, 0.5),
        # Compás 7: D3 + A4 -> B4 A4 G4 F#4
        ([50, 69], 1.0), (71, 0.5), (69, 0.5), (67, 0.5), (66, 0.5),
        # Compás 8: G3 + G4 (blanca con puntillo)
        ([55, 67], 3.0),
    ]

    for pitches, q_len in events:
        elem = note.Note(pitches) if isinstance(pitches, int) else chord.Chord(list(pitches))
        elem.quarterLength = q_len
        part.append(elem)

    score.insert(0, part)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(output_path))
    print(f"Archivo generado en: {output_path}")


if __name__ == "__main__":
    generate_minueto_guitarra()