"""Genera la primera célula del Romance Anónimo (4 eventos) para verificar
que la digitación clásica (e|7 + cuerdas al aire) entra en la frontera."""
from pathlib import Path
from music21 import chord, note, stream, tempo


def generate(out_dir: Path = Path("data/input")) -> Path:
    # Mismos valores exactos que usa generate_samples.py para el Romance
    t = 1.0 / 3.0
    events = [
        ([40, 71], t),  # E2 + B4  → esperado clásico: E|0 + e|7
        (59, t),        # G3       → G|0
        (55, t),        # B3       → B|0
        ([40,71], t),        # B4       → e|7
    ]

    score = stream.Score()
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=100.0))

    for pitches, q_len in events:
        elem = note.Note(pitches) if isinstance(pitches, int) else chord.Chord(pitches)
        elem.quarterLength = q_len
        part.append(elem)

    score.insert(0, part)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "romance_4notas.mid"
    score.write("midi", fp=str(out_path))
    print(f"Generado: {out_path}")
    return out_path


if __name__ == "__main__":
    generate()