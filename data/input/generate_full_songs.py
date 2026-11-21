from pathlib import Path
from typing import List, Sequence, Tuple, Union
from music21 import chord, meter, note, stream, tempo

PitchSpec = Union[int, Sequence[int]]


def _write_song_midi(
    events: Sequence[Tuple[PitchSpec, float]],
    output_path: Path,
    bpm: float,
    time_signature: str = "4/4",
) -> None:
    score = stream.Score()
    part = stream.Part()
    part.append(meter.TimeSignature(time_signature))
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


def build_romance_anonimo_part1() -> List[Tuple[PitchSpec, float]]:
    """16 compases completos de Romance Anónimo (Parte I en Mi menor, 3/4, 144 eventos)."""
    t = 1.0 / 3.0  # Tresillo de corchea (9 eventos por compás)
    measures: List[Tuple[int, int, int, Tuple[int, int, int]]] = [
        # Compás 1-2: Em sobre B4
        (40, 55, 59, (71, 71, 71)),
        (40, 55, 59, (71, 71, 71)),
        # Compás 3-4: descenso B4-A4-G4-F#4-E4
        (40, 55, 59, (67, 69, 71)),
        (40, 55, 59, (67, 66, 64)),
        # Compás 5: ascenso E4-G4-B4
        (40, 55, 59, (64, 67, 71)),
        # Compás 6-7: clímax en E5 (traste 12)
        (40, 55, 59, (76, 76, 76)),
        (40, 55, 59, (76, 76, 76)),
        # Compás 8: descenso Am (E5-D5-C5)
        (45, 57, 60, (76, 74, 72)),
        # Compás 9-10: Am sobre C5 -> B4 -> A4
        (45, 57, 60, (72, 72, 72)),
        (45, 57, 60, (72, 71, 69)),
        # Compás 11: Em sobre B4-C5-B4
        (40, 55, 59, (71, 72, 71)),
        # Compás 12-13: B7 (D5-C5-B4-A4-G4-F#4)
        (47, 54, 57, (74, 72, 71)),
        (47, 54, 57, (69, 67, 66)),
        # Compás 14-16: cadencia final en Em (E4)
        (40, 55, 59, (64, 64, 64)),
        (40, 55, 59, (64, 64, 64)),
        (40, 55, 59, (64, 64, 64)),
    ]

    events: List[Tuple[PitchSpec, float]] = []
    for bass, mid1, mid2, melody_triplet in measures:
        for beat_idx, mel in enumerate(melody_triplet):
            if beat_idx == 0:
                events.append(([bass, mel], t))
            else:
                events.append((mel, t))
            events.append((mid1, t))
            events.append((mid2, t))
    return events


def build_ode_to_joy_4_4() -> List[Tuple[PitchSpec, float]]:
    """16 compases completos del Himno a la Alegría en 4/4 con bajo en tiempo 1 (64 eventos)."""
    # Cada tupla es (pitches, quarterLength)
    return [
        # Compás 1 (C)
        ([48, 64], 1.0), (64, 1.0), (65, 1.0), (67, 1.0),
        # Compás 2 (G)
        ([43, 67], 1.0), (65, 1.0), (64, 1.0), (62, 1.0),
        # Compás 3 (C)
        ([48, 60], 1.0), (60, 1.0), (62, 1.0), (64, 1.0),
        # Compás 4 (G)
        ([43, 64], 1.5), (62, 0.5), (62, 2.0),
        # Compás 5 (C)
        ([48, 64], 1.0), (64, 1.0), (65, 1.0), (67, 1.0),
        # Compás 6 (G)
        ([43, 67], 1.0), (65, 1.0), (64, 1.0), (62, 1.0),
        # Compás 7 (C)
        ([48, 60], 1.0), (60, 1.0), (62, 1.0), (64, 1.0),
        # Compás 8 (G -> C)
        ([43, 62], 1.5), (60, 0.5), ([48, 60], 2.0),
        # Compás 9 (G)
        ([43, 62], 1.0), (62, 1.0), (64, 1.0), (60, 1.0),
        # Compás 10 (G)
        ([43, 62], 1.0), (64, 0.5), (65, 0.5), (64, 1.0), (60, 1.0),
        # Compás 11 (G)
        ([43, 62], 1.0), (64, 0.5), (65, 0.5), (64, 1.0), (62, 1.0),
        # Compás 12 (C -> G)
        ([48, 60], 1.0), (62, 1.0), ([43, 55], 2.0),
        # Compás 13 (C)
        ([48, 64], 1.0), (64, 1.0), (65, 1.0), (67, 1.0),
        # Compás 14 (G)
        ([43, 67], 1.0), (65, 1.0), (64, 1.0), (62, 1.0),
        # Compás 15 (C)
        ([48, 60], 1.0), (60, 1.0), (62, 1.0), (64, 1.0),
        # Compás 16 (C cadencia final)
        ([43, 62], 1.5), (60, 0.5), ([48, 52, 55, 60], 2.0),
    ]


def build_bach_minuet_in_g() -> List[Tuple[PitchSpec, float]]:
    """Primera sección completa (16 compases, 3/4) del Minueto en Sol Mayor BWV Anh. 114."""
    return [
        # Compás 1-2
        ([43, 74], 1.0), (67, 0.5), (69, 0.5), (71, 0.5), (72, 0.5),
        ([43, 74], 1.0), (67, 1.0), (67, 1.0),
        # Compás 3-4
        ([48, 76], 1.0), (72, 0.5), (74, 0.5), (76, 0.5), (78, 0.5),
        ([43, 79], 1.0), (67, 1.0), (67, 1.0),
        # Compás 5-6
        ([48, 72], 1.0), (74, 0.5), (72, 0.5), (71, 0.5), (69, 0.5),
        ([47, 71], 1.0), (72, 0.5), (71, 0.5), (69, 0.5), (67, 0.5),
        # Compás 7-8
        ([45, 66], 1.0), (67, 0.5), (69, 0.5), (71, 0.5), (67, 0.5),
        ([50, 69], 3.0),
        # Compás 9-10
        ([43, 74], 1.0), (67, 0.5), (69, 0.5), (71, 0.5), (72, 0.5),
        ([43, 74], 1.0), (67, 1.0), (67, 1.0),
        # Compás 11-12
        ([48, 76], 1.0), (72, 0.5), (74, 0.5), (76, 0.5), (78, 0.5),
        ([43, 79], 1.0), (67, 1.0), (67, 1.0),
        # Compás 13-14
        ([48, 72], 1.0), (74, 0.5), (72, 0.5), (71, 0.5), (69, 0.5),
        ([47, 71], 1.0), (72, 0.5), (71, 0.5), (69, 0.5), (67, 0.5),
        # Compás 15-16
        ([50, 69], 1.0), (71, 0.5), (69, 0.5), (67, 0.5), (66, 0.5),
        ([43, 55, 59, 67], 3.0),
    ]


def generate_all_songs(base_dir: Path = Path("data/input")) -> None:
    _write_song_midi(
        build_romance_anonimo_part1(),
        base_dir / "cancion_01_romance_anonimo_parte1.mid",
        bpm=100.0,
        time_signature="3/4",
    )
    _write_song_midi(
        build_ode_to_joy_4_4(),
        base_dir / "cancion_02_himno_alegria_4_4.mid",
        bpm=112.0,
        time_signature="4/4",
    )
    _write_song_midi(
        build_bach_minuet_in_g(),
        base_dir / "cancion_03_minueto_bach_sol_mayor.mid",
        bpm=116.0,
        time_signature="3/4",
    )
    print("Canciones completas generadas en data/input/:")
    print(" - cancion_01_romance_anonimo_parte1.mid (16 compases, 3/4, 144 eventos)")
    print(" - cancion_02_himno_alegria_4_4.mid (16 compases, 4/4, 64 eventos)")
    print(" - cancion_03_minueto_bach_sol_mayor.mid (16 compases, 3/4, 76 eventos)")


if __name__ == "__main__":
    generate_all_songs()
