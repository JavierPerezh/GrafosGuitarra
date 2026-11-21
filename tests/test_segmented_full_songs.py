from pathlib import Path
import time

from data.input.generate_full_songs import (
    build_bach_minuet_in_g,
    build_ode_to_joy_4_4,
    build_romance_anonimo_part1,
    _write_song_midi,
)
from main import run_pipeline
from src.ingestion.midi_xml_parser import MidiXmlScoreParser
from src.optimization.pareto_dominance import dominates
from src.optimization.segmented_solver import MeasureAwareSegmenter


def test_measure_segmentation_respects_barlines_and_max_10_events(tmp_path: Path) -> None:
    midi_path = tmp_path / "ode_4_4.mid"
    _write_song_midi(build_ode_to_joy_4_4(), midi_path, bpm=112.0, time_signature="4/4")

    parser = MidiXmlScoreParser()
    seq = parser.parse(midi_path)
    assert seq.metadata["time_signature"] == (4, 4)

    segmenter = MeasureAwareSegmenter()
    segments = segmenter.segment(seq)

    assert len(segments) >= 8
    total_events = 0
    for seg in segments:
        assert 1 <= len(seg.sequence.events) <= 10, (
            f"Segmento {seg.segment_index} excede 10 eventos: {len(seg.sequence.events)}"
        )
        total_events += len(seg.sequence.events)
    assert total_events == len(seq.events)


def test_full_romance_anonimo_144_events_fast_and_accurate(tmp_path: Path) -> None:
    midi_path = tmp_path / "romance_full.mid"
    _write_song_midi(
        build_romance_anonimo_part1(),
        midi_path,
        bpm=100.0,
        time_signature="3/4",
    )

    t0 = time.perf_counter()
    result, txt_file, _ = run_pipeline(
        input_path=midi_path,
        output_dir=tmp_path / "out_romance",
        max_solutions=5,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(result.solutions) >= 1
    assert len(result.solutions[0].posture_sequence) == 144
    assert elapsed_ms < 2500.0, f"Partición de 144 eventos tardó demasiado: {elapsed_ms:.2f} ms"

    # Verifica no dominancia global en el frente resultante
    for i, a in enumerate(result.solutions):
        for j, b in enumerate(result.solutions):
            if i != j:
                assert not dominates(a.total_cost, b.total_cost)

    # Verifica que el inicio del compás 3 (evento 18, G4) se toque en cuerda 1 traste 3 (e:3) y no B:8
    best = result.solutions[0]
    event_18_frets = {(a.string, a.fret) for a in best.posture_sequence[18].assignments}
    assert (1, 3) in event_18_frets, f"Se esperaba (1, 3) en compás 3, se obtuvo {event_18_frets}"
    print(f"\n[Romance Completo 144 eventos] Tiempo: {elapsed_ms:.2f} ms | Salida: {txt_file.name}")


def test_full_bach_minuet_and_ode_to_joy_4_4(tmp_path: Path) -> None:
    for name, builder, ts, bpm in [
        ("ode_4_4.mid", build_ode_to_joy_4_4, "4/4", 112.0),
        ("bach_3_4.mid", build_bach_minuet_in_g, "3/4", 116.0),
    ]:
        midi_path = tmp_path / name
        _write_song_midi(builder(), midi_path, bpm=bpm, time_signature=ts)
        t0 = time.perf_counter()
        result, _, _ = run_pipeline(
            input_path=midi_path,
            output_dir=tmp_path / f"out_{name}",
            max_solutions=4,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert len(result.solutions) >= 1
        assert elapsed_ms < 2000.0
        print(f"\n[{name}] Eventos: {len(result.solutions[0].posture_sequence)} | Tiempo: {elapsed_ms:.2f} ms")
