import pytest

from config.instrument_settings import BiomechanicsConfig, GuitarPhysicsConfig
from src.domain.fretboard_map import FretboardMap
from src.domain.musical_event import MusicalEvent, MusicalSequence, NoteEvent
from src.graph.base_dag import (
    BaseDagBuilder,
    GraphConstructionError,
    PostureGenerator,
    SINK_NODE_ID,
    SOURCE_NODE_ID,
)


def _make_event(index: int, pitches: list[int], onset: float = 0.0) -> MusicalEvent:
    notes = tuple(NoteEvent(midi_pitch=p, duration_sec=0.5) for p in pitches)
    return MusicalEvent(index=index, onset_sec=onset, notes=notes)


class TestAnatomicalPruning:
    def setup_method(self) -> None:
        self.fretboard = FretboardMap(GuitarPhysicsConfig())
        self.default_bio = BiomechanicsConfig(max_hand_span=5, allow_barre=True)
        self.generator = PostureGenerator(self.fretboard, self.default_bio)

    def test_no_string_collisions_in_chords(self) -> None:
        # Acorde de C mayor abierto/móvil: C3(48), E3(52), G3(55), C4(60), E4(64)
        event = _make_event(0, [48, 52, 55, 60, 64])
        postures = self.generator.generate_layer(event)

        assert len(postures) > 0
        for posture in postures:
            strings = [a.string for a in posture.assignments]
            assert len(strings) == len(set(strings))

    def test_impossible_same_string_collision_raises_error(self) -> None:
        # E2(40) y F2(41) solo existen en la cuerda 6 en afinación estándar
        event = _make_event(0, [40, 41])
        with pytest.raises(GraphConstructionError):
            self.generator.generate_layer(event)

    def test_max_hand_span_enforced(self) -> None:
        strict_bio = BiomechanicsConfig(max_hand_span=3)
        strict_gen = PostureGenerator(self.fretboard, strict_bio)

        # C3(48) y B3(59)
        event = _make_event(0, [48, 59])
        postures = strict_gen.generate_layer(event)

        assert len(postures) > 0
        for posture in postures:
            assert posture.fret_span <= strict_bio.max_hand_span

    def test_excessive_stretch_chord_pruned_completely(self) -> None:
        strict_bio = BiomechanicsConfig(max_hand_span=2)
        strict_gen = PostureGenerator(self.fretboard, strict_bio)

        # F2(41, cuerda 6 traste 1) + C#3(49, cuerda 5 traste 4) -> apertura mínima de 3 trastes
        event = _make_event(0, [41, 49])
        with pytest.raises(GraphConstructionError):
            strict_gen.generate_layer(event)

    def test_open_strings_do_not_penalize_fret_span(self) -> None:
        # E2(40, cuerda 6 al aire) + G4(67, cuerda 1 traste 3)
        event = _make_event(0, [40, 67])
        postures = self.generator.generate_layer(event)

        open_and_third = [
            p
            for p in postures
            if any(a.string == 6 and a.fret == 0 and a.finger == 0 for a in p.assignments)
            and any(a.string == 1 and a.fret == 3 for a in p.assignments)
        ]
        assert len(open_and_third) > 0
        for p in open_and_third:
            assert p.fret_span == 0

    def test_all_open_strings_sets_zero_hand_position(self) -> None:
        # E2(40), A2(45), D3(50), G3(55), B3(59), E4(64)
        event = _make_event(0, [40, 45, 50, 55, 59, 64])
        postures = self.generator.generate_layer(event)

        zero_pos = [p for p in postures if p.hand_position == 0]
        assert len(zero_pos) == 1
        assert all(a.fret == 0 and a.finger == 0 for a in zero_pos[0].assignments)

    def test_finger_ordering_monotonicity(self) -> None:
        event = _make_event(0, [48, 52, 55])
        postures = self.generator.generate_layer(event)

        for posture in postures:
            fretted = posture.fretted_assignments
            for i in range(len(fretted)):
                for j in range(i + 1, len(fretted)):
                    a, b = fretted[i], fretted[j]
                    if a.fret < b.fret:
                        assert a.finger < b.finger
                    elif b.fret < a.fret:
                        assert b.finger < a.finger

    def test_barre_chord_support_and_toggle(self) -> None:
        # F mayor completo (cejilla traste 1): F2(41), C3(48), F3(53), A3(57), C4(60), F4(65)
        f_major = _make_event(0, [41, 48, 53, 57, 60, 65])

        barre_postures = self.generator.generate_layer(f_major)
        assert any(
            sum(1 for a in p.assignments if a.finger == 1 and a.fret == 1) >= 3
            for p in barre_postures
        )

        no_barre_gen = PostureGenerator(
            self.fretboard, BiomechanicsConfig(max_hand_span=5, allow_barre=False)
        )
        with pytest.raises(GraphConstructionError):
            no_barre_gen.generate_layer(f_major)

    def test_base_dag_is_strictly_layered_and_acyclic(self) -> None:
        seq = MusicalSequence.from_events(
            [
                _make_event(0, [40], onset=0.0),
                _make_event(1, [45, 52], onset=0.5),
                _make_event(2, [48], onset=1.0),
            ]
        )
        dag = BaseDagBuilder(self.fretboard, self.default_bio).build(seq)

        assert dag.num_layers == 3
        assert set(dag.adjacency[SOURCE_NODE_ID]) == {n.node_id for n in dag.layers[0]}

        for idx in range(dag.num_layers - 1):
            next_layer_ids = {n.node_id for n in dag.layers[idx + 1]}
            for node in dag.layers[idx]:
                assert set(dag.adjacency[node.node_id]) == next_layer_ids

        for node in dag.layers[-1]:
            assert dag.adjacency[node.node_id] == (SINK_NODE_ID,)
        assert dag.adjacency[SINK_NODE_ID] == ()