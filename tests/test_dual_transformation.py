import numpy as np

from config.instrument_settings import DEFAULT_SETTINGS
from src.domain.fretboard_map import FretboardMap
from src.domain.musical_event import MusicalEvent, MusicalSequence, NoteEvent
from src.graph.base_dag import BaseDagBuilder, FingerAssignment, PostureNode
from src.graph.cost_vector import CostVectorEvaluator
from src.graph.dual_graph import (
    DUAL_SINK_ID,
    DUAL_SOURCE_ID,
    DualGraphBuilder,
)


def _make_seq(pitches_per_step: list[list[int]], step_sec: float = 0.2) -> MusicalSequence:
    events = [
        MusicalEvent(
            index=idx,
            onset_sec=idx * step_sec,
            notes=tuple(NoteEvent(midi_pitch=p, duration_sec=step_sec) for p in pitches),
        )
        for idx, pitches in enumerate(pitches_per_step)
    ]
    return MusicalSequence.from_events(events)


class TestDualTransformation:
    def setup_method(self) -> None:
        self.fretboard = FretboardMap(DEFAULT_SETTINGS.physics)
        self.base_builder = BaseDagBuilder(self.fretboard, DEFAULT_SETTINGS.biomechanics)
        self.evaluator = CostVectorEvaluator(
            fretboard=self.fretboard,
            weights=DEFAULT_SETTINGS.cost_weights,
            biomechanics=DEFAULT_SETTINGS.biomechanics,
        )
        self.dual_builder = DualGraphBuilder(self.evaluator)

    def test_compound_node_and_edge_topology(self) -> None:
        seq = _make_seq([[48], [52], [55]])
        base_dag = self.base_builder.build(seq)
        dual_graph = self.dual_builder.build(base_dag, seq)

        assert len(dual_graph.layers) == base_dag.num_layers
        assert len(dual_graph.layers[0]) == len(base_dag.layers[0])
        assert len(dual_graph.layers[1]) == len(base_dag.layers[0]) * len(base_dag.layers[1])
        assert len(dual_graph.layers[2]) == len(base_dag.layers[1]) * len(base_dag.layers[2])

        # Verifica la regla de continuidad v == x en cada arista interna de E_hat
        for src_id, edges in dual_graph.adjacency.items():
            for edge in edges:
                assert edge.cost_vector.shape == (3,)
                assert np.all(edge.cost_vector >= 0.0)
                if src_id == DUAL_SOURCE_ID or edge.target_id == DUAL_SINK_ID:
                    continue
                src_cnode = dual_graph.nodes_by_id[src_id]
                dst_cnode = dual_graph.nodes_by_id[edge.target_id]
                assert dst_cnode.prev_posture is not None
                assert src_cnode.curr_posture.node_id == dst_cnode.prev_posture.node_id

    def test_reverse_adjacency_mirrors_forward_adjacency(self) -> None:
        seq = _make_seq([[45], [50], [55]])
        base_dag = self.base_builder.build(seq)
        dual_graph = self.dual_builder.build(base_dag, seq)

        forward_pairs = {
            (edge.source_id, edge.target_id)
            for edges in dual_graph.adjacency.values()
            for edge in edges
        }
        reverse_pairs = {
            (edge.source_id, edge.target_id)
            for edges in dual_graph.reverse_adjacency.values()
            for edge in edges
        }
        assert forward_pairs == reverse_pairs

    def test_second_order_inertia_penalizes_v_reversal_over_continuous_slide(self) -> None:
        ev_curr = MusicalEvent(0, 0.0, (NoteEvent(50, 0.2),), ioi_sec=0.2)
        ev_next = MusicalEvent(1, 0.2, (NoteEvent(55, 0.2),), ioi_sec=0.2)

        # Trayectoria continua: traste 2 -> traste 5 -> traste 8
        p_prev_cont = PostureNode("P0", 0, (FingerAssignment(5, 2, 1, 47),), hand_position=2)
        p_curr = PostureNode("P1", 1, (FingerAssignment(5, 5, 1, 50),), hand_position=5)
        p_next = PostureNode("P2", 2, (FingerAssignment(5, 8, 1, 53),), hand_position=8)

        # Cambio de dirección en V: traste 8 -> traste 5 -> traste 8
        p_prev_rev = PostureNode("P0_rev", 0, (FingerAssignment(5, 8, 1, 53),), hand_position=8)

        cost_continuous = self.evaluator.evaluate_second_order(
            p_prev_cont, p_curr, p_next, ev_curr, ev_next
        )
        cost_reversal = self.evaluator.evaluate_second_order(
            p_prev_rev, p_curr, p_next, ev_curr, ev_next
        )

        # W_bio (índice 0) debe penalizar el cambio súbito de dirección
        assert cost_reversal[0] > cost_continuous[0]

    def test_asymmetric_ascending_string_crossing_penalty(self) -> None:
        ev_curr = MusicalEvent(0, 0.0, (NoteEvent(55, 0.25),), ioi_sec=0.25)
        ev_next = MusicalEvent(1, 0.25, (NoteEvent(59, 0.25),), ioi_sec=0.25)

        # Cruce ascendente (cuerda 3 -> cuerda 2): dedo medio (2) a índice (1)
        curr_m = PostureNode("M", 0, (FingerAssignment(3, 5, 2, 60),), hand_position=4)
        next_i = PostureNode("I", 1, (FingerAssignment(2, 4, 1, 63),), hand_position=4)

        # Cruce ascendente (cuerda 3 -> cuerda 2): dedo anular (3) a medio (2)
        curr_a = PostureNode("A", 0, (FingerAssignment(3, 6, 3, 61),), hand_position=4)
        next_m = PostureNode("M2", 1, (FingerAssignment(2, 5, 2, 64),), hand_position=4)

        cost_m_to_i = self.evaluator.evaluate_first_order(curr_m, next_i, ev_curr, ev_next)
        cost_a_to_m = self.evaluator.evaluate_first_order(curr_a, next_m, ev_curr, ev_next)

        assert cost_m_to_i[0] > cost_a_to_m[0]

    def test_extract_posture_path_recovers_exact_sequence(self) -> None:
        seq = _make_seq([[40], [45], [50]])
        base_dag = self.base_builder.build(seq)
        dual_graph = self.dual_builder.build(base_dag, seq)

        c0 = dual_graph.layers[0][0]
        c1 = next(
            c
            for c in dual_graph.layers[1]
            if c.prev_posture and c.prev_posture.node_id == c0.curr_posture.node_id
        )
        c2 = next(
            c
            for c in dual_graph.layers[2]
            if c.prev_posture and c.prev_posture.node_id == c1.curr_posture.node_id
        )

        postures = dual_graph.extract_posture_path(
            [DUAL_SOURCE_ID, c0.node_id, c1.node_id, c2.node_id, DUAL_SINK_ID]
        )
        assert len(postures) == 3
        assert postures[0].node_id == c0.curr_posture.node_id
        assert postures[1].node_id == c1.curr_posture.node_id
        assert postures[2].node_id == c2.curr_posture.node_id