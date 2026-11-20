from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from config.instrument_settings import (
    BiomechanicsConfig,
    CostVectorWeights,
    DEFAULT_SETTINGS,
)
from src.domain.fretboard_map import FretboardMap, FretboardPosition
from src.domain.musical_event import ArticulationType, MusicalEvent
from src.graph.base_dag import FingerAssignment, PostureNode

CostVector = NDArray[np.float64]


def zero_cost_vector() -> CostVector:
    return np.zeros(3, dtype=np.float64)


class CostVectorEvaluator:
    def __init__(
        self,
        fretboard: FretboardMap,
        weights: CostVectorWeights = DEFAULT_SETTINGS.cost_weights,
        biomechanics: BiomechanicsConfig = DEFAULT_SETTINGS.biomechanics,
    ) -> None:
        self.fretboard = fretboard
        self.weights = weights
        self.biomechanics = biomechanics

    def evaluate_first_order(
        self,
        curr_node: PostureNode,
        next_node: PostureNode,
        curr_event: MusicalEvent,
        next_event: MusicalEvent,
    ) -> CostVector:
        return self._compute_transition_cost(
            prev_node=None,
            curr_node=curr_node,
            next_node=next_node,
            curr_event=curr_event,
            next_event=next_event,
        )

    def evaluate_second_order(
        self,
        prev_node: PostureNode,
        curr_node: PostureNode,
        next_node: PostureNode,
        curr_event: MusicalEvent,
        next_event: MusicalEvent,
    ) -> CostVector:
        return self._compute_transition_cost(
            prev_node=prev_node,
            curr_node=curr_node,
            next_node=next_node,
            curr_event=curr_event,
            next_event=next_event,
        )

    def _compute_transition_cost(
        self,
        prev_node: Optional[PostureNode],
        curr_node: PostureNode,
        next_node: PostureNode,
        curr_event: MusicalEvent,
        next_event: MusicalEvent,
    ) -> CostVector:
        ioi = max(curr_event.ioi_sec or curr_event.max_duration_sec, 0.05)
        pairs = self._pair_assignments(curr_node.assignments, next_node.assignments)

        w_bio = self._compute_biomechanical_cost(prev_node, curr_node, next_node, pairs, ioi)
        w_timbre = self._compute_timbral_cost(next_node, pairs)
        w_fluidez = self._compute_fluency_cost(curr_node, next_node, next_event, pairs, ioi)

        return np.array([w_bio, w_timbre, w_fluidez], dtype=np.float64)

    def _compute_biomechanical_cost(
    self,
    prev_node: Optional[PostureNode],
    curr_node: PostureNode,
    next_node: PostureNode,
    pairs: List[Tuple[FingerAssignment, FingerAssignment]],
    ioi: float,
    ) -> float:
        w = self.weights
        speed_factor = max(0.6, min(3.0, 0.25 / ioi))

        # --- Posiciones efectivas con "carry-over" a través de eventos al aire ---
        # Si curr_node es todo al aire, la mano sigue donde estaba en prev_node.
        from_pos = curr_node.hand_position
        if from_pos == 0 and prev_node is not None:
            from_pos = prev_node.hand_position
        to_pos = next_node.hand_position

        # --- 1. Shift longitudinal (efectivo, con carry-over) ---
        if from_pos > 0 and to_pos > 0:
            delta_curr = to_pos - from_pos
            shift_cost = (
                w.alpha_shift
                * (abs(delta_curr) ** w.gamma_acceleration)
                * speed_factor
            )
        else:
            delta_curr = 0
            shift_cost = 0.0

        # --- 2. Inercia de segundo orden (usando la misma convención de carry-over) ---
        inertia_cost = 0.0
        if prev_node is not None and delta_curr != 0:
            prev_from = prev_node.hand_position
            prev_to = curr_node.hand_position if curr_node.hand_position > 0 else from_pos
            if prev_from > 0 and prev_to > 0:
                delta_prev = prev_to - prev_from
                if delta_prev != 0:
                    if (delta_prev > 0) != (delta_curr > 0):
                        reversal = min(abs(delta_prev), abs(delta_curr))
                        inertia_cost = (
                            w.inertia_reversal_penalty * reversal * speed_factor
                        )
                    else:
                        shift_cost *= 0.85

        # --- 3. Coste de POSICIÓN por nota trasteada (ya estaba) ---
        position_cost = 0.0
        for a in curr_node.assignments:
            if not a.is_open and a.fret > w.comfort_fret_limit:
                position_cost += (
                    w.fret_position_penalty * (a.fret - w.comfort_fret_limit)
                )

                # --- 4. Slide SOLO misma cuerda (eliminado el cruce de cuerdas) ---
      
        slide_cost = 0.0
        for a_curr, a_next in pairs:
            if a_curr.string == a_next.string:
                if a_curr.is_open or a_next.is_open:
                    continue                      # ← nuevo
                fret_jump = abs(a_next.fret - a_curr.fret)
                if fret_jump > 0:
                    slide_cost = max(slide_cost, w.slide_penalty * fret_jump)

        # --- 5. Cruce de cuerdas SIN cuerdas al aire ---
        string_work = 0.0
        for a_curr, a_next in pairs:
            if a_curr.is_open or a_next.is_open:  # ← nuevo
                continue
            s_diff = abs(a_next.string - a_curr.string)
            fret_jump = abs(a_next.fret - a_curr.fret)
            if s_diff > 0 and fret_jump >= 3:
                string_work += (
                    w.beta_string_crossing * s_diff * min(1.0, fret_jump / 7.0)
                )

        # --- 6. Elongación intra-postura (ya estaba) ---
        excess_span = max(
            0, next_node.fret_span - self.biomechanics.relaxed_hand_span
        )
        stretch_cost = w.lambda_stretch * (excess_span ** 2)

        # --- 7. Fatiga digital (ya estaba) ---
        fw = self.biomechanics.finger_fatigue_weights
        fatigue_cost = (
            sum(fw.get(a.finger, 1.0)
                for a in next_node.assignments if not a.is_open)
            * 0.25
        )

        return float(
            shift_cost + inertia_cost + position_cost + slide_cost
            + string_work + stretch_cost + fatigue_cost
        )

    def _compute_timbral_cost(
        self,
        next_node: PostureNode,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
    ) -> float:
        w = self.weights
        timbre_cost = 0.0

        for a_curr, a_next in pairs:
            pos_curr = FretboardPosition(string=a_curr.string, fret=a_curr.fret)
            pos_next = FretboardPosition(string=a_next.string, fret=a_next.fret)
            dist = self.fretboard.timbral_distance(pos_curr, pos_next)
            timbre_cost += w.theta_string_change * dist

            # Penaliza cambiar de cuerda innecesariamente al repetir la misma altura
            if a_curr.midi_pitch == a_next.midi_pitch and a_curr.string != a_next.string:
                timbre_cost += 0.5 * w.theta_string_change

        # Inarmonicidad por trastes altos en cuerdas gruesas
        for a in next_node.assignments:
            if a.fret > 12:
                string_thickness_factor = a.string / self.fretboard.physics.num_strings
                timbre_cost += (
                    w.high_fret_inharmonicity * (a.fret - 12) * string_thickness_factor
                )

        return float(timbre_cost)

    def _compute_fluency_cost(
        self,
        curr_node: PostureNode,
        next_node: PostureNode,
        next_event: MusicalEvent,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
        ioi: float,
    ) -> float:
        w = self.weights
        disfluency = w.delta_legato_bonus + w.same_position_hold_bonus

        # Descuento por mantener anclaje homeostático de la mano
        if (
            curr_node.hand_position == next_node.hand_position
            and curr_node.hand_position > 0
        ):
            disfluency -= w.same_position_hold_bonus

        requested_articulations = {n.articulation for n in next_event.notes}
        wants_legato = ArticulationType.LEGATO in requested_articulations
        wants_slide = ArticulationType.SLIDE in requested_articulations

        if wants_legato or wants_slide:
            articulation_feasible = self._is_articulation_feasible(
                pairs=pairs,
                is_slide=wants_slide,
                ioi=ioi,
            )
            if articulation_feasible:
                disfluency -= w.delta_legato_bonus
            else:
                disfluency += w.delta_legato_bonus

        return float(max(0.0, disfluency))

    def _is_articulation_feasible(
        self,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
        is_slide: bool,
        ioi: float,
    ) -> bool:
        if not pairs or ioi > self.weights.max_legato_ioi_sec * 2.0:
            return False

        for a_curr, a_next in pairs:
            if a_curr.string != a_next.string or a_curr.fret == a_next.fret:
                return False
            if is_slide:
                if a_curr.is_open or a_next.is_open or a_curr.finger != a_next.finger:
                    return False
            else:
                # Hammer-on / Pull-off requiere misma cuerda y dedos distintos (o cuerda al aire)
                if not (a_curr.is_open or a_next.is_open) and a_curr.finger == a_next.finger:
                    return False
                if abs(a_next.fret - a_curr.fret) > self.biomechanics.max_hand_span:
                    return False
        return True

    @staticmethod
    def _effective_shift(pos_a: int, pos_b: int) -> int:
        if pos_a == 0 or pos_b == 0:
            return 0
        return pos_b - pos_a

    @staticmethod
    def _pair_assignments(
        curr_assignments: Tuple[FingerAssignment, ...],
        next_assignments: Tuple[FingerAssignment, ...],
    ) -> List[Tuple[FingerAssignment, FingerAssignment]]:
        pairs: List[Tuple[FingerAssignment, FingerAssignment]] = []
        used_next = set()

        # Empareja primero notas en la misma cuerda y luego por cercanía de altura
        for a_curr in curr_assignments:
            best_idx = None
            best_score = (10**9, 10**9)
            for idx, a_next in enumerate(next_assignments):
                if idx in used_next:
                    continue
                same_string_penalty = 0 if a_curr.string == a_next.string else 1
                pitch_diff = abs(a_curr.midi_pitch - a_next.midi_pitch)
                score = (same_string_penalty, pitch_diff)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                used_next.add(best_idx)
                pairs.append((a_curr, next_assignments[best_idx]))

        return pairs