from typing import Dict, List, Optional, Tuple
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
        initial_posture: Optional[PostureNode] = None,
        initial_fretted_posture: Optional[PostureNode] = None,
    ) -> None:
        self.fretboard = fretboard
        self.weights = weights
        self.biomechanics = biomechanics
        self.initial_posture = initial_posture
        self.initial_fretted_posture = initial_fretted_posture

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
        cost = self._compute_transition_cost(
            prev_node=prev_node,
            curr_node=curr_node,
            next_node=next_node,
            curr_event=curr_event,
            next_event=next_event,
        )

        if curr_event.index == 1:
            if self.initial_posture is None:
                init_bio, init_timbre = self._compute_initial_posture_cost(prev_node)
                cost[0] += init_bio
                cost[1] += init_timbre
            else:
                boundary_cost = self._compute_transition_cost(
                    prev_node=self.initial_fretted_posture,
                    curr_node=self.initial_posture,
                    next_node=prev_node,
                    curr_event=curr_event,
                    next_event=curr_event,
                )
                cost = cost + boundary_cost

        return cost

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

        w_bio = self._compute_biomechanical_cost(
            prev_node, curr_node, next_node, next_event, pairs, ioi
        )
        w_timbre = self._compute_timbral_cost(next_node, pairs)
        w_fluidez = self._compute_fluency_cost(
            prev_node, curr_node, next_node, next_event, pairs, ioi
        )

        return np.array([w_bio, w_timbre, w_fluidez], dtype=np.float64)

    def _compute_initial_posture_cost(self, node: PostureNode) -> Tuple[float, float]:
        w = self.weights
        position_cost = 0.0
        timbre_cost = 0.0
        for a in node.assignments:
            if not a.is_open and a.fret > w.comfort_fret_limit:
                excess = a.fret - w.comfort_fret_limit
                string_penalty = 1.5 if (a.fret > 12 and a.string >= 4) else 1.0
                position_cost += w.fret_position_penalty * excess * string_penalty
            if a.fret > 12:
                string_thickness_factor = (a.string / self.fretboard.physics.num_strings) ** 1.5
                timbre_cost += (
                    w.high_fret_inharmonicity * (a.fret - 12) * string_thickness_factor
                )

        excess_span = max(0, node.fret_span - self.biomechanics.relaxed_hand_span)
        stretch_cost = w.lambda_stretch * (excess_span ** 2)
        if node.fret_span > self.biomechanics.max_hand_span:
            stretch_cost += 18.0 * ((node.fret_span - self.biomechanics.max_hand_span) ** 2)

        fw = self.biomechanics.finger_fatigue_weights
        fatigue_cost = (
            sum(fw.get(a.finger, 1.0) for a in node.assignments if not a.is_open) * 0.25
        )
        initial_entry = 0.35 * max(0, node.hand_position - 5) if node.hand_position > 0 else 0.0
        return float(position_cost + stretch_cost + fatigue_cost + initial_entry), float(timbre_cost)

    def _compute_biomechanical_cost(
        self,
        prev_node: Optional[PostureNode],
        curr_node: PostureNode,
        next_node: PostureNode,
        next_event: MusicalEvent,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
        ioi: float,
    ) -> float:
        w = self.weights
        speed_factor = max(0.6, min(3.0, 0.25 / ioi))

        from_pos = curr_node.hand_position
        if from_pos == 0 and prev_node is not None and prev_node.hand_position > 0:
            from_pos = prev_node.hand_position
        to_pos = next_node.hand_position

        # 1. Desplazamiento longitudinal
        if from_pos > 0 and to_pos > 0:
            delta_curr = to_pos - from_pos
            shift_cost = (
                w.alpha_shift
                * (abs(delta_curr) ** w.gamma_acceleration)
                * speed_factor
            )
        elif from_pos == 0 and to_pos > 0:
            delta_curr = 0
            shift_cost = 0.12 * max(0, to_pos - 4)
        else:
            delta_curr = 0
            shift_cost = 0.0

        # 2. Inercia de segundo orden
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

        # 3. Penalización por registro alto en la postura destino
        position_cost = 0.0
        for a in next_node.assignments:
            if not a.is_open and a.fret > w.comfort_fret_limit:
                excess = a.fret - w.comfort_fret_limit
                string_penalty = 1.5 if (a.fret > 12 and a.string >= 4) else 1.0
                position_cost += w.fret_position_penalty * excess * string_penalty

        # 4. Arrastre longitudinal y coherencia de digitación en la misma cuerda
        note_art_by_pitch: Dict[int, ArticulationType] = {
            n.midi_pitch: n.articulation for n in next_event.notes
        }
        slide_cost = 0.0
        finger_order_cost = 0.0
        for a_curr, a_next in pairs:
            if a_curr.string == a_next.string and not (a_curr.is_open or a_next.is_open):
                fret_diff = a_next.fret - a_curr.fret
                fret_jump = abs(fret_diff)
                if fret_jump > 0:
                    slide_cost = max(slide_cost, w.slide_penalty * fret_jump)
                    art = note_art_by_pitch.get(a_next.midi_pitch, ArticulationType.NORMAL)
                    if (fret_diff > 0 and a_next.finger < a_curr.finger) or (
                        fret_diff < 0 and a_next.finger > a_curr.finger
                    ):
                        finger_order_cost += 0.85 * speed_factor
                    elif a_next.finger == a_curr.finger and art != ArticulationType.SLIDE:
                        finger_order_cost += 0.55 * speed_factor

        # 5. Trabajo transversal y asimetría de dedos entre cuerdas
        string_work = 0.0
        for a_curr, a_next in pairs:
            if a_curr.is_open or a_next.is_open:
                continue
            s_diff = a_next.string - a_curr.string
            if s_diff != 0:
                string_work += w.beta_string_crossing * abs(s_diff) * speed_factor
                if s_diff < 0 and a_curr.finger in (2, 3) and a_next.finger == 1:
                    string_work += 0.45 * w.beta_string_crossing * speed_factor
                elif a_curr.finger == a_next.finger:
                    roll_factor = 0.5 if (a_curr.fret == a_next.fret and abs(s_diff) == 1) else 1.0
                    string_work += w.same_finger_hop_penalty * roll_factor * speed_factor

        # 6. Penalización por cuerdas interiores saltadas en acordes (>= 3 notas)
        skip_cost = 0.0
        if len(next_node.assignments) >= 3:
            used_strings = sorted(a.string for a in next_node.assignments)
            interior_gaps = (used_strings[-1] - used_strings[0] + 1) - len(used_strings)
            if interior_gaps > 0:
                skip_cost = w.interior_string_skip_penalty * interior_gaps

        # 7. Elongación intra-postura y castigo por hiper-extensión excepcional
        excess_span = max(0, next_node.fret_span - self.biomechanics.relaxed_hand_span)
        stretch_cost = w.lambda_stretch * (excess_span ** 2)
        if next_node.fret_span > self.biomechanics.max_hand_span:
            stretch_cost += 18.0 * ((next_node.fret_span - self.biomechanics.max_hand_span) ** 2)

        # 8. Fatiga digital acumulada
        fw = self.biomechanics.finger_fatigue_weights
        fatigue_cost = (
            sum(fw.get(a.finger, 1.0) for a in next_node.assignments if not a.is_open)
            * 0.25
        )

        return float(
            shift_cost
            + inertia_cost
            + position_cost
            + slide_cost
            + finger_order_cost
            + string_work
            + skip_cost
            + stretch_cost
            + fatigue_cost
        )

    def _compute_timbral_cost(
        self,
        next_node: PostureNode,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
    ) -> float:
        w = self.weights
        timbre_cost = 0.0

        for a_curr, a_next in pairs:
            melodic_interval = abs(a_next.midi_pitch - a_curr.midi_pitch)
            is_voice_switch = melodic_interval > 5

            if is_voice_switch and (a_curr.is_open or a_next.is_open):
                if a_curr.string == a_next.string:
                    timbre_cost += w.voice_choke_timbre_penalty
                else:
                    timbre_cost += w.open_string_timbre_penalty * 0.5
                continue

            pos_curr = FretboardPosition(string=a_curr.string, fret=a_curr.fret)
            pos_next = FretboardPosition(string=a_next.string, fret=a_next.fret)
            dist = self.fretboard.timbral_distance(pos_curr, pos_next)
            voice_change_factor = 0.35 if is_voice_switch else 1.0
            timbre_cost += w.theta_string_change * dist * voice_change_factor

            if is_voice_switch and a_curr.string == a_next.string:
                timbre_cost += w.voice_choke_timbre_penalty

            fret_jump = abs(a_next.fret - a_curr.fret)
            if fret_jump > 0:
                timbre_cost += w.fret_shift_timbre_factor * fret_jump * voice_change_factor

            if a_curr.is_open != a_next.is_open:
                timbre_cost += w.open_string_timbre_penalty * voice_change_factor

            if a_curr.midi_pitch == a_next.midi_pitch and a_curr.string != a_next.string:
                timbre_cost += 0.5 * w.theta_string_change * voice_change_factor

        if len(next_node.assignments) >= 3:
            used_strings = sorted(a.string for a in next_node.assignments)
            interior_gaps = (used_strings[-1] - used_strings[0] + 1) - len(used_strings)
            if interior_gaps > 0:
                timbre_cost += 0.35 * interior_gaps

        for a in next_node.assignments:
            if a.fret > 12:
                string_thickness_factor = (a.string / self.fretboard.physics.num_strings) ** 1.5
                timbre_cost += (
                    w.high_fret_inharmonicity * (a.fret - 12) * string_thickness_factor
                )

        return float(timbre_cost)

    def _compute_fluency_cost(
        self,
        prev_node: Optional[PostureNode],
        curr_node: PostureNode,
        next_node: PostureNode,
        next_event: MusicalEvent,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
        ioi: float,
    ) -> float:
        w = self.weights
        from_pos = curr_node.hand_position
        if from_pos == 0 and prev_node is not None:
            from_pos = prev_node.hand_position
        to_pos = next_node.hand_position

        if to_pos == 0:
            disfluency = max(0.0, w.same_position_hold_bonus - w.open_string_relief_bonus)
        elif from_pos == 0:
            disfluency = max(0.0, 0.10 * max(0, to_pos - 4))
        elif from_pos == to_pos:
            disfluency = 0.0
        else:
            shift_dist = abs(to_pos - from_pos)
            disfluency = w.same_position_hold_bonus * (0.75 + 0.25 * shift_dist)

        if any(a.is_open for a in next_node.assignments) and disfluency > 0.0:
            disfluency = max(0.0, disfluency - w.open_string_relief_bonus)

        note_art_by_pitch: Dict[int, ArticulationType] = {
            n.midi_pitch: n.articulation for n in next_event.notes
        }
        requested_articulations = set(note_art_by_pitch.values())
        wants_legato = ArticulationType.LEGATO in requested_articulations
        wants_slide = ArticulationType.SLIDE in requested_articulations

        if wants_legato or wants_slide:
            articulation_feasible = self._is_articulation_feasible(
                pairs=pairs,
                note_art_by_pitch=note_art_by_pitch,
                ioi=ioi,
            )
            if not articulation_feasible:
                disfluency += w.delta_legato_bonus * 2.0

        return float(max(0.0, disfluency))

    def _is_articulation_feasible(
        self,
        pairs: List[Tuple[FingerAssignment, FingerAssignment]],
        note_art_by_pitch: Dict[int, ArticulationType],
        ioi: float,
    ) -> bool:
        if not pairs or ioi > self.weights.max_legato_ioi_sec * 2.0:
            return False

        articulated_pairs = [
            (a_curr, a_next, note_art_by_pitch.get(a_next.midi_pitch, ArticulationType.NORMAL))
            for a_curr, a_next in pairs
            if note_art_by_pitch.get(a_next.midi_pitch)
            in (ArticulationType.LEGATO, ArticulationType.SLIDE)
        ]
        if not articulated_pairs:
            return False

        for a_curr, a_next, art in articulated_pairs:
            if a_curr.string != a_next.string or a_curr.fret == a_next.fret:
                return False
            if art == ArticulationType.SLIDE:
                if a_curr.is_open or a_next.is_open or a_curr.finger != a_next.finger:
                    return False
            else:
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

        for a_curr in curr_assignments:
            best_idx = None
            best_score = (10**9, 10**9)
            for idx, a_next in enumerate(next_assignments):
                if idx in used_next:
                    continue
                pitch_diff = abs(a_curr.midi_pitch - a_next.midi_pitch)
                string_diff = abs(a_curr.string - a_next.string)
                # No emparejar salto directo bajo <-> melodía como si fuera el mismo dedo
                if pitch_diff >= 7 and string_diff >= 3:
                    continue
                same_string_penalty = 0 if a_curr.string == a_next.string else 1
                score = (same_string_penalty, pitch_diff)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                used_next.add(best_idx)
                pairs.append((a_curr, next_assignments[best_idx]))

        return pairs