from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from config.instrument_settings import BiomechanicsConfig, DEFAULT_SETTINGS
from src.domain.fretboard_map import FretboardMap, FretboardPosition
from src.domain.musical_event import MusicalEvent, MusicalSequence

SOURCE_NODE_ID = "SOURCE"
SINK_NODE_ID = "SINK"


class GraphConstructionError(Exception):
    """Error producido cuando un evento musical no admite posturas viables."""


@dataclass(frozen=True, order=True)
class FingerAssignment:
    string: int
    fret: int
    finger: int      # 0 = cuerda al aire, 1..4 = índice..meñique
    midi_pitch: int

    @property
    def is_open(self) -> bool:
        return self.fret == 0


@dataclass(frozen=True)
class PostureNode:
    node_id: str
    layer_index: int
    assignments: Tuple[FingerAssignment, ...]
    hand_position: int  # p_mano (0 si todas las cuerdas son al aire)

    @property
    def fretted_assignments(self) -> Tuple[FingerAssignment, ...]:
        return tuple(a for a in self.assignments if not a.is_open)

    @property
    def fret_span(self) -> int:
        frets = [a.fret for a in self.assignments if not a.is_open]
        if len(frets) <= 1:
            return 0
        return max(frets) - min(frets)


@dataclass(frozen=True)
class LayeredDAG:
    layers: Tuple[Tuple[PostureNode, ...], ...]
    nodes_by_id: Dict[str, PostureNode]
    adjacency: Dict[str, Tuple[str, ...]]
    source_id: str = SOURCE_NODE_ID
    sink_id: str = SINK_NODE_ID

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_posture_nodes(self) -> int:
        return len(self.nodes_by_id)

    @property
    def num_edges(self) -> int:
        return sum(len(targets) for targets in self.adjacency.values())


class PostureGenerator:
    def __init__(
        self,
        fretboard: FretboardMap,
        biomechanics: BiomechanicsConfig = DEFAULT_SETTINGS.biomechanics,
    ) -> None:
        self.fretboard = fretboard
        self.biomechanics = biomechanics

    def generate_layer(self, event: MusicalEvent) -> Tuple[PostureNode, ...]:
        if event.polyphony > self.biomechanics.max_simultaneous_notes:
            raise GraphConstructionError(
                f"Evento {event.index} excede el máximo de notas simultáneas ({event.polyphony})."
            )

        candidate_lists: List[Tuple[int, Tuple[FretboardPosition, ...]]] = [
            (note.midi_pitch, self.fretboard.get_positions(note.midi_pitch))
            for note in event.notes
        ]

        valid_combinations: List[List[Tuple[int, FretboardPosition]]] = []
        self._search_positions(
            idx=0,
            candidate_lists=candidate_lists,
            used_strings=set(),
            current=[],
            min_fret=10**9,
            max_fret=-1,
            span_limit=self.biomechanics.max_hand_span,
            results=valid_combinations,
        )

        # Fallback controlado: solo si el evento polifónico es físicamente imposible
        # dentro de max_hand_span (ej. bajo grave fijo + nota muy aguda), admite
        # únicamente las combinaciones de mínima apertura posible.
        if not valid_combinations and event.polyphony > 1:
            relaxed: List[List[Tuple[int, FretboardPosition]]] = []
            self._search_positions(
                idx=0,
                candidate_lists=candidate_lists,
                used_strings=set(),
                current=[],
                min_fret=10**9,
                max_fret=-1,
                span_limit=9,
                results=relaxed,
            )
            if relaxed:
                def combo_span(c: List[Tuple[int, FretboardPosition]]) -> int:
                    frets = [p.fret for _, p in c if not p.is_open]
                    return (max(frets) - min(frets)) if len(frets) > 1 else 0

                min_possible = min(combo_span(c) for c in relaxed)
                valid_combinations = [c for c in relaxed if combo_span(c) == min_possible]

        postures: List[PostureNode] = []
        for combo in valid_combinations:
            ordered_combo = sorted(combo, key=lambda item: item[1].string)
            fingerings = self._generate_fingerings(ordered_combo)
            for assignments in fingerings:
                p_hand = self._compute_hand_position(assignments)
                node_id = f"L{event.index}_N{len(postures)}"
                postures.append(
                    PostureNode(
                        node_id=node_id,
                        layer_index=event.index,
                        assignments=assignments,
                        hand_position=p_hand,
                    )
                )

        if not postures:
            raise GraphConstructionError(
                f"No existen posturas anatómicamente viables para el evento {event.index} "
                f"(MIDI: {event.midi_pitches})."
            )

        return tuple(postures)

    def _search_positions(
        self,
        idx: int,
        candidate_lists: Sequence[Tuple[int, Tuple[FretboardPosition, ...]]],
        used_strings: Set[int],
        current: List[Tuple[int, FretboardPosition]],
        min_fret: int,
        max_fret: int,
        span_limit: int,
        results: List[List[Tuple[int, FretboardPosition]]],
    ) -> None:
        if idx == len(candidate_lists):
            results.append(list(current))
            return

        midi_pitch, positions = candidate_lists[idx]
        for pos in positions:
            if pos.string in used_strings:
                continue

            next_min = min_fret
            next_max = max_fret
            if not pos.is_open:
                next_min = min(min_fret, pos.fret)
                next_max = max(max_fret, pos.fret)
                if next_max - next_min > span_limit:
                    continue

            used_strings.add(pos.string)
            current.append((midi_pitch, pos))
            self._search_positions(
                idx + 1,
                candidate_lists,
                used_strings,
                current,
                next_min,
                next_max,
                span_limit,
                results,
            )
            current.pop()
            used_strings.remove(pos.string)

    def _generate_fingerings(
        self,
        ordered_combo: Sequence[Tuple[int, FretboardPosition]],
    ) -> List[Tuple[FingerAssignment, ...]]:
        results: List[Tuple[FingerAssignment, ...]] = []
        available = self.biomechanics.available_fingers
        open_finger = self.biomechanics.open_string_finger

        def backtrack(idx: int, current: List[FingerAssignment]) -> None:
            if idx == len(ordered_combo):
                if self._is_anatomically_valid(current):
                    results.append(tuple(current))
                return

            midi_pitch, pos = ordered_combo[idx]
            if pos.is_open:
                current.append(
                    FingerAssignment(
                        string=pos.string,
                        fret=0,
                        finger=open_finger,
                        midi_pitch=midi_pitch,
                    )
                )
                backtrack(idx + 1, current)
                current.pop()
            else:
                for finger in available:
                    current.append(
                        FingerAssignment(
                            string=pos.string,
                            fret=pos.fret,
                            finger=finger,
                            midi_pitch=midi_pitch,
                        )
                    )
                    backtrack(idx + 1, current)
                    current.pop()

        backtrack(0, [])
        return results

    def _is_anatomically_valid(self, assignments: Sequence[FingerAssignment]) -> bool:
        fretted = [a for a in assignments if not a.is_open]
        if len(fretted) <= 1:
            return True

        min_fretted = min(a.fret for a in fretted)
        max_fretted = max(a.fret for a in fretted)
        is_fallback_dyad = (
            len(fretted) == 2 and (max_fretted - min_fretted) > self.biomechanics.max_hand_span
        )
        max_extra_stretch = max(1, self.biomechanics.max_hand_span - 3)
        by_finger: Dict[int, List[FingerAssignment]] = {}

        for i in range(len(fretted)):
            a = fretted[i]
            by_finger.setdefault(a.finger, []).append(a)
            for j in range(i + 1, len(fretted)):
                b = fretted[j]
                if a.fret < b.fret and a.finger >= b.finger:
                    return False
                if b.fret < a.fret and b.finger >= a.finger:
                    return False
                if not is_fallback_dyad and a.finger != b.finger:
                    if abs(a.fret - b.fret) > abs(a.finger - b.finger) + max_extra_stretch:
                        return False

        # Validación de cejilla cuando un mismo dedo presiona múltiples cuerdas
        for finger, group in by_finger.items():
            if len(group) == 1:
                continue
            if not self.biomechanics.allow_barre:
                return False
            barre_fret = group[0].fret
            if any(a.fret != barre_fret for a in group):
                return False
            if barre_fret != min_fretted:
                return False
            max_barre_string = max(a.string for a in group)
            for other in assignments:
                if other.string <= max_barre_string and other.fret < barre_fret:
                    return False

        return True

    def _compute_hand_position(self, assignments: Sequence[FingerAssignment]) -> int:
        fretted = [a for a in assignments if not a.is_open]
        if not fretted:
            return 0
        min_fret = min(a.fret for a in fretted)
        max_fret = max(a.fret for a in fretted)
        if max_fret - min_fret > self.biomechanics.max_hand_span:
            return max(1, round(sum(a.fret for a in fretted) / len(fretted)))
        min_a = min(fretted, key=lambda a: (a.fret, a.finger))
        return max(1, min_a.fret - (min_a.finger - 1))


class BaseDagBuilder:
    def __init__(
        self,
        fretboard: FretboardMap,
        biomechanics: BiomechanicsConfig = DEFAULT_SETTINGS.biomechanics,
        kinematic_pruning: bool = False,
    ) -> None:
        self.posture_generator = PostureGenerator(fretboard, biomechanics)
        self.biomechanics = biomechanics
        self.kinematic_pruning = kinematic_pruning

    def build(self, sequence: MusicalSequence) -> LayeredDAG:
        layers: List[Tuple[PostureNode, ...]] = []
        nodes_by_id: Dict[str, PostureNode] = {}
        adjacency: Dict[str, Tuple[str, ...]] = {}

        for event in sequence.events:
            layer = self.posture_generator.generate_layer(event)
            layers.append(layer)
            for node in layer:
                nodes_by_id[node.node_id] = node

        adjacency[SOURCE_NODE_ID] = tuple(node.node_id for node in layers[0])

        for idx in range(len(layers) - 1):
            next_ids = tuple(node.node_id for node in layers[idx + 1])
            ioi_sec = max(
                sequence.events[idx].ioi_sec or sequence.events[idx].max_duration_sec,
                0.05,
            )
            max_shift = max(
                self.biomechanics.base_hand_shift,
                int(
                    self.biomechanics.base_hand_shift
                    + self.biomechanics.hand_shift_velocity_rate * ioi_sec
                ),
            )

            for node in layers[idx]:
                if not self.kinematic_pruning:
                    adjacency[node.node_id] = next_ids
                    continue

                targets = tuple(
                    next_node.node_id
                    for next_node in layers[idx + 1]
                    if node.hand_position == 0
                    or next_node.hand_position == 0
                    or abs(next_node.hand_position - node.hand_position) <= max_shift
                )
                adjacency[node.node_id] = targets if targets else next_ids

        for node in layers[-1]:
            adjacency[node.node_id] = (SINK_NODE_ID,)

        adjacency[SINK_NODE_ID] = ()

        return LayeredDAG(
            layers=tuple(layers),
            nodes_by_id=nodes_by_id,
            adjacency=adjacency,
        )