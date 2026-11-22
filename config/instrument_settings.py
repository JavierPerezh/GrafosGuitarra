from dataclasses import dataclass, field
from typing import Dict, Tuple


def midi_to_frequency(midi_note: int, a4_hz: float = 440.0) -> float:
    return a4_hz * (2.0 ** ((midi_note - 69) / 12.0))


@dataclass(frozen=True)
class GuitarPhysicsConfig:
    num_strings: int = 6
    num_frets: int = 22
    scale_length_m: float = 0.648
    # Cuerda 1 (E4) a Cuerda 6 (E2) en MIDI
    standard_tuning_midi: Tuple[int, ...] = (64, 59, 55, 50, 45, 40)
    string_names: Tuple[str, ...] = ("E4", "B3", "G3", "D3", "A2", "E2")
    # Densidad lineal (kg/m) por cuerda (1 a 6)
    linear_mass_density: Tuple[float, ...] = (
        0.00039,
        0.00069,
        0.00114,
        0.00208,
        0.00368,
        0.00635,
    )

    @property
    def open_string_frequencies(self) -> Dict[int, float]:
        return {
            string_idx + 1: midi_to_frequency(midi)
            for string_idx, midi in enumerate(self.standard_tuning_midi)
        }

    def fret_vibrating_length(self, fret: int) -> float:
        return self.scale_length_m * (2.0 ** (-fret / 12.0))


@dataclass(frozen=True)
class BiomechanicsConfig:
    available_fingers: Tuple[int, ...] = (1, 2, 3, 4)
    open_string_finger: int = 0
    max_hand_span: int = 5          # D_max: apertura límite en trastes
    relaxed_hand_span: int = 3      # f_relax: extensión neutra sin tensión
    allow_barre: bool = True
    max_simultaneous_notes: int = 6
    base_hand_shift: int = 4
    hand_shift_velocity_rate: float = 14.0
    # Fatiga relativa por dedo (0=aire, 1=índice, 2=medio, 3=anular, 4=meñique)
    finger_fatigue_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 0.0, 1: 1.0, 2: 1.05, 3: 1.26, 4: 1.44}
    )


@dataclass(frozen=True)
class CostVectorWeights:
    # W_bio: esfuerzo biomecánico e inercia
    alpha_shift: float = 1.5
    gamma_acceleration: float = 1.6
    beta_string_crossing: float = 0.8
    same_finger_hop_penalty: float = 0.55
    interior_string_skip_penalty: float = 1.80
    lambda_stretch: float = 1.4
    inertia_reversal_penalty: float = 2.0
    comfort_fret_limit: int = 5
    fret_position_penalty: float = 0.55
    slide_penalty: float = 0.18

    # W_timbre: continuidad acústica y homogeneidad tímbrica
    theta_string_change: float = 1.4
    open_string_timbre_penalty: float = 0.10
    fret_shift_timbre_factor: float = 0.06
    voice_choke_timbre_penalty: float = 0.35
    high_fret_inharmonicity: float = 0.22

    # W_fluidez: articulación, ligados y anclaje postural
    delta_legato_bonus: float = 1.2
    same_position_hold_bonus: float = 0.6
    open_string_relief_bonus: float = 0.25
    max_legato_ioi_sec: float = 0.35


@dataclass(frozen=True)
class SolverConfig:
    exact_eps: float = 1e-9
    merge_eps: float = 0.08
    max_solutions: int = 5
    max_segment_events: int = 10
    min_segment_events: int = 3
    min_rest_ioi_sec: float = 0.45
    default_time_signature: Tuple[int, int] = (4, 4)
    kinematic_pruning_min_events: int = 12


@dataclass(frozen=True)
class InstrumentSettings:
    physics: GuitarPhysicsConfig = field(default_factory=GuitarPhysicsConfig)
    biomechanics: BiomechanicsConfig = field(default_factory=BiomechanicsConfig)
    cost_weights: CostVectorWeights = field(default_factory=CostVectorWeights)
    solver: SolverConfig = field(default_factory=SolverConfig)


DEFAULT_SETTINGS = InstrumentSettings()