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
    # Densidad lineal aproximada (kg/m) por cuerda (1 a 6) para modelado tímbrico
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
    max_hand_span: int = 5          # D_max: apertura elástica límite en trastes
    relaxed_hand_span: int = 3      # f_relax: extensión neutra sin tensión
    allow_barre: bool = True
    max_simultaneous_notes: int = 6
    # Penalización de fatiga por dedo (0=aire, 1=índice, 2=medio, 3=anular, 4=meñique)
    finger_fatigue_weights: Dict[int, float] = field(
        default_factory=lambda: {0: 0.0, 1: 1.0, 2: 1.05, 3: 1.26, 4: 1.44}
    )


@dataclass(frozen=True)
class CostVectorWeights:
    # W_bio: esfuerzo biomecánico e inercia de trayectoria
    alpha_shift: float = 1.5
    gamma_acceleration: float = 1.6
    beta_string_crossing: float = 0.8
    lambda_stretch: float = 1.2
    inertia_reversal_penalty: float = 2.0

    # W_timbre: continuidad acústica y homogeneidad de cuerda
    theta_string_change: float = 1.4
    high_fret_inharmonicity: float = 0.15

    # W_fluidez: articulación y ligados (legato)
    delta_legato_bonus: float = 1.2
    same_position_hold_bonus: float = 0.6
    max_legato_ioi_sec: float = 0.35
    comfort_fret_limit: int    = 8      # por debajo de este traste no hay coste de posición
    fret_position_penalty: float = 0.40     
    slide_penalty: float       = 0.18            


@dataclass(frozen=True)
class InstrumentSettings:
    physics: GuitarPhysicsConfig = field(default_factory=GuitarPhysicsConfig)
    biomechanics: BiomechanicsConfig = field(default_factory=BiomechanicsConfig)
    cost_weights: CostVectorWeights = field(default_factory=CostVectorWeights)


DEFAULT_SETTINGS = InstrumentSettings()