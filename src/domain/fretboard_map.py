from dataclasses import dataclass
from typing import Dict, List, Tuple

from config.instrument_settings import DEFAULT_SETTINGS, GuitarPhysicsConfig


@dataclass(frozen=True, order=True)
class FretboardPosition:
    string: int  # 1 (E4) a 6 (E2)
    fret: int    # 0 (aire) a num_frets

    @property
    def is_open(self) -> bool:
        return self.fret == 0


class FretboardMap:
    def __init__(self, physics: GuitarPhysicsConfig = DEFAULT_SETTINGS.physics) -> None:
        self.physics = physics
        self._pitch_to_positions: Dict[int, List[FretboardPosition]] = {}
        self._position_to_pitch: Dict[FretboardPosition, int] = {}
        self._build_map()

    def _build_map(self) -> None:
        for string_idx, open_midi in enumerate(self.physics.standard_tuning_midi):
            string_num = string_idx + 1
            for fret in range(self.physics.num_frets + 1):
                midi_pitch = open_midi + fret
                pos = FretboardPosition(string=string_num, fret=fret)
                self._pitch_to_positions.setdefault(midi_pitch, []).append(pos)
                self._position_to_pitch[pos] = midi_pitch

        for positions in self._pitch_to_positions.values():
            positions.sort()

    @property
    def min_midi_pitch(self) -> int:
        return min(self._pitch_to_positions.keys())

    @property
    def max_midi_pitch(self) -> int:
        return max(self._pitch_to_positions.keys())

    def is_playable(self, midi_pitch: int) -> bool:
        return midi_pitch in self._pitch_to_positions

    def get_positions(self, midi_pitch: int) -> Tuple[FretboardPosition, ...]:
        positions = self._pitch_to_positions.get(midi_pitch)
        if not positions:
            raise ValueError(
                f"Altura MIDI {midi_pitch} fuera del rango del instrumento "
                f"[{self.min_midi_pitch}, {self.max_midi_pitch}]."
            )
        return tuple(positions)

    def get_midi_pitch(self, position: FretboardPosition) -> int:
        if position not in self._position_to_pitch:
            raise ValueError(f"Posición inválida en el diapasón: {position}")
        return self._position_to_pitch[position]

    def get_linear_density(self, string: int) -> float:
        if not 1 <= string <= self.physics.num_strings:
            raise ValueError(f"Número de cuerda inválido: {string}")
        return self.physics.linear_mass_density[string - 1]

    def get_vibrating_length(self, fret: int) -> float:
        if not 0 <= fret <= self.physics.num_frets:
            raise ValueError(f"Traste fuera de rango: {fret}")
        return self.physics.fret_vibrating_length(fret)

    def timbral_distance(self, pos_a: FretboardPosition, pos_b: FretboardPosition) -> float:
        # Diferencia relativa de densidad lineal y longitud vibrante entre dos posiciones
        mu_a = self.get_linear_density(pos_a.string)
        mu_b = self.get_linear_density(pos_b.string)
        density_ratio = abs(mu_a - mu_b) / max(mu_a, mu_b)
        fret_shift = abs(pos_a.fret - pos_b.fret) / self.physics.num_frets
        return density_ratio + 0.5 * fret_shift