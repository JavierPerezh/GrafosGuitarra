from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union

from src.domain.musical_event import MusicalSequence


class ScoreParsingError(Exception):
    """Error producido durante la lectura o validación de una partitura."""


class BaseScoreParser(ABC):
    supported_extensions: Tuple[str, ...] = ()

    def can_parse(self, file_path: Union[str, Path]) -> bool:
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")
        if self.supported_extensions and not self.can_parse(path):
            raise ScoreParsingError(
                f"Extensión no soportada '{path.suffix}'. "
                f"Se esperaba: {self.supported_extensions}"
            )
        return path

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> MusicalSequence:
        """Transforma el archivo de entrada en una MusicalSequence discreta."""