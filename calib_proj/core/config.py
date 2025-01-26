from dataclasses import dataclass
from enum import Enum 


class SolvingLevel(Enum): 
    FREE = 0
    PLANARITY = 1


@dataclass
class ExternalCalibratorConfig:
    """Configuration defining the parameters use in the external calibrator."""
    SOLVING_LEVEL: SolvingLevel = SolvingLevel.FREE
    reprojection_error_threshold: float = 1
    camera_score_threshold: float = 200
    ba_least_square_ftol: float = 1e-8
    verbose: int = 1
    least_squares_verbose: int = 1
