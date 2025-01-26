from dataclasses import dataclass
from enum import Enum 


class SolvingLevel(Enum): 
    FREE = 0
    PLANARITY = 1
    HOMOGRAPHY = 2


@dataclass
class ExternalCalibratorConfig:
    """Configuration defining the parameters use in the external calibrator."""
    SOLVING_LEVEL: SolvingLevel = SolvingLevel.FREE
    reprojection_error_threshold: float = 1
    camera_score_threshold: float = 200
    ba_least_square_ftol: float = 1e-8
    display: bool = False
    display_reprojection_errors: bool = False