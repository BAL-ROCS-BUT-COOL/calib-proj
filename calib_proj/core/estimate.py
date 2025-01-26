from typing import Dict
import numpy as np

import calib_commons.utils.se3 as se3
from calib_commons.types import idtype
from calib_commons.scene import SceneType
from calib_commons.objectPoint import ObjectPoint

from calib_proj.core.projector_scene import ProjectorScene
from calib_proj.core.config import SolvingLevel


class Estimate:

    def __init__(self, SOLVING_LEVEL: SolvingLevel):
        self.SOLVING_LEVEL = SOLVING_LEVEL
        self.scene = ProjectorScene(scene_type=SceneType.ESTIMATE)
        self._2d_plane_points : Dict[idtype, np.ndarray] = None

    def update_points(self):
       if self.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            ids = list(self._2d_plane_points.keys())
            _2d_points = np.vstack(list(self._2d_plane_points.values()))
            _3D_plane_points = np.hstack((_2d_points, np.zeros((_2d_points.shape[0], 1))))
            _3D_points_world = se3.change_coords(self.scene.plane.pose.mat, _3D_plane_points)
            self.scene.generic_scene.object_points = {id: ObjectPoint(id, _3D_points_world[i,:]) for i, id in enumerate(ids)}

