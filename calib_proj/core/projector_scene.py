from typing import List, Dict
import numpy as np

from calib_commons.camera import Camera
from calib_commons.objectPoint import ObjectPoint
from calib_commons.scene import SceneType
from calib_commons.types import idtype

from calib_proj.utils.plane import Plane
from calib_commons.scene import Scene

class ProjectorScene: 

    def __init__(self, 
                 cameras: Dict[idtype, Camera] = None, 
                 object_points: Dict[idtype, ObjectPoint] = None, 
                 projector: Camera = None,
                 plane: Plane = None,
                 scene_type: SceneType = None): 
        
        self.generic_scene = Scene(cameras, object_points, scene_type)
        self.projector = projector
        self.plane = plane
