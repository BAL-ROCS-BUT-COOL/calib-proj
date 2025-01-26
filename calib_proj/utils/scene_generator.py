from typing import Dict, Tuple

import numpy as np
import copy
from scipy.spatial.transform import Rotation 

from calib_commons.camera import Camera
from calib_commons.objectPoint import ObjectPoint
from calib_commons.scene import SceneType
from calib_proj.core.projector_scene import ProjectorScene
from calib_commons.intrinsics import Intrinsics

from calib_commons.utils.generateCircularCameras import generateCircularCameras
import calib_commons.utils.se3 as se3
from calib_commons.utils.se3 import SE3
from calib_commons.utils.so3 import *
from calib_commons.types import idtype
from calib_commons.utils.utils import K_from_params

class SceneGenerator: 

    def __init__(self, 
                 num_points: int, 
                 cameras: Dict[idtype, Camera],
                #  min_track_length: int, 
                 noise_std: float,
                 std_on_3d_points=None, 
                 alpha: float = 1,
                 pi = np.array([0.0,0.0,1.0,0])):
        self.alpha = alpha
        # self.range_points = range_points
        self.num_points = num_points
        self.pi = pi
        self.std_on_3d_points = std_on_3d_points
        self.cameras = cameras

        # self.min_track_length = min_track_length
        self.noise_std = noise_std
        
        K = np.array([[1055, 0, 1920/2], 
                      [0, 1055, 1080/2], 
                      [0, 0, 1]])
        self.intrinsics = Intrinsics(K, (1920, 1080))

        
        self.K_proj = K_from_params(fx=1000, fy=1000, cx=960, cy=540)

    def generate_scene(self, world_frame) -> Tuple[ProjectorScene, Dict]: 
        # cameras, T = self.generate_cameras(self.intrinsics)
        # if np.dot(self.pi)
        T = np.eye(4)
        projector_pose = se3.inv_T(T) @ se3.T_from_q(np.array([0, np.pi, 0, 0, 0, 2.7]))
        projector = Camera(0, SE3(projector_pose), Intrinsics(self.K_proj, resolution=(1920, 1080)))

        if self.pi is None:
            while 1:
                pi = np.random.rand(4)
                pi = T.T @ pi

                side = [np.dot(pi, np.append(camera.pose.get_t(),1)) < 0 for camera in self.cameras.values()]
                if all(side) or not any(side): 
                    self.pi = pi
                    break
        else:
            pi = T.T @ self.pi
            side = [np.dot(pi, np.append(camera.pose.get_t(),1)) < 0 for camera in self.cameras.values()]
            if all(side) or not any(side): 
                pass
            else:
                raise ValueError("PLANE IN MIDDLE OF CAMERAS !")
        



        # Calcul du pas pour avoir un espacement uniforme en x et y
        width, height = projector.intrinsics.resolution[0], projector.intrinsics.resolution[1]
        step = np.sqrt((width * height) / self.num_points)

        # Calcul du nombre de points en x et y
        N_x = int(width // step)
        N_y = int(height // step)

        # Recalculer le step réel pour s'assurer qu'il s'ajuste à l'image
        step_x = width / N_x
        step_y = height / N_y

        # Générer les coordonnées uniformes
        x = np.linspace(step_x / 2, width - step_x / 2, N_x)
        y = np.linspace(step_y / 2, height - step_y / 2, N_y)

        # Créer la grille des points
        _2d_points_projector = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        # _2d_points_projector = np.random.random((self.num_points, 2)) 
        
        # x = np.linspace(0, 1920, int(np.sqrt(5000)))
        # y = np.linspace(0, 1080, int(np.sqrt(5000)))
        # _x, _y = np.meshgrid(x, y)
        # _2d_points_projector = np.vstack([_x.ravel(), _y.ravel()]).T
        # _2d_points_projector[:, 0] = self.alpha * projector.intrinsics.resolution[0] * (_2d_points_projector[:, 0]-0.5) + projector.intrinsics.resolution[0] / 2
        # _2d_points_projector[:, 1] = self.alpha * projector.intrinsics.resolution[1] * (_2d_points_projector[:, 1]-0.5) + projector.intrinsics.resolution[1] / 2 
       
        points = projector.backproject_points_using_plane(pi, _2d_points_projector)

        # if self.std_on_3d_points is not None:
        points[:, 2] = np.random.normal(0, self.std_on_3d_points, points.shape[0])
      
        if world_frame is None: 
            T_world = np.eye(4)
        else:
            T_world = se3.inv_T(self.cameras[world_frame].pose.mat)
        cameras = {id: Camera(id, SE3(T_world @ camera.pose.mat), camera.intrinsics) for id, camera in self.cameras.items()}
        object_points = {i+1: ObjectPoint(id=i+1, position=(T_world @ np.append(points[i,:], 1))[:3]) for i in range(points.shape[0])}

        projector = Camera(0, SE3(T_world @ projector_pose), Intrinsics(self.K_proj, resolution=(1920, 1080)))

        scene = ProjectorScene(cameras, object_points, projector=projector, scene_type=SceneType.SYNTHETIC)
        scene.generic_scene.generate_noisy_observations(noise_std=self.noise_std)
        points_projector = {i+1: _2d_points_projector[i, :] for i in range(_2d_points_projector.shape[0])}


        return scene, points_projector

    def generate_cameras(self, intrinsics) -> Tuple[Dict[idtype, Camera], np.ndarray]: 
        point_to_look_at = np.zeros(3)
        poses = generateCircularCameras(point_to_look_at, self.distance_cameras, self.tilt_cameras, self.num_cameras)
        T = poses[0]  
        
        cameras = {}
        for i in range(self.num_cameras):            
            pose = np.linalg.inv(T) @ poses[i]
            id = i + 1
            # if id == 2: 
            #     K =  np.array([[500, 0, 435/2], 
            #             [0, 500, 333/2], 
            #             [0, 0, 1]])
            #     intrinsics = Intrinsics(K, (1000, 2000))
            cameras[id] = Camera(id, SE3(pose), intrinsics) 

        return cameras, T

    # def generate_object_points(self, T): 
    #     x = self.range_points * 2 * (np.random.rand(self.num_points) - 0.5)
    #     y = self.range_points * 2 * (np.random.rand(self.num_points) - 0.5)

    #     a, b, c = self.pi[:3]
    #     d = self.pi[3]
    #     z = -(d + a * x + b * y) / c
    #     Ps = np.column_stack((x, y, z))

    #     Ps = np.column_stack((Ps, np.ones(Ps.shape[0]))).T
    #     points = se3.inv_T(T) @ Ps
    #     points = points[:3, :].T

    #     object_points = {i + 1: ObjectPoint(id=i + 1, position=points[i, :]) for i in range(points.shape[0])}
    #     return object_points
