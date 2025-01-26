from typing import Dict
import numpy as np

from calib_proj.core.projector_scene import ProjectorScene
from calib_proj.utils.plane import Plane
from calib_proj.core.config import SolvingLevel
from calib_proj.core.ba_homography import extract_cameras_poses_from_x
from calib_proj.utils.utils import homography_to_parametrization_4pt, homography_from_parametrization_4pt

from calib_commons.utils.se3 import SE3
import calib_commons.utils.se3 as se3
from calib_commons.types import idtype
from calib_commons.scene import SceneType
from calib_commons.objectPoint import ObjectPoint
from calib_commons.correspondences import Correspondences






class Estimate:

    def __init__(self, SOLVING_LEVEL: SolvingLevel, POINTS_I0):
        self.SOLVING_LEVEL = SOLVING_LEVEL
        self.POINTS_I0 = POINTS_I0

        # estimates
        self.scene = ProjectorScene(scene_type=SceneType.ESTIMATE)
        # self.scene.plane: Plane = None
        self.H_I0_plane: np.ndarray = None
        self._2d_plane_points : Dict[idtype, np.ndarray] = None

        # self.T = None # np.eye(3) # normalization matrix for homography 


    def get_parametrization(self, T):
        x = []

        # (n-1) cameras
        for camera in self.scene.generic_scene.cameras.values():
            # we skip the first choosen camera (it's pose is identity)
            if camera.id == next(iter(self.scene.generic_scene.cameras)): # next(iter(dict)) returns the first key of the dict
                continue
            x.append(se3.q_from_T(camera.pose.mat))

        if self.SOLVING_LEVEL == SolvingLevel.FREE:
           # N 3D points
            for point in self.scene.generic_scene.object_points.values():
                x.append(point.position)

        elif self.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            # plane
            x.append(se3.q_from_T(self.scene.plane.pose.mat))
            # N 2D points in plane
            for point in self._2d_plane_points.values():
                x.append(point)

        elif self.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
            # plane
            x.append(se3.q_from_T(self.scene.plane.pose.mat))
            # homography plane-I0
            # if self.H_I0_plane[2,2] != 1:
            #     raise ValueError("H[2,2] is not = 1.")
            # x.append(self.H_I0_plane.ravel()[:8])
            # normalization = 1
            # if not normalization:
            #     _4pts = homography_to_parametrization_4pt(self.H_I0_plane)
            #     _4pts_tilde = _4pts
            # else:
                # T = 
            print(f"T:", T)
            print("H", self.H_I0_plane)
            print("4pt", homography_to_parametrization_4pt(self.H_I0_plane))
            H_tilde = T @ self.H_I0_plane
            print("H_tilde", H_tilde)
            _4pts_tilde = homography_to_parametrization_4pt(H_tilde)
            print("4pt_tilde", _4pts_tilde)
            x.append(_4pts_tilde.ravel()[:8])


        x = np.concatenate(x, axis=0)
        return x

    def update_from_x(self, x, T):
        num_cameras = self.scene.generic_scene.get_num_cameras()
        cameras_poses = extract_cameras_poses_from_x(x, num_cameras)
        for i, camera_id in enumerate(self.scene.generic_scene.cameras.keys()):
            self.scene.generic_scene.cameras[camera_id].pose = SE3(cameras_poses[:,:,i])

        if self.SOLVING_LEVEL == SolvingLevel.FREE:
            # N 3D points
            num_points = self.scene.generic_scene.get_num_points()
            points_3d = np.reshape(x[6*(num_cameras-1):], (num_points,3)).T
            for i, pt_id in enumerate(self.scene.generic_scene.object_points.keys()):
                self.scene.generic_scene.object_points[pt_id].position = points_3d[:,i]

        else: 
            # plane
            q_plane = x[6*(num_cameras-1):6*num_cameras]
            self.scene.plane = Plane(SE3.from_q(q_plane))
         
            if self.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                # N 2D points in plane
                num_points = self.scene.generic_scene.get_num_points()
                _2d_plane_points = np.reshape(x[6*num_cameras:], (num_points,2))
                self._2d_plane_points = {id: _2d_plane_points[i,:] for i, id in enumerate(list(self._2d_plane_points.keys()))}

            if self.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
                # homography (4 pt parametrization)
                _4pts_tilde = np.reshape(x[6*num_cameras:], (4,2))
                # print(_4pts_tilde)

                H_tilde = homography_from_parametrization_4pt(_4pts_tilde)
                self.H_I0_plane = np.linalg.inv(T) @ H_tilde
                # raise ValueError("UPDATE FROM X FOR HOMOGRAPHY TO IMPLEMENT")


            self.update_points()


    def update_points(self):
        if self.SOLVING_LEVEL == SolvingLevel.FREE:
            pass

        elif self.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            ids = list(self._2d_plane_points.keys())
            _2d_points = np.vstack(list(self._2d_plane_points.values()))
            _3D_plane_points = np.hstack((_2d_points, np.zeros((_2d_points.shape[0], 1))))
            _3D_points_world = se3.change_coords(self.scene.plane.pose.mat, _3D_plane_points)
            self.scene.generic_scene.object_points = {id: ObjectPoint(id, _3D_points_world[i,:]) for i, id in enumerate(ids)}

        elif self.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
            # raise ValueError("UPDATE POINTS FOR HOMOGRAPHY TO IMPLEMENT")
            # pts_src = np.array(p_proj, dtype='float32').squeeze()
            # pts_dst = np.array(p_plane, dtype='float32').squeeze()
            # H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
            # self.estimate.H_I0_plane = H

            

            # estimate using H and plane 
            ids = list(self.scene.generic_scene.object_points.keys())

            # homography estimation
            pts_src = np.array([self.POINTS_I0[id] for id in ids], dtype='float32').squeeze()
            # raise ValueError("TIM: NEED TO KEEP ONLY POINTS VISIBLE IN AT LEAST ONE CAMERA.")


            pts_src_homogeneous = np.hstack((pts_src, np.ones((pts_src.shape[0], 1))))
            projected_pts = (self.H_I0_plane @ pts_src_homogeneous.T).T

            # Convert from homogeneous to 2D
            projected_pts[:, 0] /= projected_pts[:, 2]
            projected_pts[:, 1] /= projected_pts[:, 2]   
            projected_pts[:,2] = 0
            points_world = se3.change_coords(self.scene.plane.pose.mat, projected_pts)   
            self.scene.generic_scene.object_points = {id: ObjectPoint(id, points_world[i,:]) for i, id in enumerate(ids)}


    def get_observations_array(self,
                               observations: Correspondences):
        num_cameras = self.scene.generic_scene.get_num_cameras()
        num_points = self.scene.generic_scene.get_num_points()



        observations_array = np.full((2, num_points, num_cameras), np.nan)
        for i, camera in enumerate(self.scene.generic_scene.cameras.values()):
            for k, point in enumerate(self.scene.generic_scene.object_points.values()):
                observation = observations[camera.id].get(point.id)
                # print(f"k,i {k,i}, camera id, pt id {camera.id, point.id}: {observation._2d}")
                if observation and observation._is_conform:
                    observations_array[:, k, i] = observation._2d

        return observations_array

    # def cost_function(self):
