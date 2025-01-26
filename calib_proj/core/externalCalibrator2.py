from typing import Dict, List, Tuple

import numpy as np
import itertools
import cv2
from enum import Enum 
from scipy.optimize import least_squares
import time 
import cProfile
import pstats
import copy
import matplotlib.pyplot as plt



from calib_commons.types import idtype
from calib_commons.world_frame import WorldFrame 
import calib_commons.utils.se3 as se3
from calib_commons.utils.se3 import SE3
from calib_commons.correspondences import get_conform_obs_of_cam, filter_correspondences_with_track_length, get_tracks
from calib_commons.intrinsics import Intrinsics
from calib_commons.observation import Observation
from calib_commons.scene import SceneType
from calib_commons.camera import Camera
from calib_commons.objectPoint import ObjectPoint

from calib_proj.core.projector_scene import ProjectorScene
from calib_proj.core.estimate import Estimate
from calib_proj.utils.plane import Plane
from calib_proj.core.config import ExternalCalibratorConfig, SolvingLevel
from calib_proj.utils.utils import homography_from_parametrization_4pt, homography_to_parametrization_4pt
import calib_proj.core.ba_homography as ba_homography
import calib_proj.core.ba_sparse as ba_sparse

from calib_proj.utils.homography import retrieve_motion_using_homography


# class WorldFrame(Enum): 
#     CAM_ID_1 = "CamId1"
#     CAM_FIRST_CHOOSEN = "CamFirstChoosen"



class ExternalCalibrator2: 

    def __init__(self, 
                 correspondences, 
                 intrinsics: Dict[idtype, Intrinsics], 
                 points_I0: Dict[idtype, np.ndarray], 
                 config: ExternalCalibratorConfig,
                ): 
        
        self.config = config

        # if self.config.SOLVING_LEVEL == SolvingLevel.FREE or self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
        #     self.correspondences = filter_correspondences_with_track_length(copy.deepcopy(correspondences), min_track_length=2)
        # elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
        #     self.correspondences = copy.deepcopy(correspondences) # 2d

        self.correspondences = copy.deepcopy(correspondences)
        if self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
            # check that all obs ids are in points_I0
            for cam_id in correspondences.keys(): 
                for point_id in correspondences[cam_id].keys(): 
                    if point_id not in points_I0.keys(): 
                        # print(f"point {point_id} in cam {cam_id} not in points_I0 ")
                        del self.correspondences[cam_id][point_id]

        # self.correspondences = copy.deepcopy(correspondences)

        self.intrinsics = intrinsics

        
        self.points_I0 = points_I0

        self.T = None # normalization matrix for homography


        # current estimate
        self.estimate = Estimate(SOLVING_LEVEL=self.config.SOLVING_LEVEL, POINTS_I0=points_I0)


    def select_initial_camera_pair(self, forbiden_pairs) -> Tuple[idtype, idtype]:

        cameras_id = self.correspondences.keys()
        # arrangements = list(itertools.permutations(cameras_id, 2))
        arrangements = list(itertools.combinations(cameras_id, 2))

        valid_arrangements = []
        for arrangement in arrangements:
            valid = True
            for forbiden_pair in forbiden_pairs:
                if arrangement == forbiden_pair or arrangement[::-1] == forbiden_pair:
                    valid = False
                    break
            if valid:
                valid_arrangements.append(arrangement)

        best_score = -np.inf
        best_cam_ids = None

        for arrangement in valid_arrangements:
            
            cam0_id, cam1_id = arrangement            

            common_points_id = set(self.get_conform_obs_in_cam(cam0_id).keys()) & set(self.get_conform_obs_in_cam(cam1_id).keys())
            # valid_common_checkers_id = self.filter_with_track_length(common_checkers_id)

            # tot_score = 0
            
            # 1
            image_points1 = []
            for pt_id in common_points_id:
                image_points1.append(self.correspondences[cam1_id][pt_id]._2d)

            if len(image_points1):
                image_points1 = np.vstack(image_points1)
                score1 = self.view_score(image_points1, self.intrinsics[cam1_id].resolution)
            else: 
                score1 = 0
            # print("(" + str(cam0Id) + ", " + str(cam1Id) + "): " + str(score))

        
            # 1
            image_points2 = []
            for pt_id in common_points_id:
                image_points2.append(self.correspondences[cam0_id][pt_id]._2d)

            if len(image_points2):
                image_points2 = np.vstack(image_points2)
                score2 = self.view_score(image_points2, self.intrinsics[cam0_id].resolution)
            else: 
                score2 = 0
            
            score = score1 + score2
            # print("(" + str(cam0Id) + ", " + str(cam1Id) + "): " + str(score))
            if score > best_score:
                best_score = score
                best_cam_ids = (cam0_id, cam1_id)

        return best_cam_ids
    
    def select_next_best_cam(self) -> int:
        candidates = self.get_remaining_camera_ids() 

        best_score = -np.inf
        best_cam_id = None

        for cam_id in candidates:
            score = self.get_camera_score(cam_id)
            
            # print("(" + str(cam0Id) + ", " + str(cam1Id) + "): " + str(score))
            if score > best_score:
                best_score = score
                best_cam_id = cam_id

        return best_cam_id
    
    def get_camera_score(self, 
                         cam_id: idtype) -> float:
        
        common_points_id = set(self.get_conform_obs_in_cam(cam_id).keys()) & self.estimate.scene.generic_scene.get_point_ids()

        image_points = []
        for pt_id in common_points_id:
            image_points.append(self.correspondences[cam_id][pt_id]._2d)

        if len(image_points):
                image_points = np.vstack(image_points)
                score = self.view_score(image_points, self.intrinsics[cam_id].resolution)
        else: 
            score = 0
        # image_points = np.vstack(image_points)
        # score = self.view_score(image_points, self.intrinsics[cam_id].resolution)
        return score
    
    def get_cameras_scores(self): 
        scores = {}
        for cam_id in self.estimate.scene.generic_scene.cameras.keys(): 
            scores[cam_id] = self.get_camera_score(cam_id)
        return scores

    def view_score(self, 
                   image_points: np.ndarray, 
                   image_resolution: Tuple):
        s = 0
        L = 3
        width, height = image_resolution
        for l in range(1, L+1):
            K_l = 2**l
            w_l = K_l  # Assuming w_l = K_l is correct
            grid = np.zeros((K_l, K_l), dtype=bool)
            for point in image_points:
                u,v = point
                u = np.clip(u, 0, width-1)
                v = np.clip(v, 0, height-1)
                
                x = int(np.ceil(K_l * u / width)) - 1
                y = int(np.ceil(K_l * v / height)) - 1
              
                if grid[x, y] == False:
                    grid[x, y] = True  # Mark the cell as full
                    s += w_l  # Increase the score
        return s
    
    def bootstrapping_from_initial_pair_old(self, 
                                        cam_id_1: idtype, 
                                        cam_id_2: idtype) -> bool: 
        
        print(f"*********** Bootstrapping from initial camera pair (cam1, cam2) = ({cam_id_1}, {cam_id_2}) ***********")

        cam1 = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])


        pts_cam1 = [self.correspondences[cam_id_1][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_1).keys()]
        pts_cam2 = [self.correspondences[cam_id_2][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_2).keys()]

        # test via essential matrix
        pts_cam1_array = np.array(pts_cam1).squeeze()
        pts_cam2_array = np.array(pts_cam2).squeeze()

        pts_cam_1_norm = self.intrinsics[cam_id_1].get_normalized_points(pts_cam1_array)
        pts_cam_2_norm = self.intrinsics[cam_id_2].get_normalized_points(pts_cam2_array)

        # E, mask = cv2.findEssentialMat(pts_cam_1_norm, pts_cam_2_norm, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # R1, R2, t = cv2.decomposeEssentialMat(E)
        # print("decomposeEssentialMat:")
        # print(R1)
        # print(R2)
        # print(t)

        # ret, R, t, mask = cv2.recoverPose(E, pts_cam_1_norm, pts_cam_2_norm, np.eye(3))
        # print("recoverPose:")
        # print(R)
        # print(t)

        # T_2_1 = se3.T_from_rt(R, t)
        # T_W_2 = se3.inv_T(T_2_1) # T^W_2
        # pose = SE3(T_W_2)
        # print(pose.get_R())
        # print(pose.get_t())
        # best_cam = Camera(cam_id_2, SE3(T_W_2), self.intrinsics[2])
        if 1:
            ## 
            H_normalized = self.estimate_inter_image_homography_normalized(cam_id_1=cam_id_1, 
                                                                        cam_id_2=cam_id_2)
        
            decompositions = cv2.decomposeHomographyMat(H_normalized, K=np.eye(3))

            # scenes = []

            if decompositions[0]:
                num_solutions = len(decompositions[1])
                print(f"Found {num_solutions} possible decompositions.")

                best_number_in_front = 0
                best_points = None
                best_points_ids = None
                best_cam = None
                best_plane = None
                numbers_in_front = []
                for i, (R_2_1, t_2_1, n_1) in enumerate(zip(decompositions[1], decompositions[2], decompositions[3])):
                    # print(f"Solution {i+1}:")
                    # print("Rotation Matrix R:")
                    # print(R_2_1)
                    # print("Translation Vector t:")
                    # print(t_2_1)
                    # print("Normal Vector n:")
                    # print(n)
                    # print(" ")
                    T_2_1 = se3.T_from_rt(R_2_1, t_2_1)
                    T_W_2 = se3.inv_T(T_2_1) # T^W_2
                    cam2 = Camera(cam_id_2, SE3(T_W_2), self.intrinsics[cam_id_2])
                    plane = Plane.from_normal_and_d(n_1, d=-1, id=i)
                    # scenes.append(ProjectorScene({1: cam1, 2: cam2}, object_points=None, plane=plane, scene_type=SceneType.ESTIMATE))

                    # solution disambiguation: count number of points in front of each camera
                    # cam1
                    in_front_cam1 = []
                    for pt in pts_cam1:
                        m = np.linalg.inv(self.intrinsics[cam_id_1].K) @ np.append(pt,1)
                        in_front_cam1.append(m.dot(n_1) > 0)
                    
                    # cam2
                    in_front_cam2 = []
                    n_2 = R_2_1 @ n_1
                    for pt in pts_cam2:
                        m = np.linalg.inv(self.intrinsics[cam_id_2].K) @ np.append(pt,1)
                        in_front_cam2.append(m.dot(n_2) > 0)
                    
                    # print(f"In front of cam1: {sum(in_front_cam1)} among {len(in_front_cam1)}")
                    # print(f"In front of cam2: {sum(in_front_cam2)} among {len(in_front_cam2)}")

                    in_front_all = in_front_cam1 + in_front_cam2
                    number_in_front = int(sum(in_front_all))
                    print(f"Solution {i+1} : total number of points in front of a camera: {number_in_front} among {len(in_front_all)}")
                    # print("")
                    numbers_in_front.append(number_in_front)


                    if number_in_front >= best_number_in_front: 
                        # if number_in_front == best_number_in_front and number_in_front != 0: 
                        #     raise ValueError("There are two solutions from decomposing the inter-image homography that have the same number of points with positive depth.")
                        best_number_in_front = number_in_front
                        # best_points = points_in_front
                        # best_points_ids = points_in_front_ids
                        best_cam = cam2
                        best_plane = plane

                    
                    
                max_number_in_front = max(numbers_in_front)
                if numbers_in_front.count(max_number_in_front) > 1: 
                    print("There are two solutions from decomposing the inter-image homography that have the same number of points triangulated with positive depth in all cameras.") 
                    return False
            else:
                print("Decomposition failed.")

        # return scenes
        self.estimate.scene.generic_scene.add_camera(cam1)
        self.estimate.scene.generic_scene.add_camera(best_cam)
        print("via homography decomposition:")
        print(best_cam.pose.get_R())
        print(best_cam.pose.get_t())

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE or self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
            self.correspondences = filter_correspondences_with_track_length(copy.deepcopy(self.correspondences), min_track_length=2)

        # structure parameters estimation depending on solving level
        # free          -> N 3d object points
        # planarity     -> plane + N 2d coordinates of points in the plane
        # homography    -> plane + homography plane-I_proj

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            #  N 3d object points
            object_points = self.triangulate_points(cam_id_1=cam_id_1, 
                                                    cam_id_2=cam_id_2)
            self.estimate.scene.generic_scene.object_points = object_points

        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY or self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
            self.estimate.scene.plane = best_plane

            # here we can use all points (including the ones seen in only one camera)

            # 1: points in common
            points = None
            ids = []
            object_points_in_world = self.triangulate_points(cam_id_1=cam_id_1, 
                                                            cam_id_2=cam_id_2)
            if object_points_in_world:
                points_in_common_ids = object_points_in_world.keys()
                ids += points_in_common_ids
                points = np.array([pt.position for pt in object_points_in_world.values()]).squeeze()
            else: 
                points_in_common_ids = []
            

            # 2: points seen in only one of the two cameras
            # in cam 1
            points_only_cam_1_ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_1_ids:
                img_pts = [self.correspondences[cam_id_1][point_id]._2d for point_id in points_only_cam_1_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_1 = self.estimate.scene.generic_scene.cameras[cam_id_1].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_1))
                else: 
                    points = points_cam_1
                ids += points_only_cam_1_ids
                

            # in cam 2
            points_only_cam_2_ids = list(set(self.get_conform_obs_in_cam(cam_id_2).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_2_ids:
                img_pts = [self.correspondences[cam_id_2][point_id]._2d for point_id in points_only_cam_2_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_2 = self.estimate.scene.generic_scene.cameras[cam_id_2].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_2))
                else: 
                    points = points_cam_2
                ids += points_only_cam_2_ids
            
            # ids = list(set(ids) & set(self.points_I0.keys()))
            # keep only points with valid ids (ones also in points_I0)


            points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)

            if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
                self.estimate._2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(ids)}
                self.estimate.update_points()

            elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
                p_plane = points_plane[:,:2]
                p_proj = [self.points_I0[id] for id in ids]
                

                # homography estimation
                pts_src = np.array(p_proj, dtype='float32').squeeze()
                pts_dst = np.array(p_plane, dtype='float32').squeeze()
                H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
                self.estimate.H_I0_plane = H

                _4pts = homography_to_parametrization_4pt(H)

                mu_x = np.mean(_4pts[:,0])
                mu_y = np.mean(_4pts[:,1])
                sigma_x = np.std(_4pts[:,0])
                sigma_y = np.std(_4pts[:,1])
                sigma = np.sqrt(sigma_x**2 + sigma_y**2)

                self.T = np.array([[np.sqrt(2)/sigma, 0, -np.sqrt(2)/sigma * mu_x], 
                                            [0, np.sqrt(2)/sigma, -np.sqrt(2)/sigma * mu_y], 
                                            [0,0,1]])
                
                # self.T = np.eye(3)



                ids = self.points_I0.keys() 
                # ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys() | set(self.get_conform_obs_in_cam(cam_id_2).keys())))

                pts_src = np.array(list(self.points_I0.values())).squeeze()
                pts_src_homogeneous = np.concatenate([pts_src, np.ones((pts_src.shape[0], 1))], axis=1)
                projected_pts = np.dot(H, pts_src_homogeneous.T).T

                # Convert from homogeneous to 2D
                projected_pts[:, 0] /= projected_pts[:, 2]
                projected_pts[:, 1] /= projected_pts[:, 2]   
                projected_pts[:,2] = 0
                points_world = se3.change_coords(self.estimate.scene.plane.pose.mat, projected_pts)     


                self.estimate.scene.generic_scene.object_points = {id: ObjectPoint(id, points_world[i,:]) for i, id in enumerate(ids)}
                
        else: 
            raise ValueError("Not implemented.")
        return True 
    


    def bootstrapping_from_initial_pair_old1(self, 
                                        cam_id_1: idtype, 
                                        cam_id_2: idtype) -> bool: 
        print(f"*********** Bootstrapping from initial camera pair (cam1, cam2) = ({cam_id_1}, {cam_id_2}) ***********")
        # cam1 = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
        pts_per_cam = {}
        pts_per_cam[cam_id_1] = {point_id: self.correspondences[cam_id_1][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_1).keys()}
        pts_per_cam[cam_id_2] = {point_id: self.correspondences[cam_id_2][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_2).keys()}
        sols_intial_pair = retrieve_motion_using_homography(self.intrinsics[cam_id_1].K, self.intrinsics[cam_id_2].K, pts_per_cam[cam_id_1], pts_per_cam[cam_id_2])
        if not sols_intial_pair:
            return False

        if len(sols_intial_pair) == 1:
            cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
            cam2_final = Camera(cam_id_2, sols_intial_pair[0][0], self.intrinsics[cam_id_2])
            n_1_final = sols_intial_pair[0][1]
        
        else: # two solutions
            # select third camera 
            in_pair_with_cam, third_cam_id = self.select_third_camera_for_bootsrapping(cam_id_1, cam_id_2)
            if third_cam_id is None:
                print("There is no third camera to remove the ambiguity.")
                return False
            print(f"Ambiguity removal with third camera {third_cam_id}.")
            world_id = in_pair_with_cam
            # if world_id == cam_id_1:
                # A_id = world
            # B_id = 
            pts_per_cam[third_cam_id] = {point_id: self.correspondences[third_cam_id][point_id]._2d for point_id in self.get_conform_obs_in_cam(third_cam_id).keys()}
            sols_second_pair = retrieve_motion_using_homography(self.intrinsics[world_id].K, self.intrinsics[third_cam_id].K, pts_per_cam[world_id], pts_per_cam[third_cam_id])

            # if world_id == cam_id_1:
            #     T_W_1 = SE3.idendity()
            for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
                print(f"n_world_from_2nd_pair: {n_W_from_2nd_pair}")

            point_in_front_of_third_camera = []
            for idx_first_pair, (T_1_2_, n_1_) in enumerate(sols_intial_pair):
                
                if world_id == cam_id_1: 
                    T_W_1_ = SE3.idendity()
                    T_W_2_ = T_1_2_
                    n_W_ = n_1_
                else: 
                    T_W_2 = SE3.idendity()
                    T_W_1_ = T_1_2_.inv()
                    R_W_1_ = T_W_1_.get_R()
                    n_W_ = R_W_1_ @ n_1_

                print(f"n_world: {n_W_}")
                local_pts_ = []
                for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
                    R_3_W_ = T_W_3_.inv().get_R()
                    n_3_ = R_3_W_ @ n_W_

                    # print()
                    in_front_cam3 = []
                    for pt in pts_per_cam[third_cam_id].values():
                        m = np.linalg.inv(self.intrinsics[third_cam_id].K) @ np.append(pt,1)
                        in_front_cam3.append(m.dot(n_3_) > 0)
                    number_in_front = int(sum(in_front_cam3))
                    print(f"# pts in front of the third camera for sol {idx_first_pair} of the initial pair and sol {idx_second_pair} of the second pair: {number_in_front}")
                    local_pts_.append(number_in_front)
                max_number_in_front = max(local_pts_)
                point_in_front_of_third_camera.append(max_number_in_front)

            print(f"Number of points in front of the third camera for first sol of the initial pair dec.: {point_in_front_of_third_camera[0]}")
            print(f"Number of points in front of the third camera for second sol of the initial pair dec.: {point_in_front_of_third_camera[1]}")
            max_number_in_front = max(point_in_front_of_third_camera)
            if point_in_front_of_third_camera.count(max_number_in_front) > 1:
                raise ValueError("ERROR")
            else: 
                best_sol_idx = point_in_front_of_third_camera.index(max_number_in_front)
                cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
                cam2_final = Camera(cam_id_2, sols_intial_pair[best_sol_idx][0], self.intrinsics[cam_id_2])
                n_1_final = sols_intial_pair[best_sol_idx][1]
        

                


        plane_final = Plane.from_normal_and_d(n_1_final, d=-1, id=0)




        #         # return False
        # else:
        #     print("Decomposition failed.")

        # return scenes
        self.estimate.scene.generic_scene.add_camera(cam1_final)
        self.estimate.scene.generic_scene.add_camera(cam2_final)
        print("via homography decomposition:")
        # print(best_cam.pose.get_R())
        # print(best_cam.pose.get_t())

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE or self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
            self.correspondences = filter_correspondences_with_track_length(copy.deepcopy(self.correspondences), min_track_length=2)

        # structure parameters estimation depending on solving level
        # free          -> N 3d object points
        # planarity     -> plane + N 2d coordinates of points in the plane
        # homography    -> plane + homography plane-I_proj

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            #  N 3d object points
            object_points = self.triangulate_points(cam_id_1=cam_id_1, 
                                                    cam_id_2=cam_id_2)
            self.estimate.scene.generic_scene.object_points = object_points

        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY or self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
            self.estimate.scene.plane = plane_final

            # here we can use all points (including the ones seen in only one camera)

            # 1: points in common
            points = None
            ids = []
            object_points_in_world = self.triangulate_points(cam_id_1=cam_id_1, 
                                                            cam_id_2=cam_id_2)
            if object_points_in_world:
                points_in_common_ids = object_points_in_world.keys()
                ids += points_in_common_ids
                points = np.array([pt.position for pt in object_points_in_world.values()]).squeeze()
            else: 
                points_in_common_ids = []
            

            # 2: points seen in only one of the two cameras
            # in cam 1
            points_only_cam_1_ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_1_ids:
                img_pts = [self.correspondences[cam_id_1][point_id]._2d for point_id in points_only_cam_1_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_1 = self.estimate.scene.generic_scene.cameras[cam_id_1].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_1))
                else: 
                    points = points_cam_1
                ids += points_only_cam_1_ids
                

            # in cam 2
            points_only_cam_2_ids = list(set(self.get_conform_obs_in_cam(cam_id_2).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_2_ids:
                img_pts = [self.correspondences[cam_id_2][point_id]._2d for point_id in points_only_cam_2_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_2 = self.estimate.scene.generic_scene.cameras[cam_id_2].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_2))
                else: 
                    points = points_cam_2
                ids += points_only_cam_2_ids
            
            # ids = list(set(ids) & set(self.points_I0.keys()))
            # keep only points with valid ids (ones also in points_I0)


            points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)

            if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
                self.estimate._2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(ids)}
                self.estimate.update_points()

            elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
                p_plane = points_plane[:,:2]
                p_proj = [self.points_I0[id] for id in ids]
                

                # homography estimation
                pts_src = np.array(p_proj, dtype='float32').squeeze()
                pts_dst = np.array(p_plane, dtype='float32').squeeze()
                H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
                self.estimate.H_I0_plane = H

                _4pts = homography_to_parametrization_4pt(H)

                mu_x = np.mean(_4pts[:,0])
                mu_y = np.mean(_4pts[:,1])
                sigma_x = np.std(_4pts[:,0])
                sigma_y = np.std(_4pts[:,1])
                sigma = np.sqrt(sigma_x**2 + sigma_y**2)

                self.T = np.array([[np.sqrt(2)/sigma, 0, -np.sqrt(2)/sigma * mu_x], 
                                            [0, np.sqrt(2)/sigma, -np.sqrt(2)/sigma * mu_y], 
                                            [0,0,1]])
                
                # self.T = np.eye(3)



                ids = self.points_I0.keys() 
                # ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys() | set(self.get_conform_obs_in_cam(cam_id_2).keys())))

                pts_src = np.array(list(self.points_I0.values())).squeeze()
                pts_src_homogeneous = np.concatenate([pts_src, np.ones((pts_src.shape[0], 1))], axis=1)
                projected_pts = np.dot(H, pts_src_homogeneous.T).T

                # Convert from homogeneous to 2D
                projected_pts[:, 0] /= projected_pts[:, 2]
                projected_pts[:, 1] /= projected_pts[:, 2]   
                projected_pts[:,2] = 0
                points_world = se3.change_coords(self.estimate.scene.plane.pose.mat, projected_pts)     


                self.estimate.scene.generic_scene.object_points = {id: ObjectPoint(id, points_world[i,:]) for i, id in enumerate(ids)}
                
        else: 
            raise ValueError("Not implemented.")
        return True 
        


   
    def bootstrapping_from_initial_pair(self, 
                                        cam_id_1: idtype, 
                                        cam_id_2: idtype) -> bool: 
        print(f"*********** Bootstrapping from initial camera pair (cam1, cam2) = ({cam_id_1}, {cam_id_2}) ***********")
        # cam1 = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
        pts_per_cam = {}
        pts_per_cam[cam_id_1] = {point_id: self.correspondences[cam_id_1][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_1).keys()}
        pts_per_cam[cam_id_2] = {point_id: self.correspondences[cam_id_2][point_id]._2d for point_id in self.get_conform_obs_in_cam(cam_id_2).keys()}
        sols_intial_pair = retrieve_motion_using_homography(self.intrinsics[cam_id_1].K, self.intrinsics[cam_id_2].K, pts_per_cam[cam_id_1], pts_per_cam[cam_id_2])
        if not sols_intial_pair:
            return False

        if len(sols_intial_pair) == 1:
            cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
            cam2_final = Camera(cam_id_2, sols_intial_pair[0][0], self.intrinsics[cam_id_2])
            n_1_final = sols_intial_pair[0][1]
        
        else: # two solutions
            # select third camera 
            in_pair_with_cam, third_cam_id = self.select_third_camera_for_bootsrapping(cam_id_1, cam_id_2)
            if third_cam_id is None:
                print("There is no third camera to remove the ambiguity.")
                return False
            print(f"Ambiguity removal with third camera {third_cam_id}.")
            world_id = in_pair_with_cam
            # if world_id == cam_id_1:
                # A_id = world
            # B_id = 
            pts_per_cam[third_cam_id] = {point_id: self.correspondences[third_cam_id][point_id]._2d for point_id in self.get_conform_obs_in_cam(third_cam_id).keys()}
            sols_second_pair = retrieve_motion_using_homography(self.intrinsics[world_id].K, self.intrinsics[third_cam_id].K, pts_per_cam[world_id], pts_per_cam[third_cam_id])

            # if world_id == cam_id_1:
            #     T_W_1 = SE3.idendity()
            for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
                print(f"n_world_from_2nd_pair: {n_W_from_2nd_pair}")

            point_in_front_of_third_camera = []
            
            errors_on_plane_normal = {}

            for idx_first_pair, (T_1_2_, n_1_) in enumerate(sols_intial_pair):
                
                if world_id == cam_id_1: 
                    n_W_ = n_1_
                else: 
                    T_W_1_ = T_1_2_.inv()
                    R_W_1_ = T_W_1_.get_R()
                    n_W_ = R_W_1_ @ n_1_
                
                

                for idx_second_pair, (T_W_3_, n_W_from_2nd_pair) in enumerate(sols_second_pair): 
                    errors_on_plane_normal[(idx_first_pair, idx_second_pair)] = np.linalg.norm(n_W_ - n_W_from_2nd_pair)
            
            print(f"Errors on plane normal:")
            print(errors_on_plane_normal)
            min_key = min(errors_on_plane_normal, key=errors_on_plane_normal.get)
            best_sol_idx = min_key[0]
                         
            
            cam1_final = Camera(cam_id_1, SE3.idendity(), self.intrinsics[cam_id_1])
            cam2_final = Camera(cam_id_2, sols_intial_pair[best_sol_idx][0], self.intrinsics[cam_id_2])
            n_1_final = sols_intial_pair[best_sol_idx][1]
        

                


        plane_final = Plane.from_normal_and_d(n_1_final, d=-1, id=0)




        #         # return False
        # else:
        #     print("Decomposition failed.")

        # return scenes
        self.estimate.scene.generic_scene.add_camera(cam1_final)
        self.estimate.scene.generic_scene.add_camera(cam2_final)
        print("via homography decomposition:")
        # print(best_cam.pose.get_R())
        # print(best_cam.pose.get_t())

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE or self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
            self.correspondences = filter_correspondences_with_track_length(copy.deepcopy(self.correspondences), min_track_length=2)

        # structure parameters estimation depending on solving level
        # free          -> N 3d object points
        # planarity     -> plane + N 2d coordinates of points in the plane
        # homography    -> plane + homography plane-I_proj

        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            #  N 3d object points
            object_points = self.triangulate_points(cam_id_1=cam_id_1, 
                                                    cam_id_2=cam_id_2)
            self.estimate.scene.generic_scene.object_points = object_points

        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY or self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
            self.estimate.scene.plane = plane_final

            # here we can use all points (including the ones seen in only one camera)

            # 1: points in common
            points = None
            ids = []
            object_points_in_world = self.triangulate_points(cam_id_1=cam_id_1, 
                                                            cam_id_2=cam_id_2)
            if object_points_in_world:
                points_in_common_ids = object_points_in_world.keys()
                ids += points_in_common_ids
                points = np.array([pt.position for pt in object_points_in_world.values()]).squeeze()
            else: 
                points_in_common_ids = []
            

            # 2: points seen in only one of the two cameras
            # in cam 1
            points_only_cam_1_ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_1_ids:
                img_pts = [self.correspondences[cam_id_1][point_id]._2d for point_id in points_only_cam_1_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_1 = self.estimate.scene.generic_scene.cameras[cam_id_1].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_1))
                else: 
                    points = points_cam_1
                ids += points_only_cam_1_ids
                

            # in cam 2
            points_only_cam_2_ids = list(set(self.get_conform_obs_in_cam(cam_id_2).keys()) - \
                                         set(points_in_common_ids))
            if points_only_cam_2_ids:
                img_pts = [self.correspondences[cam_id_2][point_id]._2d for point_id in points_only_cam_2_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points_cam_2 = self.estimate.scene.generic_scene.cameras[cam_id_2].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                if points is not None:
                    points = np.vstack((points, points_cam_2))
                else: 
                    points = points_cam_2
                ids += points_only_cam_2_ids
            
            # ids = list(set(ids) & set(self.points_I0.keys()))
            # keep only points with valid ids (ones also in points_I0)


            points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)

            if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY: 
                self.estimate._2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(ids)}
                self.estimate.update_points()

            elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
                p_plane = points_plane[:,:2]
                p_proj = [self.points_I0[id] for id in ids]
                

                # homography estimation
                pts_src = np.array(p_proj, dtype='float32').squeeze()
                pts_dst = np.array(p_plane, dtype='float32').squeeze()
                H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
                self.estimate.H_I0_plane = H

                _4pts = homography_to_parametrization_4pt(H)

                mu_x = np.mean(_4pts[:,0])
                mu_y = np.mean(_4pts[:,1])
                sigma_x = np.std(_4pts[:,0])
                sigma_y = np.std(_4pts[:,1])
                sigma = np.sqrt(sigma_x**2 + sigma_y**2)

                self.T = np.array([[np.sqrt(2)/sigma, 0, -np.sqrt(2)/sigma * mu_x], 
                                            [0, np.sqrt(2)/sigma, -np.sqrt(2)/sigma * mu_y], 
                                            [0,0,1]])
                
                # self.T = np.eye(3)



                ids = self.points_I0.keys() 
                # ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys() | set(self.get_conform_obs_in_cam(cam_id_2).keys())))

                pts_src = np.array(list(self.points_I0.values())).squeeze()
                pts_src_homogeneous = np.concatenate([pts_src, np.ones((pts_src.shape[0], 1))], axis=1)
                projected_pts = np.dot(H, pts_src_homogeneous.T).T

                # Convert from homogeneous to 2D
                projected_pts[:, 0] /= projected_pts[:, 2]
                projected_pts[:, 1] /= projected_pts[:, 2]   
                projected_pts[:,2] = 0
                points_world = se3.change_coords(self.estimate.scene.plane.pose.mat, projected_pts)     


                self.estimate.scene.generic_scene.object_points = {id: ObjectPoint(id, points_world[i,:]) for i, id in enumerate(ids)}
                
        else: 
            raise ValueError("Not implemented.")
        return True 
    
    def select_third_camera_for_bootsrapping(self, cam_id_1, cam_id_2):
        cameras_id =  list(set(self.correspondences.keys()) - {cam_id_1, cam_id_2})
        arrangements_1 = [(cam_id_1, cam_id) for cam_id in cameras_id]
        arrangements_2 = [(cam_id_2, cam_id) for cam_id in cameras_id]
        arrangements = arrangements_1 + arrangements_2
        best_score = -np.inf
        best_cam_ids = None

        for arrangement in arrangements:
            
            cam0_id, cam1_id = arrangement            

            common_points_id = set(self.get_conform_obs_in_cam(cam0_id).keys()) & set(self.get_conform_obs_in_cam(cam1_id).keys())
            # valid_common_checkers_id = self.filter_with_track_length(common_checkers_id)

            # tot_score = 0
            
            # 1
            image_points1 = []
            for pt_id in common_points_id:
                image_points1.append(self.correspondences[cam1_id][pt_id]._2d)

            if len(image_points1):
                image_points1 = np.vstack(image_points1)
                score1 = self.view_score(image_points1, self.intrinsics[cam1_id].resolution)
            else: 
                score1 = 0
            # print("(" + str(cam0Id) + ", " + str(cam1Id) + "): " + str(score))

        
            # 1
            image_points2 = []
            for pt_id in common_points_id:
                image_points2.append(self.correspondences[cam0_id][pt_id]._2d)

            if len(image_points2):
                image_points2 = np.vstack(image_points2)
                score2 = self.view_score(image_points2, self.intrinsics[cam0_id].resolution)
            else: 
                score2 = 0
            
            score = score1 + score2
            # print("(" + str(cam0Id) + ", " + str(cam1Id) + "): " + str(score))
            if score > best_score:
                best_score = score
                best_cam_ids = (cam0_id, cam1_id)


        third_cam_id = best_cam_ids[1]
        return best_cam_ids[0], third_cam_id

    def change_coords_object_points(self, 
                                    T_2_1, 
                                    object_points_1: Dict[idtype, ObjectPoint]): 
        points_1 = [pt.position for pt in object_points_1.values()]
        points_1 = np.array(points_1, dtype='float32').squeeze()
        points_2 = se3.change_coords(T_2_1, points_1)
        ids = list(object_points_1.keys())
        object_points_2 = {id: ObjectPoint(id=id, position=points_2[i,:]) for i, id in enumerate(ids)}
        return object_points_2
    
    def triangulate_points(self, 
                            cam_id_1: idtype, 
                            cam_id_2: idtype) -> Dict[idtype, ObjectPoint]: 
        
        common_points_ids = list(set(self.get_conform_obs_in_cam(cam_id_1).keys()) & \
                            set(self.get_conform_obs_in_cam(cam_id_2).keys()))
        if common_points_ids:
            pts_src = [self.correspondences[cam_id_1][point_id]._2d for point_id in common_points_ids]
            pts_dst = [self.correspondences[cam_id_2][point_id]._2d for point_id in common_points_ids]
            pts_src = np.array(pts_src, dtype='float32').squeeze()
            pts_dst = np.array(pts_dst, dtype='float32').squeeze()
            points = cv2.triangulatePoints(self.estimate.scene.generic_scene.cameras[cam_id_1].get_projection_matrix(), 
                                        self.estimate.scene.generic_scene.cameras[cam_id_2].get_projection_matrix(), 
                                        pts_src.T, 
                                        pts_dst.T)
            points = points / points[3,:]
            points = points[:3, :]
            # object_points = {i+1: ObjectPoint(id=i+1, position=points[:,i], color='black') for i in range(points.shape[1])}

            object_points = {id: ObjectPoint(id, points[:,i]) for i, id in enumerate(common_points_ids)}
            return object_points
        

    def triangulate_new_points(self, 
                                cam_id_1: idtype, 
                                cam_id_2: idtype) -> Dict[idtype, ObjectPoint]: 
        
        common_points_ids = set(self.get_conform_obs_in_cam(cam_id_1).keys()) & \
                            set(self.get_conform_obs_in_cam(cam_id_2).keys())
        
        new_points_ids = list(common_points_ids - self.estimate.scene.generic_scene.get_point_ids())

        if new_points_ids:
            pts_src = [self.correspondences[cam_id_1][point_id]._2d for point_id in new_points_ids]
            pts_dst = [self.correspondences[cam_id_2][point_id]._2d for point_id in new_points_ids]
            pts_src = np.array(pts_src, dtype='float32').squeeze()
            pts_dst = np.array(pts_dst, dtype='float32').squeeze()
            points = cv2.triangulatePoints(self.estimate.scene.generic_scene.cameras[cam_id_1].get_projection_matrix(), 
                                        self.estimate.scene.generic_scene.cameras[cam_id_2].get_projection_matrix(), 
                                        pts_src.T, 
                                        pts_dst.T)
            points = points / points[3,:]
            points = points[:3, :]
            # object_points = {i+1: ObjectPoint(id=i+1, position=points[:,i], color='black') for i in range(points.shape[1])}

            object_points = {id: ObjectPoint(id, points[:,i]) for i, id in enumerate(new_points_ids)}
            return object_points

                


    def calibrate(self): 

        # bootstrapping
        success = False
        forbiden_pairs = []
        # while not success:
        initial_camera_pair_ids = self.select_initial_camera_pair(forbiden_pairs=forbiden_pairs)
        # print(initial_camera_pair_ids)
        # initial_camera_pair_ids = ('A', 'B')
        success = self.bootstrapping_from_initial_pair(cam_id_1=initial_camera_pair_ids[0], 
                                                        cam_id_2=initial_camera_pair_ids[1])
            # print(f"Bootstrapping from initial camera pair (cam1, cam2) = ({initial_camera_pair_ids[0]}, {initial_camera_pair_ids[1]}): {success}")
            # if not success: 
            #     forbiden_pairs.append(initial_camera_pair_ids)
        if not success: 
            print("Bootstrapping failed.")
            return False
        
        # self.iterative_filtering()
        # return 
        
        # loop over remaining cameras: 
        # 1. add camera using observation with estimated 3d points
        # 2. triangulate new points
        # 3. iterative filtering (BA + filtering, repeated)
        while len(self.get_remaining_camera_ids()): 
            cam_id = self.select_next_best_cam()
            self.add_camera(cam_id)
            # return True
            self.add_points(cam_id)
            self.iterative_filtering()


        print(" ")
        print("##################### CALIBRATION TERMINATED #####################")

        scores = self.get_cameras_scores()
        print("scores: ", scores)
        if min(scores.values()) > self.config.camera_score_threshold: 
            print ("Calibration Status: SUCCESS: each camera has sufficient distribution score.")
            # return True, scores
        else: 
            print (f"Calibration Status: FAILURE: some cameras do not have a sufficient distribution score.")
            # return False, scores
        return True
         
    def iterative_filtering(self): 
        print(" ")
        print("----- Iterative Filtering -----")
        iter = 1
        continue_ = True
        while continue_:
            print(" ")
            print(f"----- Iteration: {iter} -----" )
            iter += 1
            print("** BA: **")
            self.BA()
            print(" ")
            print("** Filtering: **")

            # if self.config.display_reprojection_errors:
            #     self.display_histogram_reprojection_errors()
            continue_ = self.filtering()
    
    def filtering(self) -> bool: 
        num_point_filtered = 0
        points_removed = []


        max_filter = np.inf
        point_ids = sorted(list(self.estimate.scene.generic_scene.object_points.keys()))
        for camera in self.estimate.scene.generic_scene.cameras.values(): 
            # print("")
            # print(f"camera {camera.id}")
            for point_id in point_ids: 
                # print(f"point estimate {point_id}")
                point = self.estimate.scene.generic_scene.object_points.get(point_id)
                if point:
                    # print(point)
                    observation = self.correspondences[camera.id].get(point.id)
                    if observation and observation._is_conform:
                        _2d_reprojection = camera.reproject(point.position)
                        errors_xy = observation._2d - _2d_reprojection
                        error = np.sqrt(np.sum(errors_xy**2))
                        # print()
                        is_obsv_conform = error < self.config.reprojection_error_threshold
                        if self.config.display_reprojection_errors:
                            print(f"observation of point {point.id:>3} in cam {camera.id} error: {error:.2f} [pix]")
                        if not is_obsv_conform: 
                            # print(f"observation of point {point.id:>3} in cam {camera.id} error: {error:.2f} [pix]")

                            num_point_filtered += 1
                            self.correspondences[camera.id][point.id]._is_conform = False 



                            if self.config.SOLVING_LEVEL == SolvingLevel.FREE:
                                min_track_length = 2
                            elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                                min_track_length = 1
                            elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY: 
                                min_track_length = 0
                            
                            if len(self.get_tracks()[point.id]) < min_track_length:

                                # print(f"removing point {point.id} from estimate")
                                points_removed.append(point.id)
                                # self.p.append(checker.id)   
                                del self.estimate.scene.generic_scene.object_points[point.id]
                                if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
                                    del self.estimate._2d_plane_points[point.id]

                            if num_point_filtered > max_filter:

                                print(f" -> Number of observations filtered: {num_point_filtered}")
                                print(f" -> Number of points removed from estimate: {len(points_removed)}")
                                return True

        if self.config.display: 
            print(f" -> Number of observations filtered: {num_point_filtered}")
            print(f" -> Number of points removed from estimate: {len(points_removed)}")
        return num_point_filtered > 0

    def add_camera(self, cam_id: idtype): 
        K = self.intrinsics[cam_id].K
        common_points_id = set(self.get_conform_obs_in_cam(cam_id).keys()) & self.estimate.scene.generic_scene.get_point_ids()

        # PnP
        _2d = [self.correspondences[cam_id][point_id]._2d for point_id in list(common_points_id)]
        _3d = [self.estimate.scene.generic_scene.object_points[point_id].position for point_id in list(common_points_id)]
        _2d = np.vstack(_2d)
        _3d = np.vstack(_3d)

        T_W_C = self.pnp(_2d, _3d, K)
        self.estimate.scene.generic_scene.add_camera(Camera(cam_id, SE3(T_W_C), self.intrinsics[cam_id]))
        if self.config.display: 
            print(" ")
            print("*********** Cam " + str(cam_id) + " added ***********")
            print(f"estimated using PnP over {len(common_points_id)} points")

        return
    
    def add_points(self, new_cam_id: idtype) -> None: 
        new_points_ids = []
        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            for camera_id in self.estimate.scene.generic_scene.cameras: 
                if camera_id == new_cam_id:
                    continue
                object_points = self.triangulate_new_points(cam_id_1=camera_id, cam_id_2=new_cam_id)
                if object_points:
                    self.estimate.scene.generic_scene.object_points.update(object_points)
                    new_points_ids += list(object_points.keys())
                    print(f"new points triangulated using cameras ({camera_id}, {new_cam_id}): {len(list(object_points.keys()))}")
                else: 
                    print(f"new points triangulated using cameras ({camera_id}, {new_cam_id}): none")

        if self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            points_ids = set(self.get_conform_obs_in_cam(new_cam_id).keys()) 
            new_points_ids = list(points_ids - self.estimate.scene.generic_scene.get_point_ids())
            if new_points_ids:
                img_pts = [self.correspondences[new_cam_id][point_id]._2d for point_id in new_points_ids]
                img_pts = np.array(img_pts, dtype='float32').squeeze()
                points = self.estimate.scene.generic_scene.cameras[new_cam_id].backproject_points_using_plane(self.estimate.scene.plane.get_pi(), img_pts)
                points_plane = se3.change_coords(se3.inv_T(self.estimate.scene.plane.pose.mat), points)

                new_2d_plane_points = {id: points_plane[i,:2] for i, id in enumerate(new_points_ids)}
                self.estimate._2d_plane_points.update(new_2d_plane_points)
                self.estimate.update_points()


        if self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
            pass


        if self.config.display: 
            if len(new_points_ids):
                print("New points: " + str(len(sorted(new_points_ids))))
            else: 
                print("New points: None")   
    
    
    def pnp(self, 
            _2d: np.ndarray, 
            _3d: np.ndarray, 
            K: np.ndarray) -> np.ndarray: 
        """
       Solve PnP using RANSAC, followed by non-linear refinement (minimizing reprojection error) using LM. 

        Args : 
            _2d: np.ndarray, dim: (N, 3)
                image-points
            _3d: np.ndarray, dim: (M, 3)
                 object-points
            K: np.ndarray, dim: (3,3)
                intrinsics matrix
        Returns : 
            T_W_C: np.ndarray, dim: (4, 4)
                pose of the camera in the world frame {w} (the frame of the object-points)
        """
        # method = "RANSAC_AND_REFINEMENT"
        method = "RANSAC_ONLY"

        if method == "RANSAC_ONLY":
            _, rvec, tvec, inliers = cv2.solvePnPRansac(imagePoints=_2d, 
                                                        objectPoints=_3d, 
                                                        cameraMatrix=K, 
                                                        distCoeffs=None, 
                                                        useExtrinsicGuess=False)
            # _, rvec, tvec = cv2.solvePnP(objectPoints=_3d, 
            #                              imagePoints=_2d, 
            #                              cameraMatrix=K, 
            #                              distCoeffs=None)
            
            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6)

            # # Refine pose using solvePnPRefineLM
            # rvec, tvec = cv2.solvePnPRefineLM(imagePoints=_2d, 
            #                                 objectPoints=_3d, 
            #                                 cameraMatrix=K, 
            #                                 distCoeffs=None,  
            #                                 rvec=rvec0, 
            #                                 tvec=tvec0,
            #                                 criteria=criteria)

            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            return T_W_C
        

        if method == "RANSAC_AND_REFINEMENT":
            _, rvec0, tvec0, inliers = cv2.solvePnPRansac(imagePoints=_2d, 
                                                        objectPoints=_3d, 
                                                        cameraMatrix=K, 
                                                        distCoeffs=None, 
                                                        useExtrinsicGuess=False)
            # _, rvec, tvec = cv2.solvePnP(objectPoints=_3d, 
            #                              imagePoints=_2d, 
            #                              cameraMatrix=K, 
            #                              distCoeffs=None)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6)

            # Refine pose using solvePnPRefineLM
            rvec, tvec = cv2.solvePnPRefineLM(imagePoints=_2d, 
                                            objectPoints=_3d, 
                                            cameraMatrix=K, 
                                            distCoeffs=None,  
                                            rvec=rvec0, 
                                            tvec=tvec0,
                                            criteria=criteria)

            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            return T_W_C
    

  
        
   
    
    def BA(self):
        
        intrinsics = []
        for camera in self.estimate.scene.generic_scene.cameras.values():
            K = camera.intrinsics.K
            intrinsics.append({
                'fx': K[0, 0],
                'fy': K[1, 1],
                'cx': K[0, 2],
                'cy': K[1, 2]
            })

    
        camera_pose_array = np.zeros((len(self.estimate.scene.generic_scene.cameras) , 6))

        # Fill camera_pose_array
        for i, camera in enumerate(self.estimate.scene.generic_scene.cameras.values()):
            rvec, tvec = se3.rvec_tvec_from_T(np.linalg.inv(camera.pose.mat))
            camera_pose_array[i, :3] = rvec.squeeze()
            camera_pose_array[i, 3:] = tvec.squeeze()

        points_2d = []
        camera_indices = []
        point_indices = []

        for cam_idx, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            for pt_idx, (pt_id, point) in enumerate(self.estimate.scene.generic_scene.object_points.items()):
                observation = self.correspondences[cam_id].get(pt_id)
                if observation and observation._is_conform:
                    points_2d.append(observation._2d)
                    camera_indices.append(cam_idx)
                    point_indices.append(pt_idx)

        points_2d = np.array(points_2d, dtype='float32').squeeze()
        camera_indices = np.array(camera_indices, dtype='int')
        point_indices = np.array(point_indices, dtype='int')
    


        if self.config.SOLVING_LEVEL == SolvingLevel.FREE: 
            self.BA_free(camera_pose_array, intrinsics, points_2d, camera_indices, point_indices)
        elif self.config.SOLVING_LEVEL == SolvingLevel.PLANARITY:
            self.BA_planarity(camera_pose_array, intrinsics, points_2d, camera_indices, point_indices)
        elif self.config.SOLVING_LEVEL == SolvingLevel.HOMOGRAPHY:
            self.BA_homography()

        return

    def BA_free(self, camera_pose_array, intrinsics, points_2d, camera_indices, point_indices):

        points_3d = np.zeros((len(self.estimate.scene.generic_scene.object_points), 3))
        # Fill points_3d
        for i, point in enumerate(self.estimate.scene.generic_scene.object_points.values()):
            points_3d[i, :] = point.position

        sba = ba_sparse.BA_Free(camera_pose_array, points_3d, points_2d, camera_indices, point_indices, intrinsics, self.config.ba_least_square_ftol)

        # Perform bundle adjustment
        optimized_params = sba.bundleAdjust()
        optimized_camera_params, optimized_points_3d = optimized_params

        # Update the scene with the optimized parameters
        # Update camera poses
        for i, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:]
            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            camera.pose = SE3(T_W_C)

        # Update 3D points
        for i, (pt_id, point) in enumerate(self.estimate.scene.generic_scene.object_points.items()):
            point.position = optimized_points_3d[i]
    
        return

    def BA_planarity(self, camera_pose_array, intrinsics, points_2d, camera_indices, point_indices):

        # plane pose
        rvec,tvec = se3.rvec_tvec_from_T(self.estimate.scene.plane.pose.mat)
        plane_pose = np.zeros(6)
        plane_pose[:3] = rvec.squeeze()
        plane_pose[3:] = tvec.squeeze()

        # Fill points_2d_in_plane
        points_2d_in_plane = np.zeros((len(self.estimate.scene.generic_scene.object_points), 2))
        for i, point in enumerate(self.estimate._2d_plane_points.values()):
            points_2d_in_plane[i, :] = point

        # Initialize the PySBA class with fixed intrinsic parameters
        sba = ba_sparse.BA_PlaneConstraint(camera_pose_array, plane_pose, points_2d_in_plane, points_2d, camera_indices, point_indices, intrinsics, self.config.ba_least_square_ftol)

        # Perform bundle adjustment
        optimized_params = sba.bundleAdjust()
        optimized_camera_params, optimized_plane_pose, optimized_points_2d_plane = optimized_params

        # Update the scene with the optimized parameters
        # Update camera poses
        for i, (cam_id, camera) in enumerate(self.estimate.scene.generic_scene.cameras.items()):
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:]
            T_C_W = se3.T_from_rvec_tvec(rvec, tvec)
            T_W_C = se3.inv_T(T_C_W)
            camera.pose = SE3(T_W_C)

        rvec = optimized_plane_pose[:3]
        tvec = optimized_plane_pose[3:]
        T_W_plane = se3.T_from_rvec_tvec(rvec, tvec)
        self.estimate.scene.plane = Plane(SE3(T_W_plane))
        
        for i, (pt_id, point) in enumerate(self.estimate._2d_plane_points.items()):
            self.estimate._2d_plane_points[pt_id] = optimized_points_2d_plane[i]

    
        self.estimate.update_points()
    
        return
    
    def BA_homography(self):

        x = []

        # (n-1) cameras
        for camera in self.estimate.scene.generic_scene.cameras.values():
            # we skip the first choosen camera (it's pose is identity)
            if camera.id == next(iter(self.estimate.scene.generic_scene.cameras)): # next(iter(dict)) returns the first key of the dict
                continue
            x.append(se3.q_from_T(camera.pose.mat))


    
        # plane
        x.append(se3.q_from_T(self.estimate.scene.plane.pose.mat))
        # print(f"T:", self.T)
        # print("H", self.estimate.H_I0_plane)
        # print("4pt", homography_to_parametrization_4pt(self.estimate.H_I0_plane))
        H_tilde = self.T @ self.estimate.H_I0_plane
        # print("H_tilde", H_tilde)
        _4pts_tilde = homography_to_parametrization_4pt(H_tilde)
        # print("4pt_tilde", _4pts_tilde)
        x.append(_4pts_tilde.ravel()[:8])


        x0 = np.concatenate(x, axis=0)

        
        num_cameras = self.estimate.scene.generic_scene.get_num_cameras()
        num_points = self.estimate.scene.generic_scene.get_num_points()
        intrinsics_array = np.zeros((3,3,num_cameras))
        for i, camera in enumerate(self.estimate.scene.generic_scene.cameras.values()): 
            intrinsics_array[:,:,i] = self.intrinsics[camera.id].K

        observations_array = self.estimate.get_observations_array(self.correspondences)
        mask_observations = ~np.isnan(observations_array).any(axis=0)
        valid_obsv = observations_array[:, mask_observations]

        pts_ids = list(self.estimate.scene.generic_scene.object_points.keys())
        
        points_I0_array = np.ones((len(pts_ids), 3))
        for i, id in enumerate(pts_ids): 
            points_I0_array[i,:2] = self.points_I0[id]
    


        t0 = time.time()
        
        results = least_squares(fun=ba_homography.cost_function, 
                                x0=x0, 
                                verbose=2, 
                                x_scale = 'jac',
                                ftol = self.config.ba_least_square_ftol,
                                method='trf', 
                                args =  (num_cameras, num_points, intrinsics_array, valid_obsv, mask_observations, self.config.SOLVING_LEVEL, points_I0_array, self.T)
                                )
               
        t1 = time.time()
        print(f"BA optimization time: {t1-t0:>3.2f} [s]")

        # self.estimate.update_from_x(results.x, self.T)
        x = results.x
        cameras_poses = ba_homography.extract_cameras_poses_from_x(x, num_cameras)
        for i, camera_id in enumerate(self.estimate.scene.generic_scene.cameras.keys()):
            self.estimate.scene.generic_scene.cameras[camera_id].pose = SE3(cameras_poses[:,:,i])

            q_plane = x[6*(num_cameras-1):6*num_cameras]
            self.plane = Plane(SE3.from_q(q_plane))
         
           

          
        # homography (4 pt parametrization)
        _4pts_tilde = np.reshape(x[6*num_cameras:], (4,2))
        # print(_4pts_tilde)

        H_tilde = homography_from_parametrization_4pt(_4pts_tilde)
        self.estimate.H_I0_plane = np.linalg.inv(self.T) @ H_tilde


        self.estimate.update_points()

    def get_scene(self, world_frame) -> ProjectorScene:
        if world_frame == WorldFrame.CAM_FIRST_CHOOSEN: 
            T = np.eye(4)
        elif self.estimate.scene.generic_scene.cameras.get(world_frame): 
            T = se3.inv_T(self.estimate.scene.generic_scene.cameras[world_frame].pose.mat)
        # elif world_frame == WorldFrame.CAM_ID_1: 
        #     T = se3.inv_T(self.estimate.scene.generic_scene.cameras[1].pose.mat)
        else: 
            raise ValueError("World Frame not valid.")
        
        cameras = {id: Camera(id, SE3(T @ camera.pose.mat), camera.intrinsics) for id, camera in self.estimate.scene.generic_scene.cameras.items()}
        object_points = {id: ObjectPoint(id, (T @ np.append(p.position, 1))[:3]) for id, p in self.estimate.scene.generic_scene.object_points.items()}
        if self.estimate.scene.plane: 
            plane = Plane(pose=SE3(T@self.estimate.scene.plane.pose.mat))
        else: 
            plane = None
        return ProjectorScene(cameras=cameras, 
                      object_points=object_points, 
                      plane = plane,
                      scene_type=SceneType.ESTIMATE)
    
    def get_conform_obs_in_cam(self, 
                               cam_id: idtype): 
        return get_conform_obs_of_cam(cam_id=cam_id, correspondences=self.correspondences)

    def get_common_obs_ids_in_cameras(self, 
                                  cam_1_id: idtype, 
                                  cam_2_id: idtype): 
        return set(self.get_conform_obs_in_cam(cam_1_id).keys()) & set(self.get_conform_obs_in_cam(cam_2_id).keys())
    
    def get_camera_ids(self) -> set[idtype]: 
        return set(self.correspondences.keys())

    def get_num_cameras(self) -> int: 
        return len(self.correspondences)
    
    def get_tracks(self) -> Dict[idtype, set[idtype]]: 
       return get_tracks(self.correspondences)
    
    def get_remaining_camera_ids(self) -> set[idtype]: 
        return self.get_camera_ids() - self.estimate.scene.generic_scene.get_camera_ids()
    
