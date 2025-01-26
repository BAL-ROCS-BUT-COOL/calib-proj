import json 
import numpy as np
import cv2 
from skspatial.objects import Points, Plane

from calib_commons.utils.se3 import SE3

PTS_SRC_HOMOGRAPHY = np.array([[-1,-1], 
                                [-1,+1], 
                                [+1,+1], 
                                [+1,-1]]) * 1


def K_from_params(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0,  cx], 
                     [0,  fy, cy], 
                     [0,  0,  1]])

# def reproject(P: np.ndarray, 
#                 objectPointsinWorld) -> np.ndarray: 
#     assert (objectPointsinWorld.ndim == 2 and objectPointsinWorld.shape[1] == 3) or (objectPointsinWorld.ndim == 1 and objectPointsinWorld.shape[0] == 3), "Array must have 3 columns"

#     if (objectPointsinWorld.ndim == 1 and objectPointsinWorld.shape[0] == 3): 
#          objectPointsinWorld = objectPointsinWorld[:,None].T
#     augmentedPoints = np.ones((4, objectPointsinWorld.shape[0]))
#     augmentedPoints[:3,:] = objectPointsinWorld.T

#     # _2dHom = P @ np.vstack((objectPointsinWorld.T, np.ones((1, objectPointsinWorld.shape[0]))))
#     _2dHom = P @ augmentedPoints
#     _2d = _2dHom / _2dHom[2,:]
#     _2d = _2d[:2, :].T

#     return _2d 


def reprojections(estimate, checkerboardGeometry): 
    repr = {}
    for checker in estimate.checkers.values(): 
        repr[checker.id] = {}
        _3d = checkerboardGeometry.get3DPointsInWorld(checker.pose)
        for camera in estimate.cameras.values(): 
                _2dReprojection = camera.reproject(_3d)
                repr[checker.id][camera.id] = _2dReprojection
                
    return repr


def extractObservationsForBA(estimate, correspondences):
    num_cameras = estimate.getNumCameras()
    num_points = estimate.getNumPoints()
    
    observations = np.full((2, num_cameras, num_points), np.nan)
    for i, camera in enumerate(estimate.cameras.values()): 
        for k, point in enumerate(estimate.object_points.values()):
            observation = correspondences[camera.id].get(point.id)
            if observation and observation._isConform:
                observations[:, i, k] = observation._2d
            
    
    return observations
         
     
def compute_inter_image_homography(pts_src, pts_dst):
    H, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
    return H


def fit_plane(points: np.ndarray): 
 
    # Creating Points object from numpy array
    points = Points(points)

    # Fitting a plane to the points
    plane = Plane.best_fit(points)
    return plane.normal, plane.point
   
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
    # _, rvec, tvec, inliers = cv2.solvePnPRansac(imagePoints=_2d, 
    #                                             objectPoints=_3d, 
    #                                             cameraMatrix=K, 
    #                                             distCoeffs=None, 
    #                                             iterationsCount=self.PNPRANSAC_Iter, 
    #                                             reprojectionError=self.PNPRANSAC_Threshold, 
    #                                             confidence=self.PNPRANSAC_Confidence,
    #                                             useExtrinsicGuess=False, 
    #                                             flags=cv2.SOLVEPN)
    _, rvec, tvec = cv2.solvePnP(objectPoints=_3d, 
                                    imagePoints=_2d, 
                                    cameraMatrix=K, 
                                    distCoeffs=None)
    
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


def homography_to_parametrization_4pt(H: np.ndarray) -> np.ndarray: 
    if ~np.isclose(H[2,2], 1):
        raise ValueError("H[2,2] is not = 1.")
   
    pts_src = PTS_SRC_HOMOGRAPHY
    
    pts_src_homogeneous = np.vstack((pts_src.T, np.ones((1, pts_src.shape[0]))))
    pts_dst = (H @ pts_src_homogeneous).T

    # Convert from homogeneous to 2D
    pts_dst[:, 0] /= pts_dst[:, 2]
    pts_dst[:, 1] /= pts_dst[:, 2]

    return pts_dst[:,:2]

def homography_from_parametrization_4pt(pts_dst: np.ndarray) -> np.ndarray: 
    # pts_src = np.array([[-1,-1], 
    #                     [-1,+1], 
    #                     [+1,+1], 
    #                     [+1,-1]]) * 1e0
    pts_src = PTS_SRC_HOMOGRAPHY

    # p_plane
    # pts_src = np.array(p_proj, dtype='float32').squeeze()
    # pts_dst = np.array(p_plane, dtype='float32').squeeze()
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
    return H


def convert_correspondences_to_observations_for_eval(correspondences): 
    observations = {}
    for cam_id, corr in correspondences.items(): 
        observations[cam_id] = {}
        for point_id, obs in corr.items(): 
            if obs._is_conform: 
                observations[cam_id][point_id] = obs._2d
    return observations


def get_reprojections_for_eval(scene): 
    reprojections = {}
    for camera in scene.cameras.values(): 
        reprojections[camera.id] = {}
        for object_point in scene.object_points.values(): 
            _2d = camera.reproject(object_point.position).squeeze()
            # _2d += noise_std * np.random.normal(size=(_2d.shape[0], 2))
            # within_fov = all(camera.intrinsics.valid_points(_2d))
            # if not mask_outside_fov or within_fov:
            reprojections[camera.id][object_point.id] = _2d

    return reprojections



