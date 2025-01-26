import numpy as np
from calib_proj.core.config import SolvingLevel
# from calib_proj.calib.estimate import Estimate
# import calib_proj.utils.se3 as se3
import calib_commons.utils.se3 as se3
from calib_proj.utils.utils import homography_from_parametrization_4pt

def cost_function(x: np.ndarray, 
                  num_cameras: int, 
                  num_points: int,
                  intrinsics_array: np.ndarray, 
                  valid_obsv: np.ndarray, 
                  mask_observations: np.ndarray, 
                  solving_level: SolvingLevel, 
                  points_I0_array: np.ndarray, 
                  T: np.ndarray
                  ): 
    
    reprojections = reproject_points(x=x, 
                                    num_cameras=num_cameras, 
                                    num_points=num_points, 
                                    intrinsics=intrinsics_array, 
                                    solving_level=solving_level,
                                    points_I0_array=points_I0_array, 
                                    T=T)
    valid_reprojections = reprojections[:, mask_observations]
    errors = valid_obsv - valid_reprojections
    errors_vector = errors.ravel()
    return errors_vector


def reproject_points(x: np.ndarray, 
                     num_cameras: int,
                     num_points: int,
                     intrinsics: np.ndarray, 
                     solving_level: SolvingLevel, 
                     points_I0_array: np.ndarray, 
                     T: np.ndarray): 
   
    camera_poses = extract_cameras_poses_from_x(x, num_cameras)
    camera_poses_inv = np.linalg.inv(camera_poses.transpose(2, 0, 1)).transpose(1, 2, 0)

    # _3D_points_world is of shape (3, M)
    # poses is of shape (4, 4, n)
    # intrinsics is of shape (3, 3, n)

    if solving_level == SolvingLevel.FREE:
        _3D_points_world = np.reshape(x[6*(num_cameras-1):], (num_points,3)).T

    elif solving_level == SolvingLevel.PLANARITY:
        q_plane = x[6*(num_cameras-1):6*num_cameras]
        plane_pose = se3.T_from_q(q_plane)
        _2d_plane_points = np.reshape(x[6*num_cameras:], (num_points,2))
        _3D_plane_points = np.hstack((_2d_plane_points, np.zeros((_2d_plane_points.shape[0], 1))))
        _3D_points_world = se3.change_coords(plane_pose, _3D_plane_points).T

    elif solving_level == SolvingLevel.HOMOGRAPHY:
        # plane
        q_plane = x[6*(num_cameras-1):6*num_cameras]
        plane_pose = se3.T_from_q(q_plane)
        # print(q_plane)

        # homography (4 pt parametrization)
        _4pts_tilde = np.reshape(x[6*num_cameras:], (4,2))
        # print(_4pts_tilde)
        H_tilde = homography_from_parametrization_4pt(_4pts_tilde)
        H = np.linalg.inv(T) @ H_tilde

        _2d_plane_points_hom = (H @ points_I0_array.T).T

        # Convert from homogeneous to 2D
        _2d_plane_points_hom[:, 0] /= _2d_plane_points_hom[:, 2]
        _2d_plane_points_hom[:, 1] /= _2d_plane_points_hom[:, 2]

        _2d_plane_points = _2d_plane_points_hom[:,:2]
        _3D_plane_points = np.hstack((_2d_plane_points, np.zeros((_2d_plane_points.shape[0], 1))))
        _3D_points_world = se3.change_coords(plane_pose, _3D_plane_points).T


    # Add a row of ones to the 3D points to make them homogeneous coordinates (4, M)
    _3D_points_world_homogeneous = np.vstack((_3D_points_world, np.ones((1, num_points))))
    # Transform points using camera extrinsics
    _3D_points_world_homogeneous = np.expand_dims(_3D_points_world_homogeneous, axis=2)
    transformed_points_homogeneous = np.einsum('ijk,jlk->ilk', camera_poses_inv, _3D_points_world_homogeneous)
    # Keep only the first three coordinates (x, y, z)
    transformed_points = transformed_points_homogeneous[:3, :, :]
    # Use einsum to multiply intrinsics (3, 3, n) with transformed points (3, M, n) resulting in (3, M, n)
    projected_points = np.einsum('ijk,jlk->ilk', intrinsics, transformed_points)
    # Normalize by the third coordinate to get pixel coordinates
    pixel_coordinates = projected_points[:2, :, :] / projected_points[2, :, :]
    return pixel_coordinates
    


def transform_points(points_world, frame_poses):
    # Extract the rotation matrices and translation vectors
    R = frame_poses[:3, :3, :]  # (3, 3, n)
    t = frame_poses[:3, 3, :]  # (3, n)
    
    # Add a new axis to points_world to broadcast the addition of translation
    points_world_expanded = points_world[:, np.newaxis, :]  # (3, 1, M)
    
    # Perform the matrix multiplication and addition in a vectorized manner
    transformed_points = np.einsum('ijk,jm->ikm', R, points_world_expanded) + t[:, :, np.newaxis]  # (3, n, M)
    
    return transformed_points

def extract_cameras_poses_from_x(x: np.ndarray, 
                                num_cameras: int): 
    
    camera_poses = np.zeros((4, 4, num_cameras))
    camera_poses[:,:,0] = np.eye(4)
    other_camera_poses = compute_poses_from_6dof_vectorized(x[:(num_cameras-1)*6])
    camera_poses[:,:,1:] = other_camera_poses
 
    return camera_poses


def compute_poses_from_6dof_vectorized(x):
    """Compute the array of camera poses from the vector x using vectorized operations."""
    N = x.shape[0] // 6  # Number of cameras

    # Reshape x to separate Euler angles and translation vectors
    x = x.reshape(N, 6)
    euler_angles = x[:, :3]
    translations = x[:, 3:]

    # Compute rotation matrices
    Z = euler_angles[:, 0]
    Y = euler_angles[:, 1]
    X = euler_angles[:, 2]

    cosZ = np.cos(Z)
    sinZ = np.sin(Z)
    cosY = np.cos(Y)
    sinY = np.sin(Y)
    cosX = np.cos(X)
    sinX = np.sin(X)

    Rz = np.stack([cosZ, -sinZ, np.zeros_like(Z),
                   sinZ, cosZ, np.zeros_like(Z),
                   np.zeros_like(Z), np.zeros_like(Z), np.ones_like(Z)], axis=1).reshape(N, 3, 3)

    Ry = np.stack([cosY, np.zeros_like(Y), sinY,
                   np.zeros_like(Y), np.ones_like(Y), np.zeros_like(Y),
                   -sinY, np.zeros_like(Y), cosY], axis=1).reshape(N, 3, 3)

    Rx = np.stack([np.ones_like(X), np.zeros_like(X), np.zeros_like(X),
                   np.zeros_like(X), cosX, -sinX,
                   np.zeros_like(X), sinX, cosX], axis=1).reshape(N, 3, 3)

    # Combined rotation matrix R = Rz @ Ry @ Rx
    R = np.einsum('nij,njk,nkl->nil', Rz, Ry, Rx)

    # Initialize homogeneous transformation matrices
    poses = np.zeros((N, 4, 4))
    poses[:, :3, :3] = R
    poses[:, :3, 3] = translations
    poses[:, 3, 3] = 1.0

    # Transpose to (4, 4, N)
    return poses.transpose(1, 2, 0)


