


import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

class BA_Free:
    """Python class for Simple Bundle Adjustment using Rotation Vectors for rotation."""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, intrinsics, ftol = 1e-8, verbose=0):
        """
        Initializes all the class attributes and instance variables.

        cameraArray with shape (n_cameras, 6) contains initial estimates of the rotation (rotvec) 
        and translation for all cameras. The first camera is fixed to identity.

        points3D with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.

        cameraIndices with shape (n_observations,) contains indices of cameras involved in each observation.

        pointIndices with shape (n_observations,) contains indices of points involved in each observation.

        points2D with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images.

        intrinsics is a list containing dictionaries of fx, fy, cx, cy for each camera.
        """
        self.verbose = verbose

        self.ftol = ftol

        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D
        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

        # Store intrinsics as separate numpy arrays
        self.fx = np.array([intrinsics[i]['fx'] for i in range(len(intrinsics))])
        self.fy = np.array([intrinsics[i]['fy'] for i in range(len(intrinsics))])
        self.cx = np.array([intrinsics[i]['cx'] for i in range(len(intrinsics))])
        self.cy = np.array([intrinsics[i]['cy'] for i in range(len(intrinsics))])

        # Fix the first camera pose (rotation vector = [0, 0, 0], translation = [0, 0, 0])
        self.fixed_camera = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Identity pose for first camera

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors using Rodrigues' formula."""
        rotation_matrices = R.from_rotvec(rot_vecs).as_matrix()
        return np.einsum('nij,nj->ni', rotation_matrices, points)  # Apply rotation matrices to points

    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images using fixed intrinsic parameters."""
        points_proj = self.rotate(points, cameraArray[:, :3])  # Rotate using rotation vectors
        points_proj += cameraArray[:, 3:6]  # Apply translation

        # Use the pre-stored fx, fy, cx, cy without a for loop
        fx = self.fx[self.cameraIndices]
        fy = self.fy[self.cameraIndices]
        cx = self.cx[self.cameraIndices]
        cy = self.cy[self.cameraIndices]

        # Convert 3D points to 2D using perspective projection
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        points_proj[:, 0] = points_proj[:, 0] * fx + cx
        points_proj[:, 1] = points_proj[:, 1] * fy + cy

        return points_proj

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals between projected and observed 2D points."""
        # Concatenate the fixed first camera pose with the remaining camera poses
        camera_params = np.vstack((self.fixed_camera, params[:(n_cameras - 1) * 6].reshape((n_cameras - 1, 6))))
        points_3d = params[(n_cameras - 1) * 6:].reshape((n_points, 3))

        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        """Create the sparsity structure for the Jacobian matrix."""
        m = cameraIndices.size * 2
        n = (numCameras - 1) * 6 + numPoints * 3  # Exclude the first camera from the optimization
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        # Only add sparsity for cameras excluding the first one
        for s in range(6):
            mask = cameraIndices > 0  # Ignore the first camera (index 0)
            A[2 * i[mask], (cameraIndices[mask] - 1) * 6 + s] = 1
            A[2 * i[mask] + 1, (cameraIndices[mask] - 1) * 6 + s] = 1

        for s in range(3):
            A[2 * i, (numCameras - 1) * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, (numCameras - 1) * 6 + pointIndices * 3 + s] = 1

        return A

    def optimizedParams(self, params, n_cameras, n_points):
        """Retrieve optimized camera parameters and 3D coordinates."""
        # First camera is fixed to identity, so we add it back here
        camera_params = np.vstack((self.fixed_camera, params[:(n_cameras - 1) * 6].reshape((n_cameras - 1, 6))))
        points_3d = params[(n_cameras - 1) * 6:].reshape((n_points, 3))
        return camera_params, points_3d

    def bundleAdjust(self):
        """Perform bundle adjustment to optimize rotation (rotvec) and translation vectors."""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        # Initial guess: Flatten the camera poses (excluding the first one) and 3D points arrays
        x0 = np.hstack((self.cameraArray[1:].ravel(), self.points3D.ravel()))  # Skip the first camera's parameters

        # Compute the initial residuals
        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        # Optimize the parameters (excluding the first camera)
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=self.verbose, x_scale='jac', ftol=self.ftol, method='trf', 
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))
        params = self.optimizedParams(res.x, numCameras, numPoints)
        return params

class BA_PlaneConstraint:
    """Python class for Bundle Adjustment with points constrained on a plane."""

    def __init__(self, cameraArray, plane_pose, points2D_in_plane, points2D, cameraIndices, point2DIndices, intrinsics, ftol = 1e-8, verbose=0):
        """
        Initializes all the class attributes and instance variables.

        cameraArray with shape (n_cameras, 6) contains initial estimates of the rotation (rotvec)
        and translation for all cameras. The first camera is fixed to identity.

        plane_pose with shape (6,) contains initial estimates of the rotation (rotvec) and translation of the plane.

        points2D_in_plane with shape (n_points, 2) contains initial estimates of point 2D coordinates in the plane frame.

        points2D with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images.

        cameraIndices with shape (n_observations,) contains indices of cameras involved in each observation.

        pointIndices with shape (n_observations,) contains indices of points involved in each observation.

        intrinsics is a list containing dictionaries of fx, fy, cx, cy for each camera.
        """

        self.ftol = ftol
        self.verbose = verbose
        
        self.cameraArray = cameraArray
        self.plane_pose = plane_pose
        self.points2D_in_plane = points2D_in_plane
        self.points2D = points2D
        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

        # Store intrinsics as separate numpy arrays
        self.fx = np.array([intrinsics[i]['fx'] for i in range(len(intrinsics))])
        self.fy = np.array([intrinsics[i]['fy'] for i in range(len(intrinsics))])
        self.cx = np.array([intrinsics[i]['cx'] for i in range(len(intrinsics))])
        self.cy = np.array([intrinsics[i]['cy'] for i in range(len(intrinsics))])

        # Fix the first camera pose (rotation vector = [0, 0, 0], translation = [0, 0, 0])
        self.fixed_camera = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Identity pose for first camera

    def plane_to_world(self, points2D, plane_pose):
        """Convert 2D points in the plane to 3D points in the world frame using plane pose."""
        # Convert 2D points in plane to 3D points (z = 0 in plane's frame)
        points3D_in_plane = np.column_stack((points2D, np.zeros(points2D.shape[0])))

        # Rotate and translate the points using the plane's pose
        rotation_matrix = R.from_rotvec(plane_pose[:3]).as_matrix()
        points3D_in_world = points3D_in_plane @ rotation_matrix.T + plane_pose[3:6]

        return points3D_in_world

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors using Rodrigues' formula."""
        rotation_matrices = R.from_rotvec(rot_vecs).as_matrix()
        return np.einsum('nij,nj->ni', rotation_matrices, points)  # Apply rotation matrices to points

    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images using fixed intrinsic parameters."""
        points_proj = self.rotate(points, cameraArray[:, :3])  # Rotate using rotation vectors
        points_proj += cameraArray[:, 3:6]  # Apply translation

        # Use the pre-stored fx, fy, cx, cy without a for loop
        fx = self.fx[self.cameraIndices]
        fy = self.fy[self.cameraIndices]
        cx = self.cx[self.cameraIndices]
        cy = self.cy[self.cameraIndices]

        # Convert 3D points to 2D using perspective projection
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        points_proj[:, 0] = points_proj[:, 0] * fx + cx
        points_proj[:, 1] = points_proj[:, 1] * fy + cy

        return points_proj

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals between projected and observed 2D points."""
        camera_params = np.vstack((self.fixed_camera, params[:(n_cameras - 1) * 6].reshape((n_cameras - 1, 6))))
        plane_pose = params[(n_cameras - 1) * 6:(n_cameras - 1) * 6 + 6]
        points_2d_plane = params[(n_cameras - 1) * 6 + 6:].reshape((n_points, 2))

        points_3d = self.plane_to_world(points_2d_plane, plane_pose)
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        """Create the sparsity structure for the Jacobian matrix."""
        m = cameraIndices.size * 2
        n = (numCameras - 1) * 6 + 6 + numPoints * 2  # Exclude the first camera from the optimization, include plane params
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        # Only add sparsity for cameras excluding the first one
        for s in range(6):
            mask = cameraIndices > 0  # Ignore the first camera (index 0)
            A[2 * i[mask], (cameraIndices[mask] - 1) * 6 + s] = 1
            A[2 * i[mask] + 1, (cameraIndices[mask] - 1) * 6 + s] = 1

        for s in range(6):
            A[2 * i, (numCameras - 1) * 6 + s] = 1
            A[2 * i + 1, (numCameras - 1) * 6 + s] = 1

        for s in range(2):
            A[2 * i, (numCameras - 1) * 6 + 6 + pointIndices * 2 + s] = 1
            A[2 * i + 1, (numCameras - 1) * 6 + 6 + pointIndices * 2 + s] = 1

        return A

    def optimizedParams(self, params, n_cameras, n_points):
        """Retrieve optimized camera parameters, plane pose, and 2D coordinates."""
        camera_params = np.vstack((self.fixed_camera, params[:(n_cameras - 1) * 6].reshape((n_cameras - 1, 6))))
        plane_pose = params[(n_cameras - 1) * 6:(n_cameras - 1) * 6 + 6]
        points_2d_plane = params[(n_cameras - 1) * 6 + 6:].reshape((n_points, 2))
        return camera_params, plane_pose, points_2d_plane

    def bundleAdjust(self):
        """Perform bundle adjustment to optimize rotation, translation vectors, and plane parameters."""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points2D_in_plane.shape[0]

        # Initial guess: Flatten the camera poses (excluding the first one), plane pose, and 2D points arrays
        x0 = np.hstack((self.cameraArray[1:].ravel(), self.plane_pose, self.points2D_in_plane.ravel()))

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        # Optimize the parameters (excluding the first camera)
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=self.verbose, x_scale='jac', ftol=self.ftol, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))
        params = self.optimizedParams(res.x, numCameras, numPoints)
        return params