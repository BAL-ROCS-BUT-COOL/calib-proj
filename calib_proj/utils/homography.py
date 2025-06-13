import cv2
import numpy as np
import calib_commons.utils.se3 as se3


def estimate_inter_image_homography_normalized(K1: np.ndarray, 
                                               K2: np.ndarray, 
                                               pts_cam1: dict, 
                                               pts_cam2: dict) -> np.ndarray: 
    """
    Estimate the normalized homography between two camera views.

    1. Find the set of point-IDs observed by both cameras.
    2. Back-project those image points into the normalized plane
       by applying K⁻¹ to each (u,v,1) pixel.
    3. Run RANSAC homography estimation on the normalized points.

    Args:
        K1: 3×3 intrinsics of camera 1.
        K2: 3×3 intrinsics of camera 2.
        pts_cam1: {pt_id: (u,v)} observations in image 1.
        pts_cam2: {pt_id: (u,v)} observations in image 2.

    Returns:
        H_norm: 3×3 homography mapping normalized coords of cam1 → cam2,
                      or None if not enough matches.
    """
    # Get all point-IDs seen by both cams
    common_points_ids = list(set(pts_cam1.keys()) & set(pts_cam2.keys()))
    if len(common_points_ids) < 4: 
        print(f"Cannot estimate homography with less than 4 common points.")
        return None
    
    pts_src = np.array([pts_cam1[point_id] for point_id in common_points_ids], dtype='float32').squeeze()
    pts_dst = np.array([pts_cam2[point_id] for point_id in common_points_ids], dtype='float32').squeeze()

    # Normalize to camera-centric coordinates: (x,y,1) = K⁻¹ · (u,v,1)
    pts_src_normalized = (np.linalg.inv(K1) @ np.vstack((pts_src.T, np.ones(pts_src.shape[0]))) )[:2].T
    pts_dst_normalized = (np.linalg.inv(K2) @ np.vstack((pts_dst.T, np.ones(pts_dst.shape[0]))) )[:2].T
    
    # Compute homography in normalized space with RANSAC
    H_norm, _ = cv2.findHomography(pts_src_normalized, pts_dst_normalized, cv2.RANSAC)
    
    return H_norm
        

def decompose_inter_image_homography_normalized(H_norm: np.ndarray,
                                               K1: np.ndarray,
                                               K2: np.ndarray,
                                               pts_cam1: dict,
                                               pts_cam2: dict,
                                               verbose: int = 0) -> list[tuple[se3.SE3, np.ndarray]]:
    """
    Decompose a normalized homography into candidate relative poses.

    1. Call OpenCV’s decomposeHomographyMat with K=I
       → yields lists of (R₂₁, t₂₁, n₁) solutions.
    2. Convert each to SE3, then count how many points
       project in front of **both** cameras for disambiguation.
    3. Return only the solution(s) with maximal “in-front” count.

    Args:
        H_norm: normalized 3×3 homography.
        K1,K2: intrinsics (used to test front-side via K⁻¹).
        pts_cam1, pts_cam2: {pt_id: (u,v)} for each cam.
        verbose: debug verbosity.

    Returns:
        List of tuples (SE3_of_cam2_in_cam1_frame, plane_normal_in_cam1_frame).
    """
    decompositions = cv2.decomposeHomographyMat(H_norm, K=np.eye(3))
    if not decompositions[0]:
        print("Homography decomposition failed.")
        return None
    
    number_in_front_list = []
    cam2_pose_list = []
    n_1_list = []
    num_solutions = len(decompositions[1])
    if verbose >= 2:
        print(f"Found {num_solutions} possible decompositions.")
    for i, (R_2_1, t_2_1, n_1) in enumerate(zip(decompositions[1], decompositions[2], decompositions[3])):
        T_2_1 = se3.T_from_rt(R_2_1, t_2_1)
        T_W_2 = se3.inv_T(T_2_1) # T^W_2
        cam2_pose = se3.SE3(T_W_2)
        cam2_pose_list.append(cam2_pose)
        n_1_list.append(n_1)

        # solution disambiguation: count number of points in front of each camera
        # cam1
        in_front_cam1 = []
        for pt in pts_cam1.values():
            m = np.linalg.inv(K1) @ np.append(pt,1)
            in_front_cam1.append(m.dot(n_1) > 0)
        # cam2
        in_front_cam2 = []
        n_2 = R_2_1 @ n_1
        for pt in pts_cam2.values():
            m = np.linalg.inv(K2) @ np.append(pt,1)
            in_front_cam2.append(m.dot(n_2) > 0)
        in_front_all = in_front_cam1 + in_front_cam2
        number_in_front = int(sum(in_front_all))
        number_in_front_list.append(number_in_front)

        if verbose >= 2:

            print(f"Solution {i+1} : total number of points in front of a camera: {number_in_front} among {len(in_front_all)}")

    sol_indices = [i for i, n in enumerate(number_in_front_list) if n == max(number_in_front_list)]
    if len(sol_indices) == 1:
        if verbose >= 2:
            print("Solution is unique.")
    else:
        if verbose >= 2:
            print("There are two solutions from decomposing the inter-image homography that have the same number of points triangulated with positive depth in both cameras.") 
    return [(cam2_pose_list[i], n_1_list[i]) for i in sol_indices]
        
def retrieve_motion_using_homography(K1: np.ndarray,
                                     K2: np.ndarray,
                                     pts_cam1: dict,
                                     pts_cam2: dict,
                                     verbose: int = 0) -> list[tuple[se3.SE3, np.ndarray]]:
    """
    Given intrinsics K1,K2 and matched 2D points, estimate the possible
    relative motions and plane normals between cam1→cam2 via homography.

    Steps:
      1. Estimate normalized homography H_norm.
      2. Decompose H_norm into rotations, translations, normals.
      3. Return only the solution(s) with the most points in front of both cams.

    Returns:
        A list of (SE3_pose_of_cam2, plane_normal_in_cam1_frame), possibly length 1 or 2.
        Or None if homography fails.
    """
    H_norm = estimate_inter_image_homography_normalized(K1, K2, pts_cam1, pts_cam2)
    if H_norm is None:
        return None
    sols = decompose_inter_image_homography_normalized(H_norm, K1, K2, pts_cam1, pts_cam2, verbose=verbose)
    return sols
   