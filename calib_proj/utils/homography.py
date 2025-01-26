import cv2
import numpy as np
import calib_commons.utils.se3 as se3


def estimate_inter_image_homography_normalized(K1, K2, pts_cam1, pts_cam2): 
    common_points_ids = list(set(pts_cam1.keys()) & set(pts_cam2.keys()))
    if len(common_points_ids) >= 4: 
        pts_src = np.array([pts_cam1[point_id] for point_id in common_points_ids], dtype='float32').squeeze()
        pts_dst = np.array([pts_cam2[point_id] for point_id in common_points_ids], dtype='float32').squeeze()
        pts_src_normalized = (np.linalg.inv(K1) @ np.vstack((pts_src.T, np.ones(pts_src.shape[0]))) )[:2].T
        pts_dst_normalized = (np.linalg.inv(K2) @ np.vstack((pts_dst.T, np.ones(pts_dst.shape[0]))) )[:2].T
        H_normalized, _ = cv2.findHomography(pts_src_normalized, pts_dst_normalized, cv2.RANSAC)
        return H_normalized
    else:
        print(f"Cannot estimate homography with less than 4 common points.")
        return None
        
def retrieve_motion_using_homography(K1, K2, pts_cam1, pts_cam2, verbose=0):
    H_normalized = estimate_inter_image_homography_normalized(K1, K2, pts_cam1, pts_cam2)
    if H_normalized is None:
        return None
    sols = decompose_inter_image_homography_normalized(H_normalized, K1, K2, pts_cam1, pts_cam2, verbose=verbose)
    return sols
    

def decompose_inter_image_homography_normalized(H_normalized, K1, K2, pts_cam1, pts_cam2, verbose=0):
    decompositions = cv2.decomposeHomographyMat(H_normalized, K=np.eye(3))
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
   