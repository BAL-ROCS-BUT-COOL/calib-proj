import numpy as np

def order_centers(centers, seq_info): 
    ordered_centers = {}
    for cam in centers:
        ordered_centers[cam] = {}  
        for proj_idx in centers[cam]:
            shift_idx, scale_idx = seq_info['shift_scale_indices'][str(proj_idx)] 
            if shift_idx not in ordered_centers[cam].keys():
                ordered_centers[cam][shift_idx] = {}

            for marker_id in centers[cam][proj_idx]: 
                if marker_id not in ordered_centers[cam][shift_idx].keys(): 
                    ordered_centers[cam][shift_idx][marker_id] = {}
                ordered_centers[cam][shift_idx][marker_id][scale_idx] = centers[cam][proj_idx][marker_id]


    return ordered_centers


def msm_centers_from_marker_centers(centers): 
    msm_centers = {}
    for cam in centers: 
        msm_centers[cam] = {}
        for shift_idx in centers[cam]: 
            for marker_id in centers[cam][shift_idx]: 
                global_id = f"{shift_idx}_{marker_id}"
                centers_ = [centers[cam][shift_idx][marker_id][scale_idx] for scale_idx in centers[cam][shift_idx][marker_id]]
                msm_centers[cam][global_id] = np.median(centers_, axis=0)
    return msm_centers