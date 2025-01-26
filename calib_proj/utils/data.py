from calib_commons.observation import Observation

def convert_to_correspondences(dict_ndarray): 
    """
    Convert a dictionnary: Dict[str, Dict[str, np.ndarray]] to a dictionnary: Dict[str, Dict[str, Observation]]
    """
    correspondences = {}
    for cam_id, cam in dict_ndarray.items(): 
        correspondences[cam_id] = {}
        for pt_id, pt in cam.items(): 
            correspondences[cam_id][pt_id] = Observation(_2d=pt) 
    return correspondences

