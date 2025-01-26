import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path

from calib_commons.data.data_pickle import save_to_pickle
from calib_commons.eval_generic_scene import eval_generic_scene
from calib_commons.utils.generateCircularCameras import generateCircularCameras
from calib_commons.camera import Camera
from calib_commons.intrinsics import Intrinsics
from calib_commons.utils.se3 import SE3
from calib_commons.viz import visualization as generic_vizualization

from calib_proj.utils.sceneGenerator import SceneGenerator
from calib_proj.utils import visualization
from calib_proj.core.externalCalibrator2 import ExternalCalibrator2, WorldFrame, SolvingLevel
from calib_proj.core.config import ExternalCalibratorConfig, SolvingLevel



def create_cameras(n_cameras):
    cameras = {}
    K_gopro = np.array([[917, 0, 1920/2], 
                      [0, 917, 1080/2], 
                      [0, 0, 1]])
    intrinsics_gopro = Intrinsics(K_gopro, (1920, 1080))

    n_far = 4
    d_far = 1
    tilt_far = 45*np.pi/180
  
    point_to_look_at = np.zeros(3)
    poses = generateCircularCameras(point_to_look_at, d_far, tilt_far, n_far)
    # for i in range(len(poses)):            
    pose = poses[1]
    id = 'A'
    cameras[id] = Camera(id, SE3(pose), intrinsics_gopro) 


    n_far = 4
    d_far = 3.5
    tilt_far = 45*np.pi/180
  
    point_to_look_at = np.zeros(3)
    poses = generateCircularCameras(point_to_look_at, d_far, tilt_far, n_far)
    pose = poses[2]
    id = 'B'
    cameras[id] = Camera(id, SE3(pose), intrinsics_gopro) 

    
    n_far = 4
    d_far = 7
    tilt_far = 45*np.pi/180
  
    point_to_look_at = np.zeros(3)
    poses = generateCircularCameras(point_to_look_at, d_far, tilt_far, n_far)
    pose = poses[3]
    id = 'C' 
    cameras[id] = Camera(id, SE3(pose), intrinsics_gopro) 

    return cameras

# random seed
np.random.seed(3)


############################### USER INTERFACE ####################################

# SYNTHETIC SCENE GENERATION PARAMETERS
num_points = 100
noise_std = 0.5 # [pix]
n_cameras = 5


# CALIBRATION PARAMETERS
external_calibrator_config = ExternalCalibratorConfig(
    SOLVING_LEVEL=SolvingLevel.FREE,
    reprojection_error_threshold = 1,
    ba_least_square_ftol = 1e-6, # Non linear Least-Squares 
    camera_score_threshold = 200,
    display=1, 
    display_reprojection_errors=0
)
out_folder_calib = Path("results")
show_viz = 1
save_viz = 0
save_eval_metrics_to_json = 1



###################### SYNTHETIC SCENE GENERATION ###########################

cameras = create_cameras(n_cameras)
scene_generator = SceneGenerator(num_points=num_points,
                                cameras=cameras,
                                noise_std=noise_std, 
                                std_on_3d_points=0.                                
                                )
synthetic_scene, points_I0 = scene_generator.generate_scene(world_frame=None)
correspondences = synthetic_scene.generic_scene.get_correspondences()
intrinsics = synthetic_scene.generic_scene.get_intrinsics()

visualization.visualize_scenes(synthetic_scene, show_ids=False, show_fig=show_viz, save_fig=False)


###################### EXTERNAL CALIBRATION ###########################
out_folder_calib.mkdir(parents=True, exist_ok=True)
external_calibrator = ExternalCalibrator2(correspondences=correspondences, 
                                        intrinsics=intrinsics, 
                                        # min_track_length=min_track_length, 
                                        points_I0=points_I0,
                                        config=external_calibrator_config
                                        )

external_calibrator.calibrate()
scene_proj_estimate = external_calibrator.get_scene(world_frame=WorldFrame.CAM_FIRST_CHOOSEN)
generic_scene = scene_proj_estimate.generic_scene
generic_obsv = external_calibrator.correspondences
generic_scene.print_cameras_poses()



# Save files
generic_scene.save_cameras_poses_to_json(out_folder_calib / "camera_poses.json")
print("camera poses saved to", out_folder_calib / "camera_poses.json")
scene_estimate_file = out_folder_calib / "scene_estimate.pkl"
save_to_pickle(scene_estimate_file, generic_scene)
print("scene estimate saved to", scene_estimate_file)
correspondences_file = out_folder_calib / "correspondences.pkl"
save_to_pickle(correspondences_file, generic_obsv)
print("correspondences saved to", correspondences_file)
# metrics = eval_generic_scene(generic_scene, generic_obsv, camera_groups=None, save_to_json=save_eval_metrics_to_json, output_path=out_folder_calib / "metrics.json", print_ = True)
print("")

# Visualization
if show_viz or save_viz:
    dpi = 300
    save_path = out_folder_calib / "scene.png"
    visualization.visualize_scenes(scene_proj_estimate, show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
    # visualization.visualize_scenes([checkerboard_scene_estimate], show_ids=False, show_fig=show_viz, save_fig=save_viz, save_path=save_path)
    if save_viz:
        print("scene visualization saved to", save_path)
    save_path = out_folder_calib / "2d.png"
    visualization.visualize_2d(scene=scene_proj_estimate, observations=generic_obsv, show_only_points_with_both_obsv_and_repr=0,  show_ids=False, which="both", show_fig=show_viz, save_fig=save_viz, save_path=save_path)

    if save_viz:
        print("2d visualization saved to", save_path)
    save_path = out_folder_calib / "2d_errors.png"
    generic_vizualization.plot_reprojection_errors(scene_estimate=generic_scene, 
                                        observations=generic_obsv, 
                                        show_fig=show_viz,
                                        save_fig=save_viz, 
                                        save_path=save_path)
    if save_viz:
        print("2d errors visualization saved to", save_path)

    if show_viz:
        plt.show()

