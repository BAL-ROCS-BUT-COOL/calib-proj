import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from calib_commons.data.load_calib import construct_cameras_intrinsics
from calib_commons.data.data_pickle import save_to_pickle, load_from_pickle
from calib_commons.eval_generic_scene import eval_generic_scene
from calib_commons.viz import visualization as generic_vizualization
from calib_commons.world_frame import WorldFrame

from calib_proj.utils import visualization
from calib_proj.core.external_calibrator import ExternalCalibrator
from calib_proj.core.config import ExternalCalibratorConfig
from calib_proj.utils.data import convert_to_correspondences
from calib_proj.preprocessing.detection import detect_marker_centers
from calib_proj.preprocessing.preprocess import order_centers, msm_centers_from_marker_centers
from calib_proj.video_generator.generate_video import load_seq_info_json
from calib_proj.synch.synch import synch
from calib_proj.synch.extract_frames import extract_frames

# Set random seed for reproducibility
np.random.seed(3)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run external projector-camera calibration with optional caching of synchronization and frame extraction steps"
    )
    parser.add_argument(
        "--videos_folder", type=Path, required=True,
        help="Path to folder containing video files"
    )
    parser.add_argument(
        "--intrinsics_folder", type=Path, required=True,
        help="Path to folder containing camera intrinsics"
    )
    parser.add_argument(
        "--sequence_info", type=Path, default=Path("video/seq_info.json"),
        help="Path to sequence info JSON file"
    )
    parser.add_argument(
        "--out_folder", type=Path, default=Path("results"),
        help="Output folder for calibration results"
    )
    parser.add_argument(
        "--sync_start", type=float, default=0.0,
        help="(in seconds) start time for synchronization search. Default=0.0"
    )
    parser.add_argument(
        "--sync_end", type=float, default=None,
        help="(in seconds) end time for synchronization search. Default=None (i.e., entire video)"
    )
    parser.add_argument(
        "--sync_threshold", type=float, default=0.6,
        help="Threshold for temporal synchronization"
    )
    parser.add_argument(
        "--force_sync", action="store_true",
        help="Force recompute temporal synchronization even if cache exists"
    )
    parser.add_argument(
        "--force_extract", action="store_true",
        help="Force re-extraction of frames even if already extracted"
    )
    parser.add_argument(
        "--force_detect", action="store_true",
        help="Force re-detection of marker centers even if cached"
    )
    parser.add_argument(
        "--show_detection", action="store_true",
        help="Show detection images"
    )
    parser.add_argument(
        "--show_viz", action="store_true",
        help="Show final visualizations"
    )
    parser.add_argument(
        "--save_viz", action="store_true",
        help="Save final visualizations"
    )
    parser.add_argument(
        "--reproj_error_threshold", type=float, default=1.0,
        help="Reprojection error threshold for external calibrator"
    )
    parser.add_argument(
        "--camera_score_threshold", type=float, default=200.0,
        help="Camera score threshold for external calibrator"
    )
    parser.add_argument(
        "--bootstrap_cams", nargs=2, metavar=("CAM1", "CAM2"), type=str,
        help=(
            "Force those two camera IDs to be the initial bootstrap pair. "
            "Example: --bootstrap_cams gopro1 gopro3"
        )
    )
    return parser.parse_args()


def run_synchronization(
        videos_folder: Path,
        seq_info_path: Path,
        threshold: float,
        cache_path: Path,
        force: bool,
        start_time: float = 0.0,
        end_time: float = None
    ):
    if cache_path.exists() and not force:
        print(f"Loading cached synchronization from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Running temporal synchronization (time‐window {start_time}s to {end_time if end_time is not None else 'end'})...")
    # Pass the two new parameters down into synch()
    start_end_frames = synch(
        videos_folder,
        seq_info_path,
        threshold=threshold,
        start_time=start_time,
        end_time=end_time
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(start_end_frames, f)
    print(f"Synchronization cached to {cache_path}")
    return start_end_frames



def run_frame_extraction(videos_folder: Path, seq_info_path: Path, start_end_frames, frames_folder: Path, force: bool):
    if frames_folder.exists() and not force:
        print(f"Frames already extracted in {frames_folder}, skipping extraction")
        return
    print("Extracting frames from videos...")
    extract_frames(videos_folder, seq_info_path, start_end_frames, frames_folder)
    print(f"Extracted frames to {frames_folder}")


def run_detection(frames_folder: Path, intrinsics_folder: Path, seq_info: dict,
                  cache_path: Path, show: bool, force: bool):
    if cache_path.exists() and not force:
        print(f"Loading cached marker centers from {cache_path}")
        return load_from_pickle(cache_path)
    print("Detecting marker centers...")
    centers = detect_marker_centers(
        frames_folder,
        intrinsics_folder,
        marker_system=seq_info['marker_system'],
        inverted_projections=seq_info['invert_colors'],
        show_detections=show
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_pickle(cache_path, centers)
    print(f"Marker centers cached to {cache_path}")
    return centers


def run_calibration(msm_centers, frames_folder: Path, intrinsics_folder: Path,
                    out_folder: Path, config: ExternalCalibratorConfig,
                    show_viz: bool, save_viz: bool,
                    save_scene: bool, save_corr: bool,
                    save_metrics: bool):
    out_folder.mkdir(parents=True, exist_ok=True)
    intrinsics = construct_cameras_intrinsics(frames_folder, intrinsics_folder)
    correspondences = convert_to_correspondences(msm_centers)
    calibrator = ExternalCalibrator(
        correspondences=correspondences,
        intrinsics=intrinsics,
        config=config
    )
    print("Calibration started...")
    success = calibrator.calibrate()
    if not success:
        print("Calibration failed.")
        return
    scene_est = calibrator.get_scene(world_frame=WorldFrame.CAM_FIRST_CHOOSEN)
    generic_scene = scene_est.generic_scene
    generic_obsv = calibrator.correspondences
    generic_scene.print_cameras_poses()

    poses_path = out_folder / "camera_poses.json"
    generic_scene.save_cameras_poses_to_json(poses_path)
    print(f"Camera poses saved to {poses_path}")

    if save_scene:
        scene_file = out_folder / "scene_estimate.pkl"
        save_to_pickle(scene_file, generic_scene)
        print(f"Scene saved to {scene_file}")
    if save_corr:
        corr_file = out_folder / "correspondences.pkl"
        save_to_pickle(corr_file, generic_obsv)
        print(f"Correspondences saved to {corr_file}")

    eval_generic_scene(
        generic_scene,
        generic_obsv,
        camera_groups=None,
        save_to_json=save_metrics,
        output_path=out_folder / "metrics.json",
        print_=True
    )

    if show_viz or save_viz:
        viz_path = out_folder / "scene.png"
        visualization.visualize_scenes(
            scene_est,
            show_ids=False,
            show_fig=show_viz,
            save_fig=save_viz,
            save_path=viz_path
        )
        if save_viz:
            print(f"Scene visualization saved to {viz_path}")

        viz2d_path = out_folder / "2d.png"
        visualization.visualize_2d(
            scene=scene_est,
            observations=generic_obsv,
            show_only_points_with_both_obsv_and_repr=0,
            show_ids=False,
            which="both",
            show_fig=show_viz,
            save_fig=save_viz,
            save_path=viz2d_path
        )
        if save_viz:
            print(f"2D visualization saved to {viz2d_path}")

        err_path = out_folder / "2d_errors.png"
        generic_vizualization.plot_reprojection_errors(
            scene_estimate=generic_scene,
            observations=generic_obsv,
            show_fig=show_viz,
            save_fig=save_viz,
            save_path=err_path
        )
        if save_viz:
            print(f"Reprojection errors plot saved to {err_path}")

        if show_viz:
            plt.show()


def main():
    args = parse_args()

    sync_cache = args.out_folder / "sync" / "start_end_frames.pkl"
    frames_folder = args.videos_folder / "frames"
    seq_info = load_seq_info_json(args.sequence_info)

    start_end_frames = run_synchronization(
        args.videos_folder,
        args.sequence_info,
        threshold=args.sync_threshold,
        cache_path=sync_cache,
        force=args.force_sync,
        start_time=args.sync_start,
        end_time=args.sync_end
    )

    print(start_end_frames)

    run_frame_extraction(
        args.videos_folder,
        args.sequence_info,
        start_end_frames,
        frames_folder,
        force=args.force_extract
    )

    centers_cache = args.out_folder / "preprocessing" / "centers_unordered.pkl"
    centers_unordered = run_detection( # PROBLEMATIC FUNCTION
        frames_folder,
        args.intrinsics_folder,
        seq_info,
        cache_path=centers_cache,
        show=args.show_detection,
        force=args.force_detect
    )

    centers_ordered = order_centers(centers_unordered, seq_info)
    msm_centers = msm_centers_from_marker_centers(centers_ordered)



    calib_config = ExternalCalibratorConfig(
        reprojection_error_threshold=args.reproj_error_threshold,
        camera_score_threshold=args.camera_score_threshold,
        verbose=1,
        least_squares_verbose=0,
    )

    run_calibration(
        msm_centers,
        frames_folder,
        args.intrinsics_folder,
        args.out_folder,
        config=calib_config,
        show_viz=args.show_viz,
        save_viz=args.save_viz,
        save_scene=False,
        save_corr=False,
        save_metrics=True
    )

if __name__ == "__main__":
    main()
