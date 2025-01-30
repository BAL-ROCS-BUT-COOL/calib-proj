import cv2
import os
from calib_proj.video_generator.generate_video import load_seq_info_json
from tqdm import tqdm

def extract_frames(videos_folder, sequence_info_path, start_end_frames, out_folder_path: str): 
    """Extract frames from videos based on start and end frames.
    
    Args:
    videos_path (dict): Dictionary containing video names as keys and video paths as values.
    start_end_frames (dict): Dictionary containing start and end frames for each video and sequence.
    
    Returns:
    None
    """
    print(f"\nExtracting synchronized frames started...")

    videos_path = {}
    for video_file in os.listdir(videos_folder):
        if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(videos_folder, video_file)
            videos_path[video_file.split('.')[0]] = video_path

    seq_info = load_seq_info_json(sequence_info_path)
    

    for cam_name, video_path in tqdm(videos_path.items(), desc="Extraction"):
        start_frame = start_end_frames[cam_name]['start']
        end_frame = start_end_frames[cam_name]['end']

        if start_frame is None or end_frame is None:
            print(f"Skipping video {cam_name} as start or end frame could not be detected.")
            continue
        print(f" Processing {cam_name}...")
      

        n_grids = len(seq_info['shift_scale_indices'])
        n_frames_per_grid = (end_frame - start_frame) / n_grids
        frames_indices = {grid_idx: round((start_frame + (grid_idx + 0.5)* n_frames_per_grid)) for grid_idx in range(n_grids)}



        video_folder_path = out_folder_path / cam_name
        video_folder_path.mkdir(parents=True, exist_ok=True)

        video_capture = cv2.VideoCapture(str(video_path))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame


        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_idx in frames_indices.values():
                grid_id = list(frames_indices.keys())[list(frames_indices.values()).index(frame_idx)]+1
                filename = f"{grid_id:06d}.png"
                filepath = video_folder_path / filename
                cv2.imwrite(filepath, frame)
            frame_idx += 1
           
        video_capture.release()
        cv2.destroyAllWindows()
    print("Frames extracted successfully.")
    return None