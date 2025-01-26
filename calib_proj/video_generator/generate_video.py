import numpy as np
from typing import List
import cv2
import os
from tqdm import tqdm

from calib_proj.video_generator.marker_generator.aruco_generator import generate_aruco_markers
from calib_proj.video_generator.generate_grid import generate_grid


def save_seq_info_json(seq_info, seq_path): 
    import json
    with open(seq_path, 'w') as f:
        json.dump(seq_info, f)

def load_seq_info_json(seq_path):
    import json
    with open(seq_path, 'r') as f:
        seq_info = json.load(f)
    return seq_info

def generate_video(msm_base_size: int, 
                   msm_scales: List[int],
                   n_shifts: int, 
                   projector_resolution: tuple[int, int], 
                   grid_size: tuple[int, int], 
                   marker_system: str = 'aruco_4X4_50', 
                   video_folder: str = None, 
                   grid_fps: int = 10, 
                   white_duration: int = 1, 
                   invert_colors: bool = True):
    
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    msm_sizes = [msm_base_size * scale for scale in msm_scales]

    s_w = projector_resolution[0] / (grid_size[1]+1)
    s_h = projector_resolution[1] / (grid_size[0]+1)
    s = int(min(s_h, s_w))
  
    shift_range_x = s/2
    shift_range_y = s/2

    shifts = [(int(shift_x), int(shift_y)) 
              for shift_x in np.linspace(-shift_range_x, shift_range_x, n_shifts) 
              for shift_y in np.linspace(-shift_range_y, shift_range_y, n_shifts)]
    
    ids = range(grid_size[0] * grid_size[1])

    # tmp_folder = video_folder + "\\tmp"
    # marker_folder = tmp_folder + "\\" + marker_system


    seq_info = {'marker_system': marker_system,
                'grid_size': grid_size,
                'grid_fps': grid_fps,
                'msm_base_size': msm_base_size,
                'msm_scales': msm_scales,
                'invert_colors': invert_colors,
                'shift_scale_indices': {}}
    
    marker_system_name, *rest = marker_system.split('_', 1)
    dict = rest[0] 

    # generate base markers
    markers = generate_aruco_markers(msm_sizes,
                                        ids,
                                        dictionary = dict, 
                                        save = False)
    
    # generate video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(video_folder, 'video.mp4')
    video_writer = cv2.VideoWriter(video_path, fourcc, grid_fps, projector_resolution)

    # Add white image at the beginning of the video
    white_frame = np.ones((projector_resolution[1], projector_resolution[0], 3), dtype=np.uint8) 
    white_frame[:,:,0] = 0
    white_frame[:,:,1] = 0
    white_frame[:,:,2] = 255
    white_frame_count = int(grid_fps * white_duration)
    for _ in range(white_frame_count):
        video_writer.write(white_frame)


    grid_idx = 1
    for shift_idx, shift in tqdm(enumerate(shifts), total=len(shifts), desc="Generating video frames"):
        for scale_idx, scale in enumerate(msm_scales):
            seq_info['shift_scale_indices'][grid_idx] = (shift_idx, scale_idx+1)

            marker_size = msm_base_size * scale
            # base_folder_path = os.path.join(marker_folder, str(marker_size))
            grid_img, _ = generate_grid(grid_size=grid_size, 
                                        size_in_out_image=marker_size, 
                                        margin_min=0, 
                                        out_image_res=projector_resolution,
                                        shift=shift, 
                                        markers=markers)
            grid_img = cv2.cvtColor(np.array(grid_img), cv2.COLOR_RGB2BGR)
            if invert_colors:
                grid_img = 255 - grid_img
            video_writer.write(grid_img)
            
            grid_idx += 1

    # Add white image at the end of the video
    white_frame_count = int(grid_fps * white_duration)
    for _ in range(white_frame_count):
        video_writer.write(white_frame)

    seq_info_path = os.path.join(video_folder, 'seq_info.json')
    save_seq_info_json(seq_info, seq_info_path)


    video_writer.release()

    print(f"Video generated successfully. Video saved at {video_path}")
    print(f"Sequence info saved at {seq_info_path}")



