import numpy as np
from typing import List
import cv2
import os
from tqdm import tqdm
from PIL import Image

from calib_proj.video_generator.marker_generator.aruco_generator import generate_aruco_markers
from calib_proj.video_generator.generate_grid import generate_grid
from calib_proj.synch.seq_generator import generate_sequences


def save_seq_info_json(seq_info, seq_path): 
    import json
    with open(seq_path, 'w') as f:
        json.dump(seq_info, f)

def load_seq_info_json(seq_path):
    import json
    with open(seq_path, 'r') as f:
        seq_info = json.load(f)
    return seq_info

def generate_video(
    msm_base_size: int, 
    msm_scales: List[int],
    n_grids: int, 
    projector_resolution: tuple[int, int], 
    grid_size: tuple[int, int], 
    marker_system: str = 'aruco_4X4_50', 
    video_folder: str = 'video', 
    grid_fps: int = 10, 
    video_fps: int = 30,
    white_duration: int = 1, 
    invert_colors: bool = True, 
    synch_seq_duration = 3, 
    save_grids: bool = False
):
    
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    
    msm_sizes = [msm_base_size * scale for scale in msm_scales]
    
    grids_coords = generate_grid(projector_resolution=projector_resolution,
                  margin=10,
                  grid_size=grid_size,
                  n_grids=n_grids)
    
    # # tmp_folder = video_folder + "\\tmp"
    # # marker_folder = tmp_folder + "\\" + marker_system

    ids = range(grid_size[0] * grid_size[1])

    seq_info = {'marker_system': marker_system,
                'grid_size': grid_size,
                'grid_fps': grid_fps,
                'video_fps': video_fps,
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
    video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, projector_resolution)

    # Add white image at the beginning of the video
    white_frame = np.ones((projector_resolution[1], projector_resolution[0], 3), dtype=np.uint8) 
    white_frame[:,:,0] = 0
    white_frame[:,:,1] = 0
    white_frame[:,:,2] = 255
    white_frame_count = int(video_fps * white_duration)
    for _ in range(white_frame_count):
        video_writer.write(white_frame)

    synch_sequences = generate_sequences(seq_duration=synch_seq_duration, seq_fps=10, proj_fps=video_fps)
    seq_info['synch_sequences'] = synch_sequences
    
    for value in synch_sequences['start']:
        intensity = 255 if value == 1 else 0
        image_array = np.full((projector_resolution[1], projector_resolution[0], 3), intensity, dtype=np.uint8)
        video_writer.write(image_array)

    repeat_grid_factor = video_fps // grid_fps

    if save_grids:
        grids_folder = os.path.join(video_folder, 'grids')
        if not os.path.exists(grids_folder):
            os.makedirs(grids_folder)

    grid_idx = 1
    for shift_idx, grid_coords in tqdm(grids_coords.items(), total=len(grids_coords), desc="Generating video frames"):
        for scale_idx, scale in enumerate(msm_scales):
            seq_info['shift_scale_indices'][grid_idx] = (shift_idx, scale_idx+1)
            marker_size = msm_base_size * scale

            cols, rows = grid_size
            grid_image = Image.new('RGB', projector_resolution, (255, 255, 255))
            id = 0
            for row in range(rows): 
                for col in range(cols): 
                    marker_np = markers[marker_size][id]
                    marker = Image.fromarray(cv2.cvtColor(marker_np, cv2.COLOR_GRAY2RGB))
                    center = grid_coords[id]
                    top_left = (round(center[0] - marker_size / 2), round(center[1] - marker_size / 2))
                    grid_image.paste(marker, top_left)
                    id += 1
            grid_img = cv2.cvtColor(np.array(grid_image), cv2.COLOR_RGB2BGR)

          

            # cv2.imshow('Grid Image', grid_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if invert_colors:
                grid_img = 255 - grid_img

            if save_grids:
                grid_img_path = os.path.join(grids_folder, f'{grid_idx}.png')
                cv2.imwrite(grid_img_path, grid_img)

            for _ in range(repeat_grid_factor):
                video_writer.write(grid_img)
            
            grid_idx += 1

    for value in synch_sequences['end']:
        intensity = 255 if value == 1 else 0
        image_array = np.full((projector_resolution[1], projector_resolution[0], 3), intensity, dtype=np.uint8)
        video_writer.write(image_array)

    # Add white image at the end of the video
    white_frame_count = int(video_fps * white_duration)
    for _ in range(white_frame_count):
        video_writer.write(white_frame)



    seq_info_path = os.path.join(video_folder, 'seq_info.json')
    save_seq_info_json(seq_info, seq_info_path)


    video_writer.release()

    print(f"Video generated successfully. Video saved at {video_path}")
    print(f"Sequence info saved at {seq_info_path}")



