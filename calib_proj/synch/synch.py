import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
import json
import cv2
import os
from tqdm import tqdm

from calib_proj.video_generator.generate_video import load_seq_info_json
from calib_proj.synch.decodage import detect_sync_sequence, process_video_to_gray_mean
from calib_proj.synch.utils import plot_sequence
from calib_proj.synch.extract_frames import extract_frames

def synch_and_extract(video_folder, sequence_info_path, threshold=0.6):
    start_end_frames = synch(video_folder, sequence_info_path, threshold)

    videos_path = {}
    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            videos_path[video_file.split('.')[0]] = video_path


    for cam_name in list(start_end_frames.keys()):
        for seq_name in list(start_end_frames[cam_name].keys()):
            if start_end_frames[cam_name][seq_name] is None:
                del videos_path[cam_name]


    seq_info = load_seq_info_json(sequence_info_path)

    out_folder_path = Path(video_folder) / "frames"

    extract_frames(videos_path, sequence_info_path, start_end_frames, out_folder_path)

def synch(
    video_folder: Path,
    sequence_info_path: Path,
    threshold: float,
    start_time: float = 0.0,
    end_time: float = None
):
    print(f"Synchronizing cameras with sequence started...")
    seq_info = load_seq_info_json(sequence_info_path)
    seq_fps = seq_info['video_fps']
    synch_sequences = seq_info['synch_sequences']

    videos_path = {}
    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_path = os.path.join(video_folder, video_file)
            videos_path[video_file.split('.')[0]] = video_path

    cameras_fps = {cam_name: cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) for cam_name, video_path in videos_path.items()}
    # print(f"Cameras FPS: {cameras_fps}")

    
    start_end_frames = {}

    # start_end_frames_path = Path(r"video/start_end_frames.json")

    # if 1:
    for cam_name, video_path in tqdm(videos_path.items(), desc="Synchronization"):
        print(f" Processing {cam_name}...")
        start_end_frames[cam_name] = {}
        received_signal = process_video_to_gray_mean(video_path, start_time, end_time)
        # plot_sequence(received_signal)
        for seq_name, synch_sequence in synch_sequences.items():
            # print(f"Detecting sequence {seq_name}...")
            detected_index, mes_length = detect_sync_sequence(received_signal, synch_sequence, seq_fps, cameras_fps[cam_name], threshold, plot=0)
            # plt.show()
            # print(f"Sequence {seq_name} detected at index {detected_index}.")
            if detected_index is not None:
                if seq_name == 'start':
                    start_end_frames[cam_name][seq_name] = detected_index + mes_length + start_time * cameras_fps[cam_name]
                    # print(f"Sequence {seq_name} start at index {detected_index + len(synch_sequence)}.")
                    print(f"Sequence start signal detected.")
                elif seq_name == 'end':
                    start_end_frames[cam_name][seq_name] = detected_index + start_time * cameras_fps[cam_name]
                    print(f"Sequence end signal detected.")

                    # print(f"Sequence {seq_name} end at index {detected_index}.")
                # print(f"")
            else: 
                print(f"Sequence {seq_name} signal not detected.")
                start_end_frames[cam_name][seq_name] = None




    #     # Save start and end frames to JSON
    #     with open(start_end_frames_path, 'w') as f:
    #         json.dump(start_end_frames, f, indent=4)

    # # Load start and end frames from JSON if it exists
    # if start_end_frames_path.exists():
    #     with open(start_end_frames_path, 'r') as f:
    #         start_end_frames = json.load(f)
    # else:
    #     start_end_frames = {}
    # print("Start and end frames:")
    # print(start_end_frames)

    # Remove entries with None values
    cam_names_with_missing_frames = [
        cam_name for cam_name, frames in start_end_frames.items()
        if frames.get('start') is None or frames.get('end') is None
    ]
    if len(cam_names_with_missing_frames) == 0:
        print(f"All cameras synchronized successfully.")
    else:
        print(f"Some cameras could not be synchronized: {cam_names_with_missing_frames}")

    return start_end_frames