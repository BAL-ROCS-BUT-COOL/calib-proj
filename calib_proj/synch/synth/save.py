import cv2
import os
from pathlib import Path

def save_video_frames(video_path, output_folder):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    # Loop through video frames
    while True:
        ret, frame = video_capture.read()

        # If there are no more frames, exit the loop
        if not ret:
            break

        # Construct the filename and save the frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)

        print(f"Saved {frame_filename}")
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"All frames saved in {output_folder}.")

# Example usage

video_path = Path(r"C:\Users\timfl\Documents\test_calibProj\video\iphone.MOV")
out_folder = Path(r"C:\Users\timfl\Documents\test_calibProj\video\allframes")
save_video_frames(video_path, out_folder)
