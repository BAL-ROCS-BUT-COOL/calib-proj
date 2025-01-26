import cv2
import cv2.aruco as aruco
import os
from calib_proj.video_generator.marker_generator.utils import closest_multiple, is_multiple
from calib_proj.preprocessing.marker_systems import ARUCO_DICTIONARIES

from pathlib import Path

MARKER_ORIGINAL_SIZE = {'4X4_50': 6,
                        '4X4_100': 6,
                        '5X5_50': 8, 
                        '6X6_50': 10,
                        '7X7_50': 9}

                        
def generate_aruco_markers(sizes: list, 
                           ids, 
                           dictionary = '4X4_50', 
                           marker_folder = None, 
                           save = False):
    
    if save:
        if not os.path.exists(marker_folder):
            os.makedirs(marker_folder)
    else: 
        markers = {}

    for size in sizes:
        markers[size] = {}
        if not is_multiple(size, MARKER_ORIGINAL_SIZE[dictionary]):
            raise ValueError(f"Marker size {size} px is not a multiple of the original marker size = {MARKER_ORIGINAL_SIZE[dictionary]} px of the dictionary {dictionary}.")

        if save:
            size_folder = marker_folder + "\\" + str(size)
            if not os.path.exists(size_folder):
                os.makedirs(size_folder)

        for id in ids:
            marker = aruco.generateImageMarker(dictionary=ARUCO_DICTIONARIES[dictionary], 
                                id=id, 
                                sidePixels=size, 
                                borderBits=1)
            if save:
                name = size_folder + "\\" + str(id) + ".png"
                cv2.imwrite(name, marker)
            else: 
                markers[size][id] = marker

    if not save:
        return markers
    
if "__main__" == __name__:

    sizes = [6, 12, 24, 48, 96]
    ids = [i for i in range(32)]
    marker_folder = "video/markers/aruco"
    generate_aruco_markers(sizes, ids, save=False)
    print("Markers generated successfully.")