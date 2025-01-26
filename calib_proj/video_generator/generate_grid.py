import cv2
import numpy as np
from PIL import Image

def generate_grid(grid_size = (1,1),  # (rows, cols)
                  size_in_out_image: int = 100, 
                  margin_min: int = 50,
                  out_image_res = (1920, 1080),
                  shift = (0,0), 
                  from_folder = False, 
                  markers = None,
                  base_folder_path: str = None):  
    
    s_w = out_image_res[0] / (grid_size[1]+1)
    s_h = out_image_res[1] / (grid_size[0]+1)
    s = int(min(s_h, s_w))

    margin = (s-size_in_out_image) // 2
    # print(margin)
    if margin < margin_min: 
        raise ValueError(f"The margin ={margin} is too small for the given size_in_out_image.")
    
    # s = size_in_out_image + 2 * margin
    grid_image = Image.new('RGB', (out_image_res[0], out_image_res[1]), (255, 255, 255))

    rows, cols = grid_size
    delta_x = (out_image_res[0] - cols * s) / 2
    delta_y = (out_image_res[1] - rows * s) / 2

    id = 0
    points = {}
    box_image = np.ones((1080, 1920, 3), dtype=np.uint8)*255


    for row in range(rows): 
        for col in range(cols): 

            if from_folder:
                marker_path = base_folder_path + "\\" + str(id) + ".png"
                marker = Image.open(marker_path)
            else: 
                marker_np = markers[size_in_out_image][id]
                marker = Image.fromarray(cv2.cvtColor(marker_np, cv2.COLOR_GRAY2RGB))


            
            box_top_left_corner_x = int(delta_x + col * s + shift[0])
            box_top_left_corner_y = int(delta_y + row * s + shift[1])
            marker_top_left_corner_x = box_top_left_corner_x + (s-size_in_out_image) // 2
            marker_top_left_corner_y = box_top_left_corner_y + (s-size_in_out_image) // 2

           
            grid_image.paste(marker, (marker_top_left_corner_x, marker_top_left_corner_y))
          
            cv2.rectangle(box_image, (box_top_left_corner_x, box_top_left_corner_y), (box_top_left_corner_x + s, box_top_left_corner_y + s), (0, 0, 255), 2)

            points[id] = np.array([[marker_top_left_corner_x + size_in_out_image / 2, marker_top_left_corner_y + size_in_out_image / 2]])

            id += 1
     
   
    return grid_image, points


     
