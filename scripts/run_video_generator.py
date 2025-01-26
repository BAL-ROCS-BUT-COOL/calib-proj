from calib_proj.video_generator.generate_video import generate_video


################# Parameters ###################
msm_base_size = 18 # [px] must be a multiple of the original size of the dictionary
msm_scales = [1,3,5,7]
n_shifts = 10 # number of shifts of the MSMs grid on x and y axis (total shifts = n_shifts^2)
grid_fps = 10 # number of MSMs grids projected per second
###################################################

n_grids = len(msm_scales) * n_shifts**2
projection_duration = n_grids / grid_fps
print(f"Projection duration: {projection_duration:.1f} s")

generate_video(msm_base_size=msm_base_size,
               msm_scales=msm_scales,
               n_shifts=n_shifts,
               projector_resolution=(1920, 1080), # [px]
               grid_size=(4, 8), # number of MSMs on x and y axis
               video_folder="video", 
               grid_fps=grid_fps) # folder where the video will be saved
# print("Video generated successfully.")