from calib_proj.video_generator.generate_video import generate_video
from calib_proj.video_generator.generate_grid import generate_grid

################# Parameters ###################
msm_base_size = 18 # [px] must be a multiple of the original size of the dictionary (6 px)
msm_scales = [1,3,5,7]  # scales of the MSMs
n_grids = 100 # number of MSMs grids
grid_fps = 10 # number of MSMs grids projected per second (recommended to be <= 3 * min(cameras fps) to avoid temporal synchronization problems)

projector_resolution = (1920, 1080) # [px]
grid_size = (4, 8) # number of MSMs on x and y axis. (4, 8) is optimized for the aspect ratio of a full HD (1920x1080) projector.

invert_colors = True # invert colors of MSMs grids        
###################################################

n_grids_tot = len(msm_scales) * n_grids
projection_duration = n_grids_tot / grid_fps
print(f"Projection duration: {projection_duration:.1f} s")


generate_video(msm_base_size=msm_base_size,
               msm_scales=msm_scales,
               n_grids=n_grids,
               projector_resolution=projector_resolution, 
               grid_size=grid_size,
               grid_fps=grid_fps, 
               invert_colors=invert_colors, 
               save_grids=False) 
