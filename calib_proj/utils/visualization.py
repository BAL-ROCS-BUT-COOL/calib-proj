from typing import List

import matplotlib.pyplot as plt
from matplotlib import rcParams
import math 
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

from calib_commons.scene import SceneType
from calib_commons.types import idtype
from calib_commons.correspondences import Correspondences, get_conform_obs_of_cam
from calib_commons.viz.visualization_tools import get_color_from_error, get_color_from_id

import calib_commons.viz.visualization as generic_visualization
import calib_commons.viz.visualization_tools as visualization_tools
from calib_proj.core.projector_scene import ProjectorScene




def visualize_scenes(scene: ProjectorScene, show_ids = True, show_only_points_with_estimate = False,
                show_fig = True, 
                save_fig = False, 
                save_path = None, 
                dpi = 300, 
                elev=None, 
                azim=None) -> None: 
    generic_visualization.visualize_scenes([scene.generic_scene], show_ids, show_only_points_with_estimate, show_fig, save_fig, save_path, dpi, elev, azim)


# def visualize_scenes(scene: ProjectorScene, show_ids = True, show_only_points_with_estimate = False,
#                 show_fig = True, 
#                 save_fig = False, 
#                 save_path = None, 
#                 dpi = 300, 
#                 elev=None, 
#                 azim=None) -> None: 
#     fig = plt.figure(figsize=(6, 6)) 
#     rcParams['text.usetex'] = True
#     ax = fig.add_subplot(projection='3d')

#     if elev and azim:
#         ax.view_init(elev=elev, azim=azim)  # Change elev and azim as needed
    
#     points_ids = None 
#     if show_only_points_with_estimate:
#         for scene in scenes:
#             if scene.generic_scene.type == SceneType.ESTIMATE: 
#                 points_ids = list(scene.generic_scene.object_points.keys())


#     # xmin = np.inf
#     # ymin = np.inf
#     # zmin = np.inf

#     # xmax = -np.inf
#     x = []
#     y = []
#     z = []

#     for scene in scenes: 
#         generic_visualization.get_coords(scene.generic_scene, x,y,z, points_ids)

#     x_min = min(x)
#     x_max = max(x)

#     y_min = min(y)
#     y_max = max(y)

#     z_min = min(z)
#     z_max = max(z)

    
        
#     min_coords = min([x_min, y_min, z_min])
#     max_coords = max([x_max, y_max, z_max])

#     # print("x: ", x_min, x_max)
#     # print("y: ", y_min, y_max)
#     # print("z: ", z_min, z_max)
#     size_x = x_max-x_min
#     size_y = y_max-y_min
#     size_z = z_max-z_min

#     size = max([size_x, size_y, size_z])
#     # print("size ", size)
#     x_mid = (x_max+x_min)/2
#     y_mid = (y_max+y_min)/2
#     z_mid = (z_max+z_min)/2

#     xlim = [x_mid-size/2, x_mid+size/2]
#     ylim = [y_mid-size/2, y_mid+size/2]
#     zlim = [z_mid-size/2, z_mid+size/2]

#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.set_zlim(zlim)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')   
#     ax.set_box_aspect([1, 1, 1]) 
#     ax.set_title('3D Scenes', fontsize=14, pad=50)   




#     for scene in scenes: 
#         plot_scene(scene, xlim, ylim, zlim, show_ids, points_ids)

#         if save_fig:
#                 # Create the directory if it does not exist
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)
#                 fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

#     if show_fig:
#         plt.show(block=False)

    # plt.show(block=False)

def visualize_2d(scene: ProjectorScene = None, 
                observations = None, 
                show_only_points_with_both_obsv_and_repr=True,
                show_ids = False, 
                which = "both",
                show_fig = True, 
                save_fig = False, 
                save_path = None, 
                dpi = 300) -> None: 
    
    generic_visualization.visualize_2d(scene.generic_scene, observations, show_only_points_with_both_obsv_and_repr, show_ids, which, show_fig, save_fig, save_path, dpi)



        
def plot_reprojections_in_camera(correspondences, 
                             cameraId: idtype,
                             ax, 
                             cmap = None, 
                             reprojectionErrors = None, 
                             show_ids=True, 
                             points_ids=None) -> None:
    
    if not points_ids: 
        points_ids=correspondences[cameraId]

    if reprojectionErrors:
        errorsList = []
        for errorsOfCamera in reprojectionErrors.values():
            errorsList.extend(errorsOfCamera.values())

        if errorsList:
            errorMin = min(errorsList)
            errorMax = max(errorsList)
    marker = MARKERS_OBSERVATIONS[SceneType.ESTIMATE]

    for point_id in points_ids:
        observation = correspondences[cameraId][point_id]
   
        if cmap=="id":
            color = get_color_from_id(point_id)
        elif cmap=="reprojectionError": 
            error = reprojectionErrors[cameraId].get(point_id)
            if error:
                color = get_color_from_error(error, errorMin, errorMax)
            else:
                raise ValueError("no error found for this point.")
        else:
            raise ValueError("cmap not implemented.")
        alpha = 1
        # print(point_id)
        ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
        if show_ids:
            ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color='black', fontsize=10, alpha=1)

    if cmap=="reprojectionError": 
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=errorMin, vmax=errorMax))
        sm.set_array([])  # You can safely ignore this line.

        # Add the colorbar to the figure
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Reprojection Error')


        
    

def plot_observations_in_camera(correspondences, 
                             cameraId: idtype,
                             ax, 
                             cmap = None, 
                             reprojectionErrors = None, 
                             show_ids=True, 
                            points_ids=None) -> None:
    
    if not points_ids: 
        points_ids=correspondences[cameraId]

    for point_id in points_ids:
        observation = correspondences[cameraId][point_id]
        if observation and observation._is_conform:
        
            marker = MARKERS_OBSERVATIONS[SceneType.SYNTHETIC]
            alpha=0.5
            color = "black"
    
            ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
            if show_ids:
                ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color=color, fontsize=10, alpha=1)

  

def plot_scene(scene: ProjectorScene, xlim, ylim, zlim, show_ids = True, points_ids = None) -> None: 
    ax = plt.gca() 

    generic_visualization.plot_scene(scene.generic_scene, xlim, ylim, zlim, show_ids, points_ids)

    if scene.projector:
        scene.projector.plot("proj")

    if scene.plane: 
        scene.plane.plot()
        # print(scene.plane.get_origin())
        X = [xlim[0], xlim[0], xlim[1], xlim[1]]
        Y = [ylim[0], ylim[1], ylim[1], ylim[0]]
        Z = [- 1 / scene.plane.get_n()[-1] * scene.plane.get_pi().dot(np.array([X[i], Y[i], 0, 1])) for i in range(len(X))]
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        points = np.column_stack((X,Y,Z))
        vertices = np.zeros((4,3))
        vertices[:4,:] = points
        vertices = [vertices.tolist()]
        # plane_collection.set_verts(vertices, closed=True)
        plane_collection = Poly3DCollection(vertices, facecolors='cyan',
                                            linewidths=1, edgecolors='black', alpha=0.5)
        ax.add_collection3d(plane_collection)

def plot_reprojection_errors(scene_estimate: ProjectorScene, 
                             observations: Correspondences, 
                             show_fig = True,
                            save_fig = False, 
                            save_path = None, 
                            dpi = 300) -> None: 
    generic_visualization.plot_reprojection_errors(scene_estimate.generic_scene, observations, show_fig, save_fig, save_path, dpi)
    # n = len(scene_estimate.cameras)
    # cols = math.ceil(math.sqrt(n))
    # rows = math.ceil(n/ cols)

    # fig_width = cols * 4  # Each subplot has a width of 4 inches (adjust as needed)
    # fig_height = rows * 4  # Each subplot has a height of 3 inches (adjust as needed)

    # # Create a figure with dynamic size
    # fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # # fig, axs = plt.subplots(rows, cols)
    # axs = np.array(axs).reshape(-1) if n > 1 else [axs]
    # axsUsed = [False for ax in axs]

    # axIndex = 0
    
    # all_errors = []
    # all_errors_conform = []

    # for camera in scene_estimate.cameras.values(): 
    #     ax = axs[axIndex]
    #     axsUsed[axIndex] = True
    #     axIndex += 1
        
    #     errors_conform = get_errors_x_y_in_cam(camera, scene_estimate.object_points, observations, which="conform")
    #     errors_nonconform = get_errors_x_y_in_cam(camera, scene_estimate.object_points, observations, which="non-conform")
    #     # print(errors_conform)
    #     # print(errors_nonconform)
    #     errors_conform = list(errors_conform.values())
    #     errors_nonconform = list(errors_nonconform.values())
    #     errors_tot = np.concatenate(errors_conform+errors_nonconform, axis=0)        # plt.xlim(-plot_lim, plot_lim)

        
    #     all_errors.extend(errors_tot)
    #     all_errors_conform.extend(errors_conform)
        
    #     # error_min = np.nanmin(errors_tot)
    #     # error_max = np.nanmax(errors_tot)

    #     # plot_lim = max(abs(error_min), abs(error_max))

    #     # # fig = plt.figure()
    #     # # plt.xlim(-plot_lim, plot_lim)
    #     # # plt.ylim(-plot_lim, plot_lim)
    #     # ax.set_xlim([-plot_lim, plot_lim])
    #     # ax.set_ylim([-plot_lim, plot_lim])
    #     # ax = plt.gca()
    #     aspect_ratio = 1
    #     ax.set_aspect(aspect_ratio / ax.get_data_ratio())


    #     for errors_ in errors_conform: 
    #         ax.scatter(errors_[:,0], errors_[:,1], s=0.5, c='blue', marker = 'o', alpha=1)  

    #     for errors_ in errors_nonconform: 
    #         ax.scatter(errors_[:,0], errors_[:,1], s=0.5, c='red', marker = 'o', alpha=1)  
      
    #     circle = plt.Circle((0,0), 1, edgecolor='red', facecolor='none', linewidth=1.5, alpha = 1)
    #     ax.add_patch(circle)

    #     circle = plt.Circle((0,0), 0.7, edgecolor='red', facecolor='none', linewidth=1.5, alpha = 1)
    #     ax.add_patch(circle)

    #     title = f"Repr. errors in {camera.id}"
    #     ax.set_title(title)
    #     # plt.title(title)
    
    # # compute axis limits (same for all subplots)
    # # error_min = np.nanmin(all_errors)
    # # error_max = np.nanmax(all_errors)
    # # plot_lim = max(abs(error_min), abs(error_max))
    # error_min = np.nanmin(all_errors_conform)
    # error_max = np.nanmax(all_errors_conform)
    # # plot_lim = max(abs(error_min), abs(error_max))
    # plot_lim = 2
    # axIndex = 0
    # for camera in scene_estimate.cameras.values(): 
    #     ax = axs[axIndex]
    #     axIndex += 1
    #     ax.set_xlim([-plot_lim, plot_lim])
    #     ax.set_ylim([-plot_lim, plot_lim])

    #     if save_fig:
    #         # Create the directory if it does not exist
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    #     if show_fig:
    #         plt.show(block=False)
            



    return 