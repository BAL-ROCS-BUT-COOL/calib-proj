from typing import List
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from calib_commons.types import idtype
from calib_commons.correspondences import Correspondences

import calib_commons.viz.visualization as generic_visualization
from calib_proj.core.projector_scene import ProjectorScene




def visualize_scenes(scene: ProjectorScene, show_ids = True, show_only_points_with_estimate = False,
                show_fig = True, 
                save_fig = False, 
                save_path = None, 
                dpi = 300, 
                elev=None, 
                azim=None) -> None: 
    generic_visualization.visualize_scenes(
        [scene.generic_scene], 
        show_ids, 
        show_only_points_with_estimate, 
        show_fig, 
        save_fig, 
        save_path, 
        dpi, 
        elev, 
        azim
    )


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



        
# def plot_reprojections_in_camera(correspondences, 
#                              cameraId: idtype,
#                              ax, 
#                              cmap = None, 
#                              reprojectionErrors = None, 
#                              show_ids=True, 
#                              points_ids=None) -> None:
    
#     if not points_ids: 
#         points_ids=correspondences[cameraId]

#     if reprojectionErrors:
#         errorsList = []
#         for errorsOfCamera in reprojectionErrors.values():
#             errorsList.extend(errorsOfCamera.values())

#         if errorsList:
#             errorMin = min(errorsList)
#             errorMax = max(errorsList)
#     marker = MARKERS_OBSERVATIONS[SceneType.ESTIMATE]

#     for point_id in points_ids:
#         observation = correspondences[cameraId][point_id]
   
#         if cmap=="id":
#             color = get_color_from_id(point_id)
#         elif cmap=="reprojectionError": 
#             error = reprojectionErrors[cameraId].get(point_id)
#             if error:
#                 color = get_color_from_error(error, errorMin, errorMax)
#             else:
#                 raise ValueError("no error found for this point.")
#         else:
#             raise ValueError("cmap not implemented.")
#         alpha = 1
#         # print(point_id)
#         ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
#         if show_ids:
#             ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color='black', fontsize=10, alpha=1)

#     if cmap=="reprojectionError": 
#         sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('coolwarm'), norm=plt.Normalize(vmin=errorMin, vmax=errorMax))
#         sm.set_array([])  # You can safely ignore this line.

#         # Add the colorbar to the figure
#         cbar = plt.colorbar(sm, ax=ax)
#         cbar.set_label('Reprojection Error')


        
    

# def plot_observations_in_camera(correspondences, 
#                              cameraId: idtype,
#                              ax, 
#                              cmap = None, 
#                              reprojectionErrors = None, 
#                              show_ids=True, 
#                             points_ids=None) -> None:
    
#     if not points_ids: 
#         points_ids=correspondences[cameraId]

#     for point_id in points_ids:
#         observation = correspondences[cameraId][point_id]
#         if observation and observation._is_conform:
        
#             marker = MARKERS_OBSERVATIONS[SceneType.SYNTHETIC]
#             alpha=0.5
#             color = "black"
    
#             ax.plot(observation._2d[:,0], observation._2d[:,1], marker, markersize=4, markeredgewidth=1, color=color, alpha=alpha)  
#             if show_ids:
#                 ax.text(observation._2d[0,0], observation._2d[0,1], point_id, color=color, fontsize=10, alpha=1)

  


def plot_reprojection_errors(scene_estimate: ProjectorScene, 
                             observations: Correspondences, 
                             show_fig = True,
                            save_fig = False, 
                            save_path = None, 
                            dpi = 300) -> None: 
    generic_visualization.plot_reprojection_errors(scene_estimate.generic_scene, observations, show_fig, save_fig, save_path, dpi)
  