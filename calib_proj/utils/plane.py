import numpy as np
from typing import Tuple
# from calib_proj.utils.visualization_tools import plot_frame
# from calib_proj.utils.se3 import SE3
# import calib_proj.utils.se3 as se3

from calib_commons.utils.se3 import SE3
from calib_commons.viz.visualization_tools import plot_frame

# from calib_proj.utils import utils
class Plane: 

    def __init__(self, 
                 pose: SE3, 
                 id=None): 
        
        self.pose = pose
        self.id = id

    def get_pi(self): 
        return np.append(self.get_n(), self.get_d())
    
    def get_n(self): 
        return self.pose.mat[:3,2]
    
    def get_d(self): 
        return -self.get_n().dot(self.get_origin())

    def get_origin(self): 
        return self.pose.mat[:3,3]
       
    def plot(self): 
        name=r"$\{ \pi_" + str(self.id) + r"\}$"
        plot_frame(self.pose.mat, name=name, axis_length=0.2)

    @classmethod
    def from_normal_and_origin(cls, n, o, id=None):
        n = n.squeeze()

        v1 = np.array([1, 0, -n[0] / n[2]])
        v2 = np.cross(n, v1)

        # Normalize vectors to make them unit vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = n

        # print(v1, v2, v3)

        R = np.column_stack((v1,v2,v3))
        return cls(SE3.from_rt(R,o), id=id)
    
    @classmethod
    def from_normal_and_d(cls, n, d, id=None):
        n = n.squeeze()

        v1 = np.array([1, 0, -n[0] / n[2]])
        v2 = np.cross(n, v1)

        # Normalize vectors to make them unit vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = n

        # print(v1, v2, v3)

        z = float(-d/n[2])
        o = np.array([0,0,z])

        R = np.column_stack((v1,v2,v3))
        return cls(SE3.from_rt(R,o), id=id)
