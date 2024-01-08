#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped, PoseArray, Pose,Point
from visualization_msgs.msg import Marker
import math
from utils import vehicle_coordinate_transformation
import numpy as np
from numpy import linalg as LA
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import casadi
from timeit import default_timer as timer
import feasible_pathgen as path_gen
import sys
import matplotlib.pyplot as plt
from std_srvs.srv import SetBool
import rosparam
import rospkg
import os
rp = rospkg.RosPack()


def MPC_Class():
    def __init__(self,):
        self.N = rosparam.get_param("N")
        self.future_time = rosparam.get_param("future_time")
        self.WB = rosparam.get_param("WB")
        self.vel_max = rosparam.get_param("vel_max")
        self.vel_min = rosparam.get_param("vel_min")
        self.max_steer = rosparam.get_param("max_steer")
        self.max_acc = rosparam.get_param("max_acc")
        # self.track_file = os.path.join(rp.get_path('nonlinear_mpc'),rosparam.get_param("track_file"))
        # self.track_length = path_gen.get_track_length(track_file)
        self.qx = rosparam.get_param("Q")[0]
        self.qy = rosparam.get_param("Q")[1]
        self.q_yaw = rosparam.get_param("Q")[2]
        self.q_vel = rosparam.get_param("Q")[3]
        self.r_acc = rosparam.get_param("R")[0]
        self.r_steer = rosparam.get_param("R")[1]
        self.u_acc = rosparam.get_param("U")[0]
        self.u_steer = rosparam.get_param("U")[1]
        self.v_max = rosparam.get_param("v_max")
        
        self.N_x = 4
        self.N_u = 2
        
        self.dt = self.future_time/self.N