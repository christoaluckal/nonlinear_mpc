#!/usr/bin/env python3

'''
Topic: Kinematic Nonlinear Model Predictive Controller for F1tenth simulator
Author: Rongyao Wang
Instiution: Clemson University Mechanical Engineering
'''

import rospy
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped, PoseArray, Pose,Point
from visualization_msgs.msg import Marker
import math
from utils import yaw_change_correction, vehicle_coordinate_transformation
import numpy as np
from numpy import linalg as LA
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import casadi
from timeit import default_timer as timer
import csv
import os
import pathgen
import sys
import matplotlib.pyplot as plt
from std_srvs.srv import SetBool
import rosparam
import rospkg
rp = rospkg.RosPack()

global_speed = rosparam.get_param("global_speed")
N = rosparam.get_param("N")
future_time = rosparam.get_param("future_time")
WB = rosparam.get_param("WB")
vel_max = rosparam.get_param("vel_max")
vel_min = rosparam.get_param("vel_min")
max_steer = rosparam.get_param("max_steer")
max_acc = rosparam.get_param("max_acc")

track_file = os.path.join(rp.get_path('nonlinear_mpc'),rosparam.get_param("track_file"))
track_length = pathgen.get_track_length(track_file)
qx = rosparam.get_param("Q")[0]
qy = rosparam.get_param("Q")[1]
q_yaw = rosparam.get_param("Q")[2]
q_vel = rosparam.get_param("Q")[3]
r_acc = rosparam.get_param("R")[0]
r_steer = rosparam.get_param("R")[1]
u_acc = rosparam.get_param("U")[0]
u_steer = rosparam.get_param("U")[1]

N_x = 4
N_u = 2

dt = future_time/N
time = int(track_length/global_speed)

print("Track Time:",time)
print("Track Length:",track_length)

class VehicleState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vel = 0.0
        self.x_vel = 0
        self.y_vel = 0
        self.odom = []
        self.odom_sub = rospy.Subscriber(rosparam.get_param('odom_topic'), Odometry, self.odom_callback)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vel = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        quaternion = (qx,qy,qz,qw)
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.x = x
        self.y = y
        self.vel = vel
        self.x_vel = msg.twist.twist.linear.x
        self.y_vel = msg.twist.twist.linear.y
        self.yaw = yaw
        # self.odom.append([self.x,self.y,self.x_vel,self.y_vel,self.yaw])
        self.odom.append([self.x,self.y,self.x_vel,self.y_vel])

    def vehicle_state_output(self):
        vehicle_state = np.array([self.x, self.y, self.yaw, self.vel,self.x_vel,self.y_vel])
        return vehicle_state

def nonlinear_kinematic_mpc_solver(x_ref, x_0, N):
    opti = casadi.Opti()
    
    print(x_ref.shape)
    print(x_0.shape)
    
    print(x_ref)
    print(x_0)

    x = opti.variable(N_x, N+1)
    u = opti.variable(N_u, N)
    x_ref = casadi.MX(x_ref)
    x_0 = casadi.MX(x_0)

    Q = casadi.diag([qx, qy, q_yaw, q_vel])
    R = casadi.diag([r_acc, r_steer])
    cost = 0

    for t in range(N-1):
        cost += u[:, t].T @ R @ u[:, t]
        if t != 0:
            cost += (x_ref[:, t].T - x[:, t].T) @ Q @ (x_ref[:, t] - x[:, t])
        opti.subject_to(x[0, t + 1] == x[0, t] + x[3, t]*casadi.cos(x[2, t])*dt)
        opti.subject_to(x[1, t + 1] == x[1, t] + x[3, t]*casadi.sin(x[2, t])*dt)
        opti.subject_to(x[2, t + 1] == x[2, t] + x[3, t]*casadi.tan(u[1, t])*dt/WB)
        opti.subject_to(x[3, t + 1] == x[3, t] + u[0, t]*dt)

        if t < N-2:
            opti.subject_to(u[1, t+1] - u[1, t] >= -u_steer)
            opti.subject_to(u[1, t+1] - u[1, t] <= u_steer)
            opti.subject_to(u[0, t+1] - u[0, t] <= u_acc)
            opti.subject_to(u[0, t+1] - u[0, t] >= -u_acc)

        # if t < N-2:
        #     opti.subject_to(u[1, t+1] - u[1, t] >= -0.06)
        #     opti.subject_to(u[1, t+1] - u[1, t] <= 0.06)
        #     opti.subject_to(u[0, t+1] - u[0, t] <= 0.1)
        #     opti.subject_to(u[0, t+1] - u[0, t] >= -0.1)

    opti.subject_to(x[:, 0] == x_0)
    opti.subject_to(u[1, :] <= max_steer)
    opti.subject_to(u[1, :] >= -max_steer)
    opti.subject_to(u[0, :] <= max_acc)
    opti.subject_to(u[0, :] >= -max_acc)

    opti.minimize(cost)
    opti.solver('ipopt',{"print_time": False}, {"print_level": 0})#, {"acceptable_tol": 0.0001}
    sol = opti.solve()

    acceleration = sol.value(u[0,0])
    steering = sol.value(u[1,0])

    return acceleration, steering



def rviz_markers(pose,idx):
    if idx == 0:
        points = Marker()
        points.type = Marker.POINTS
        points.header.frame_id = "map"
        points.ns = "raceline"
        points.action = Marker.ADD
        points.pose.orientation.w = 1
        points.scale.x = 0.5
        points.scale.y = 0.5
        points.color.r = 0.5
        points.color.g = 0.5
        points.color.a = 1

        for i in pose:
            p = Point()
            p.x = i[0]
            p.y = i[1]
            points.points.append(p)

    elif idx == 1:
        points = Marker()
        points.type = Marker.POINTS
        points.header.frame_id = "map"
        points.ns = "spline"
        points.action = Marker.ADD
        points.pose.orientation.w = 1
        points.scale.x = 1
        points.scale.y = 1
        points.color.b = 1
        points.color.a = 1

    for i in pose:
        p = Point()
        p.x = i[0]
        p.y = i[1]
        points.points.append(p)

    return points


def reference_pose_selection(x_spline,y_spline, curr_t,N):
    delta_t = future_time
    # while curr_t+delta_t > time:
    #     delta_t-=dt
    # if delta_t<0.01:
    #     curr_t = 0
    #     delta_t = future_time
    
    t_vec = np.linspace(curr_t,curr_t+delta_t,N)
    if curr_t+delta_t > time:
        t_vec = t_vec%time
    xTraj = x_spline(t_vec)
    yTraj = y_spline(t_vec)

    xdTraj = x_spline(t_vec,1)
    ydTraj = y_spline(t_vec, 1)

    thTraj = np.arctan2(ydTraj,xdTraj)
    
    vTraj = np.sqrt(np.power(xdTraj,2)+ np.power(ydTraj,2))
    path_array = np.array([xTraj,yTraj,thTraj,vTraj]).T

    return path_array


if __name__ == "__main__":
    rospy.init_node("nmpc_node",anonymous=True)
    rospy.loginfo("Start NMPC")
    drive_pub = rospy.Publisher(rosparam.get_param("drive_topic"), AckermannDrive, queue_size=1)
    raceline_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)
    spline_marker_pub = rospy.Publisher('visualization_markers',Marker,queue_size=1)

    csv_f = track_file

    global_path,x_spline,y_spline = pathgen.get_spline_path(csv_f,time)

    rate = rospy.Rate(rosparam.get_param("rate"))
    vehicle_state = VehicleState()
    # N = 5

    ref_list = np.array(global_path[:,0:2])

    init_time = rospy.get_time()
    delta_t = 0
    speed_ref = []
    odoms = []
    
    while not rospy.is_shutdown():
        try:
            curr_t = rospy.get_time() - init_time
            delta_t = rospy.get_time() - curr_t
            current_state = vehicle_state.vehicle_state_output()
            # rospy.loginfo(curr_t)
            if curr_t > time + 1:
                drive_msg = AckermannDrive()
                drive_pub.publish(drive_msg)
                break
            # Transform the reference pose to vehicle coordinate
            # if curr_t > future_time:
            # if True:
            reference_pose = reference_pose_selection(x_spline,y_spline, curr_t, N)
            
            x_ref = np.zeros((N, 4))
            for i in range(N):
                x, ref = vehicle_coordinate_transformation(reference_pose[i,:], current_state)
                x_ref[i, :] = ref
            
            # Compute Control Output from Nonlinear Model Predictive Control
            acceleration, steering = nonlinear_kinematic_mpc_solver(x_ref.T, x.T, N)
            
            # speed = np.clip(current_state[3] + acceleration*dt, vel_min, vel_max)
            speed = current_state[3]+acceleration*dt
            # print(current_state[3],speed)
            drive_msg = AckermannDrive()
            drive_msg.speed = speed
            drive_msg.steering_angle = steering
            drive_pub.publish(drive_msg)



            raceline_pub.publish(rviz_markers(global_path,0))
            spline_marker_pub.publish(rviz_markers(reference_pose,1))
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue
        except rospy.exceptions.ROSInterruptException:
            break
    