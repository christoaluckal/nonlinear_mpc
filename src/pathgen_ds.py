from scipy.interpolate import interp1d, CubicSpline
import numpy as np
import math

def get_track_length(csv_f):
    path = csv_f

    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[1:,0]
    yCoords = waypoints[1:,1]

    correction_angle = -math.atan2(yCoords[1]-yCoords[0],xCoords[1]- xCoords[0])
    R_z = np.array(
                    [[math.cos(correction_angle), -math.sin(correction_angle)],
                    [math.sin(correction_angle), math.cos(correction_angle)]])
    coords = zip(xCoords, yCoords)
    corrected_xCoords = []
    corrected_yCoords = []

    for p in coords:
        p = np.matmul(R_z,np.array(p).T)
        corrected_xCoords.append(p[0])
        corrected_yCoords.append(p[1])

    xCoords = corrected_xCoords
    yCoords = corrected_yCoords

    track_distance = 0
    for i in range(1,len(xCoords)-1):
        track_distance += math.sqrt((xCoords[i+1]-xCoords[i])**2+(yCoords[i+1]-yCoords[i])**2)

    return track_distance

def get_spline_path(csv_f,time):
    path = csv_f

    waypoints = np.genfromtxt(path, dtype=float, delimiter=",")
    xCoords = waypoints[1:,0]
    yCoords = waypoints[1:,1]

    correction_angle = -math.atan2(yCoords[1]-yCoords[0],xCoords[1]- xCoords[0])
    R_z = np.array(
                    [[math.cos(correction_angle), -math.sin(correction_angle)],
                    [math.sin(correction_angle), math.cos(correction_angle)]])
    coords = zip(xCoords, yCoords)
    corrected_xCoords = []
    corrected_yCoords = []

    for p in coords:
        p = np.matmul(R_z,np.array(p).T)
        corrected_xCoords.append(p[0])
        corrected_yCoords.append(p[1])

    xCoords = corrected_xCoords
    yCoords = corrected_yCoords

    svec = np.linspace(0,get_track_length(path),len(xCoords))
    xTrajCS = CubicSpline(svec,xCoords)
    yTrajCS = CubicSpline(svec,yCoords)

    xTraj = xTrajCS(svec)
    yTraj = yTrajCS(svec)

    xdTraj = xTrajCS(svec,1)
    ydTraj = yTrajCS(svec, 1)

    xddTraj = xTrajCS(svec,2)
    yddTraj = yTrajCS(svec, 2)

    thTraj = np.arctan2(ydTraj,xdTraj)
    thdTraj = np.divide(np.multiply(yddTraj, xdTraj)-np.multiply(ydTraj, xddTraj), (np.power(xdTraj,2)+np.power(ydTraj,2)))

    vTraj = np.sqrt(np.power(xdTraj,2)+ np.power(ydTraj,2))

    path_array = np.array([xTraj,yTraj,thTraj,vTraj]).T

    return (path_array,xTrajCS,yTrajCS)

