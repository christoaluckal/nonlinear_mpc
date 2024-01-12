import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import math
import matplotlib.pyplot as plt
from scipy import signal

# Example cubic splines for x and y
# Replace these with your actual spline functions
t = np.linspace(0, 2 * np.pi, 100)
x_spline = CubicSpline(t, np.sin(t))
y_spline = CubicSpline(t, np.cos(t))

plt.plot(t, x_spline(t), label='x')
plt.plot(t, y_spline(t), label='y')
plt.show()

# Example point (replace with your current point)
current_x, current_y = -math.sqrt(2)/2, -math.sqrt(2)/2

# Function to calculate distance from a point to a point on the spline
def distance_to_spline(t):
    spline_x, spline_y = x_spline(t), y_spline(t)
    print("Spline x:", spline_x)
    print("Spline y:", spline_y)
    print("Current x:", current_x)
    print("Current y:", current_y)
    return np.sqrt((spline_x - current_x) ** 2 + (spline_y - current_y) ** 2)

# Minimize the distance to find the closest point on the spline
result = minimize_scalar(distance_to_spline, bounds=(0,5), method='bounded')

# The parameter value 't' at the closest point
closest_param = result.x

# Optional: Velocity spline - replace with your actual velocity spline
# velocity_spline = CubicSpline(t, velocities)
# velocity_at_closest = velocity_spline(closest_param)

print("Closest parameter:", closest_param)

print("X at closest point:", x_spline(closest_param))
print("Y at closest point:", y_spline(closest_param))

# print("Velocity at closest point:", velocity_at_closest)

a = 1
b = 4
t = np.linspace(0, 3, 500)

y = (b+a)/2 + ((b-a)/2) * signal.sawtooth(np.pi * 4 * t)

plt.plot(t, y)
plt.show()
