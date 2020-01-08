import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.special import erf
from scipy.stats import gaussian_kde
import math
import random
from numpy import linalg as LA

# Evaluates a folded Gaussian centered at 0 with variance sigma at the point theta
def eval_folded(sigma, theta, cutoff=10000):
    total = 0
    for n in range(-cutoff, cutoff):
        total += math.exp(-((theta - 2 * np.pi * n)**2) / (2 * sigma * sigma))
    return 1/(math.sqrt(2 * np.pi) * sigma) * total

# Evaluates a folded Gaussian centered at 0 with variance sigma at the points in the theta_arr
def eval_folded_grid(sigma, theta_arr, cutoff=10000):
    total = np.zeros(len(theta_arr))
    for n in range(-cutoff, cutoff):
        total += np.exp(-((theta_arr - 2 * np.pi * n)**2) / (2 * sigma * sigma))
    return 1/(math.sqrt(2 * np.pi) * sigma) * total

# Debug this later
def get_J(points):
    points += [points[0]]
    area = 0
    for ii in range(len(points) - 1):
        area += (points[ii][0] * points[ii + 1][1] - points[ii][1] * points[ii + 1][0])
    area = area/2
    J = area * 2 # Jacobian is twice the area of the polygon
    return J


# Find closure solutions
# Crazy formula for the completion of an isosceles triangle between (a, b), and (-L, 0)
# Formula from Wolfram Alpha, obtained by solving (x + L)^2 + y^2 = L^2 and (x - a)^2 + (y - b)^2 = L^2
# Note that there are typically two or zero answers here (depending on if (a, b) is close enough to (-L, 0))
# This is vectorized; a and b can be np.array's

def get_triangles(a, b, L):
    a = np.array([a]) # Handle the case if a is a single number
    b = np.array([b]) # Handle the case if b is a single number
    a = a.flatten()
    b = b.flatten()
    
    sum1 = a**4 + 4 * (a**3) * L + 2 * (a**2) * (b**2) + 2 * (a**2) * (L**2) + 4 * a * (b**2) * L \
        - 4 * a * (L**3) + b**4 - 2 * (b**2) * (L**2) - 3 * (L**4)
    sum1 = np.sqrt(-(b**2 * sum1))
    x1 = (a**3 + (a**2) * L - sum1 + a * (b**2) - a * (L**2) - (b**2) * L - L**3)/(2*((a+L)**2 + b**2))
    x2 = (a**3 + (a**2) * L + sum1 + a * (b**2) - a * (L**2) - (b**2) * L - L**3)/(2*((a+L)**2 + b**2))
    y1 = ((a * b)**2 + (a + L) * sum1 + 2 * a * (b**2) * L + b**4 + (b**2) * (L**2))/(2 * b * ((a+L)**2 + b**2))
    y2 = ((a * b)**2 - (a + L) * sum1 + 2 * a * (b**2) * L + b**4 + (b**2) * (L**2))/(2 * b * ((a+L)**2 + b**2))
    
    no_triangle_idxs = (a + L)**2 + b**2 > (2*L)**2
    x1[no_triangle_idxs] = np.nan # Cases where isosceles triangle cannot form
    x2[no_triangle_idxs] = np.nan
    y1[no_triangle_idxs] = np.nan
    y2[no_triangle_idxs] = np.nan
    return [(x1, y1), (x2, y2)]

# points can be (x_np_array, y_np_array) for vectorization 
# Gets angle between point1, point2, and point3. 
# Note, to get the angle of turning from the vector point1-point2 to the vector point2-point3, 
# you may have to do np.pi minus this value.
def get_angle(point1, point2, point3):
    vec1_x = point2[0] - point1[0]
    vec1_y = point2[1] - point1[1]
    vec2_x = point2[0] - point3[0]
    vec2_y = point2[1] - point3[1]

    vec1_norm = np.sqrt(vec1_x**2 + vec1_y**2)
    vec2_norm = np.sqrt(vec2_x**2 + vec2_y**2)
    vec_dot = vec1_x * vec2_x + vec1_y * vec2_y
    cos_val = np.clip(vec_dot/(vec1_norm * vec2_norm), -1, 1)
    return np.arccos(cos_val)

#orig in MM section.
def get_coords_from_thetas(thetas, L):
    #theta: input, array_like. Can be single list of thetas or array.
    #given list of angles, get final x, y coordinates.
    
    #first ind is link location, second ind is samples
#     if isinstance(thetas,list):
#         print('list')
#         if len(thetas[0])==1:
#             thetas = np.array(thetas).reshape(-1,1)
#         else:
#             thetas = np.array(thetas)

#     print(thetas.shape)
#     #first ind is link location, second is xy, third is samples
#     points = np.zeros([thetas.shape[0]+1, 2, thetas.shape[1]])
#     print(points.shape)
    
#     for i in range(1,thetas.shape[0]+1):
#         print(i, 'thetas_i', thetas[:i])
#         points[i,0,:] = np.sum([L * np.cos(subset) for subset in np.cumsum(thetas[:i],axis=2)]) 
#         points[i,1,:] = np.sum([L * np.sin(subset) for subset in np.cumsum(thetas[:i],axis=2)]) 

    theta_total = x_total = y_total = 0
    points = [[0, 0]]
    for theta in thetas:
        theta_total = (theta_total + theta) % (2 * np.pi)
        x_total += L * np.cos(theta_total)
        y_total += L * np.sin(theta_total)
        points += [np.array([x_total, y_total])]
    return points