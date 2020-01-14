import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.special import erf
from scipy.stats import gaussian_kde
import math
import random
from numpy import linalg as LA

def eval_folded(sigma, theta, cutoff=10000):
	# Evaluates a folded Gaussian centered at 0 with variance sigma at the point theta

	total = 0
	for n in range(-cutoff, cutoff):
		total += math.exp(-((theta - 2 * np.pi * n)**2) / (2 * sigma * sigma))
	return 1/(math.sqrt(2 * np.pi) * sigma) * total

def eval_folded_grid(sigma, theta_arr, cutoff=10000):
	# Evaluates a folded Gaussian centered at 0 with variance sigma at the points in the theta_arr

	total = np.zeros(len(theta_arr))
	for n in range(-cutoff, cutoff):
		total += np.exp(-((theta_arr - 2 * np.pi * n)**2) / (2 * sigma * sigma))
	return 1/(math.sqrt(2 * np.pi) * sigma) * total

def wrapped_gaussian(dtheta, sigma, cutoff=1000):
	# HW version -- used in semianalytical

	cc,tt = np.meshgrid([x for x in range(-cutoff,cutoff+1)],dtheta)
	return (1/np.sqrt(2*np.pi)/sigma)*np.sum(np.exp(-(tt-2*np.pi*cc)**2/2/sigma**2),axis=1)

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

def get_angle(point1, point2, point3):
	# points can be (x_np_array, y_np_array) for vectorization 
	# Gets angle between point1, point2, and point3. 
	# Note, to get the angle of turning from the vector point1-point2 to the vector point2-point3, 
	# you may have to do np.pi minus this value.
	vec1_x = point2[0] - point1[0]
	vec1_y = point2[1] - point1[1]
	vec2_x = point2[0] - point3[0]
	vec2_y = point2[1] - point3[1]

	vec1_norm = np.sqrt(vec1_x**2 + vec1_y**2)
	vec2_norm = np.sqrt(vec2_x**2 + vec2_y**2)
	vec_dot = vec1_x * vec2_x + vec1_y * vec2_y
	cos_val = np.clip(vec_dot/(vec1_norm * vec2_norm), -1, 1)
	return np.arccos(cos_val)

def get_coords_from_thetas(thetas, L=1, prepend_origin=True):
	# Given array-like thetas and float L (link length), compute set of corresponding xy coordinates of chain.
	
	# Inputs
	# thetas: If single instance, can be list or nx1 array.  If multiple cases (i.e. for vectorized
	# samples, first index should be theta position, second should be sample index. I.e. for 4-link case (one theta),
	# would be 1x100 for 100 samples.)
	
	# Returns: array of points with dimensions (theta index, 2, sample index. (2 is for x or y)).
	
	was_list = False
	if isinstance(thetas,list):
		was_list = True

	thetas = np.atleast_2d(thetas)
	
	if thetas.shape[0] == 1 and thetas.shape[1] > 1 and was_list:
		thetas = np.transpose(thetas)
	
	x_coords = np.cumsum(L * np.cos( np.cumsum(thetas, axis = 0) % (np.pi*2)), axis = 0)
	y_coords = np.cumsum(L * np.sin( np.cumsum(thetas, axis = 0) % (np.pi*2)), axis = 0)
		
	if prepend_origin:
		zero_prefix = np.zeros([1, thetas.shape[1]])
		x_coords = np.vstack([zero_prefix, x_coords])
		y_coords = np.vstack([zero_prefix, y_coords])

	points = np.swapaxes(np.array([x_coords,y_coords]),0,1)
	
	if points.shape[-1] == 1:
		points = np.squeeze(points,axis=-1)

	return points

def generate_for_samples(sigma, theta_0=0, n_links=3, L=1, n_iter=10):
	'''Propagate chains forward by sampling wrapped gaussian N(theta_0, sigma).

	Returns: [final_theta_vals, x_coords, y_coords]

	final_theta_vals: list of final theta vals, length n_iter long.
	x_coords: list of final x vals, length n_iter long.
	y_coords: list of final y vals, length n_iter long.'''

	theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=(n_links, n_iter)) % (2*np.pi)
	final_theta_vals = np.sum(theta_vals,axis=0) % (2*np.pi)
	coords = get_coords_from_thetas(theta_vals) #, prepend_origin = False)

	return [final_theta_vals, np.squeeze(coords[-1,0]), np.squeeze(coords[-1,1])]
	
def generate_rev_samples(sigma, theta_0=0, n_links=1, L=1, n_iter=10, add_noise=False):
	'''Propagate reverse chains by sampling wrapped gaussian N(theta_0, sigma).

	Returns: [final_theta_vals, x_coords, y_coords]

	final_theta_vals: list of final theta vals, length n_iter long.
	x_coords: list of final x vals, length n_iter long.
	y_coords: list of final y vals, length n_iter long.'''

	if add_noise:
		x_rev = np.random.normal(loc=-1*L, scale=0.0001, size=n_iter)
		y_rev = np.random.normal(loc=0, scale=0.0001, size=n_iter)
	else:
		x_rev = -1 * L * np.ones([n_iter])
		y_rev = np.zeros([n_iter])

	theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=(n_links, n_iter)) % (2*np.pi)
	theta_rev_totals = 2*np.pi - np.sum(theta_vals,axis=0) % (2*np.pi)

	if n_links == 1:
		return [theta_rev_totals, x_rev, y_rev]
	else:
		coords = get_coords_from_thetas(-1*theta_vals[:-1])

		x_rev = x_rev - np.squeeze(coords[-1,0]) # final coords
		y_rev = y_rev - np.squeeze(coords[-1,1])

		return [theta_rev_totals, x_rev, y_rev]
	
	# theta_rev_totals = np.zeros(niter)
	# x_rev_totals = np.zeros(niter)
	# y_rev_totals = np.zeros(niter)

	# for ii in range(nrev):
		
	#     x_rev_totals -= L * np.cos(2 * np.pi - theta_rev_totals)
		
	#     if add_noise and nrev == 1:
	#         x_rev_totals += np.random.normal(loc=0, scale=0.001, size=niter)
			
	#     y_rev_totals -= L * np.sin(2 * np.pi - theta_rev_totals)

	#     if add_noise and nrev == 1:
	#         y_rev_totals += np.random.normal(loc=0, scale=0.001, size=niter)

	#     theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=niter)
	#     theta_rev_totals += theta_vals

	# theta_rev_totals = 2 * np.pi - theta_rev_totals
	# theta_rev_totals %= 2 * np.pi
