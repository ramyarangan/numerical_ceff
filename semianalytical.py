import numpy as np
import math
from scipy.special import jv
from itertools import product
from utils import *

def analytical_inf_sigma(n_links, k_min=0, k_max=100000, n_theta_points = 1000000, L=1):
    #TODO: Rename this to reflect Fourier rigid rod

    del_k = (k_max - k_min)/n_theta_points
    k_vals = np.arange(k_min, k_max, del_k)

    return np.sum([jv(0,k)**n_links for k in k_vals]*k_vals*del_k) / (2 * np.pi) 

    # Divide by 2 pi because this probability currently sums over all values of theta

def analytical_ceff_3(sigma):
    J = math.sqrt(3)/2
    ceff = 2 * np.pi * (eval_folded(sigma, 0)**3 + eval_folded(sigma, 2*np.pi/3)**3)/J
    return ceff
    
def old_analytical_ceff_4(sigma):
    theta_vals = np.arange(-np.pi, np.pi, np.pi/100)
    ceff = 0
    for theta in theta_vals:
        points0 = np.array([0, 0])
        points1 = np.array([math.cos(theta), math.sin(theta)])
        points2 = points1 + np.array([-1, 0])
        points3 = points2 + np.array([-math.cos(theta), -math.sin(theta)])
        points = [points0, points1, points2, points3]
        J = abs(get_J(points))
        ceff += (eval_folded(sigma, theta)**4) * 2 * np.pi/abs(np.cos(theta)) * np.pi/100 #J
    return ceff
    
def analytical_ceff(sigma, theta_0, n_links, n_theta_points = int(1e4), sanity_check=False, L=1, plot_paths=False):
    
    theta_vals = np.linspace(-np.pi+1e-4, np.pi-1e-4, n_theta_points, endpoint=True)
    d_theta = theta_vals[1] - theta_vals[0]
    
    if (n_links == 3):
        return analytical_ceff_3(sigma)
    
    if (n_links == 4 and sanity_check):
        return old_analytical_ceff_4(sigma)
    else:

        n_thetas_to_sample = int(n_links-3)
        ceff = 0
        theta_iter = np.transpose(np.array(list(product(theta_vals, repeat = n_thetas_to_sample))))
        # shape: (n_theta_points^(n-3),n-3)
        
        zero_prefix = np.zeros([1, theta_iter.shape[1]])
        x_coords = np.vstack([zero_prefix, np.cumsum(L*np.cos(np.cumsum(theta_iter,axis=0) % (np.pi*2)), axis=0)])
        y_coords = np.vstack([zero_prefix, np.cumsum(L*np.sin(np.cumsum(theta_iter,axis=0) % (np.pi*2)), axis=0)])
        #shape: (n-3 (# thetas), n_theta_points^(n-3) (# samples))
        
        # Get closure solutions (there's 2)
        [(x1, y1), (x2, y2)] = get_triangles(x_coords[-1], y_coords[-1], L)
        #each is vector length n_samples (n_theta_pts^(n-3))
        
        nan_locs = np.isnan(x2)
            
        x1 = x1[~nan_locs]
        y1 = y1[~nan_locs]
        x2 = x2[~nan_locs]
        y2 = y2[~nan_locs]
        
        theta_iter = theta_iter[:,~nan_locs]
        x_coords = x_coords[:,~nan_locs]
        y_coords = y_coords[:,~nan_locs]


        for (x_n,y_n) in [(x1, y1), (x2, y2)]:
            # Get angles for probability calculation
            thetas = np.copy(theta_iter)
            vals_to_append = np.array([np.pi - get_angle([x_coords[-2], y_coords[-2]], [x_coords[-1], y_coords[-1]], [x_n, y_n]),
                          np.pi - get_angle([x_coords[-1], y_coords[-1]], [x_n, y_n], [-L, 0]),
                          np.pi - get_angle([x_n, y_n], [-L, 0], [0, 0])])
            
            nan_vals = np.isnan(vals_to_append)
            vals_to_append = vals_to_append[:,~nan_vals.any(axis=0)]
            thetas = thetas[:,~nan_vals.any(axis=0)]
            xloc = x_coords[:,~nan_vals.any(axis=0)]
            yloc = y_coords[:,~nan_vals.any(axis=0)]
            x_n_loc = x_n[~nan_vals.any(axis=0)]
            y_n_loc = y_n[~nan_vals.any(axis=0)]

            thetas = np.vstack([thetas, vals_to_append])
            
            if plot_paths:
                plt.plot(x_coords,y_coords)

            J = abs(get_J([[x_coords[-1], y_coords[-1]], [x_n, y_n], [-L, 0]]))
            
            prob_array = wrapped_gaussian(thetas - theta_0, sigma).reshape(n_links, -1)
            prob_array[np.isnan(prob_array)] = 0 #deal with NaNs from paths that didn't close
            J[np.isnan(J)] = 1
            ceff += 2 * np.pi * d_theta * np.sum(np.prod(prob_array,axis=0) / J)
            
        return ceff