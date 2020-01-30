import numpy as np
from scipy.stats import gaussian_kde
import math
import random
import matplotlib as plt
from numpy import linalg as LA
from itertools import product
from utils import *
from copy import copy
# from fast_histogram import histogram3d

def naive_ceff(sigma, theta_0, n_links, L=1, niter=int(1e6), d_theta = 0.01, d_xy = 0.01):
    [theta_totals, x_totals, y_totals] = generate_for_samples(sigma, theta_0=theta_0, n_links=n_links, L=L, n_iter=niter)

    x_pass = np.abs(x_totals) < d_xy
    y_pass = np.abs(y_totals) < d_xy
    theta_pass = np.abs(theta_totals-np.pi*2) < d_theta
        
    prob = np.sum(x_pass * y_pass * theta_pass)/niter
    ceff = 2 * np.pi * prob/(np.pi * d_theta * d_xy * d_xy)
    return ceff

def grid_ceff(sigma, theta_0, nfor, nrev, density_method='hist', L=1, niter=int(1e4), d_theta = 0.01, d_xy = 0.01):
    '''
    'density_method': 'hist' or 'kde'
    '''
    
    # Generate forward samples
    [theta_totals, x_totals, y_totals] = generate_for_samples(sigma, theta_0=theta_0, n_links=nfor, L=L, n_iter=niter)

    # Generate reverse samples
    if nrev == 1 and density_method=='kde': # and kde
        add_noise = True
    else:
        add_noise = False

    [theta_rev_totals, x_rev_totals, y_rev_totals] = generate_rev_samples(sigma, theta_0=theta_0, 
            n_links=nrev, L=L, n_iter = niter, add_noise=add_noise)

    # plt.scatter(x_totals,y_totals)
    # plt.scatter(x_rev_totals,y_rev_totals)

    if density_method=='hist':
        
        # Define x, y, theta bin dimensions for histogram
        xmin = ymin = -2 * L
        xmax = ymax = 2 * L

        n_x_bins, n_y_bins = int((xmax - xmin)/d_xy)+1, int((ymax - ymin)/d_xy)+1 # + 1 to put integers in bins
        n_theta_bins = int(L/d_theta)

        arr_data = np.concatenate(([theta_totals], [x_totals], [y_totals])).T
        rev_arr_data = np.concatenate(([theta_rev_totals], [x_rev_totals], [y_rev_totals])).T
        
        # Assemble histogram using sampled values
        # density=True takes care of dividing by the total count and the bin volumes
    
        for_p_arr, _ = np.histogramdd(arr_data, bins=(n_theta_bins, n_x_bins, n_y_bins), 
            range=[(0,2*np.pi), (xmin,xmax), (ymin,ymax)], density=True)

        rev_p_arr, _ = np.histogramdd(rev_arr_data, bins=(n_theta_bins, n_x_bins, n_y_bins), 
            range=[(0,2*np.pi), (xmin,xmax), (ymin,ymax)], density=True)
        
        product = np.sum(np.multiply(for_p_arr, rev_p_arr))*d_theta*d_xy*d_xy

    elif density_method=='fast_hist':
        # Define x, y, theta bin dimensions for histogram
        xmin = ymin = -2 * L
        xmax = ymax = 2 * L

        n_theta_bins = int(L/d_theta)
        n_x_bins, n_y_bins = int((xmax - xmin)/d_xy)+1, int((ymax - ymin)/d_xy)+1
        
        for_p_arr = histogram3d(theta_totals, x_totals, y_totals, [n_theta_bins, n_x_bins, n_y_bins],
         [(0,2*np.pi), (xmin,xmax), (ymin,ymax)])  

        rev_p_arr = histogram3d(theta_rev_totals, x_rev_totals, y_rev_totals, [n_theta_bins, n_x_bins, n_y_bins],
         [(0,2*np.pi), (xmin,xmax), (ymin,ymax)])  

        bin_vol = (xmax-xmin)*(ymax-ymin)*2*np.pi/(n_x_bins*n_y_bins*n_theta_bins)
        # divide by total count and bin volumes ourselves here for this method

        for_p_arr /= (for_p_arr.sum() * bin_vol)
        rev_p_arr /= (rev_p_arr.sum() * bin_vol)

        product = np.sum(np.multiply(for_p_arr, rev_p_arr))*d_theta*d_xy*d_xy


    elif density_method=='kde':
        
        arr_data = np.concatenate(([theta_totals], [x_totals], [y_totals]))
        rev_arr_data = np.concatenate(([theta_rev_totals], [x_rev_totals], [y_rev_totals]))
        
        for_kde = gaussian_kde(arr_data)
        rev_kde = gaussian_kde(rev_arr_data)
        
        product = for_kde.integrate_kde(rev_kde)

    return 2 * np.pi * product

def last_link_1(sigma, theta_0, n_links, L=1, niter=10000, d_r=0.01, d_theta=0.01):
    theta_prod = np.ones(niter)

    # Sample first links
    [theta_totals, x_totals, y_totals] = generate_for_samples(sigma, theta_0, n_links-1, L, niter)

    # Last link
    theta_vals = 2 * np.pi - theta_totals
    folded_vals = wrapped_gaussian(theta_vals-theta_0, sigma)
    #folded_vals = eval_folded_grid(sigma, theta_vals - theta_0)

    # Indicator for length pass
    r_pass = np.abs(L - np.sqrt(x_totals**2 + y_totals**2)) < d_r
    r_pass = r_pass.astype(int)
    
    # Indicator for angle pass
    phi_final = np.arctan2(x_totals * np.sin(theta_totals) - y_totals * np.cos(theta_totals), \
        - x_totals * np.cos(theta_totals) - y_totals * np.sin(theta_totals))
    phi_pass = np.abs(theta_vals - phi_final) < d_theta
    phi_pass = phi_pass.astype(int)

    prob = sum(r_pass * folded_vals * phi_pass)/niter
    return 2 * np.pi * prob/(np.pi * d_r * d_theta)

def last_link_2_special(sigma, theta_0, n_links, L=1, niter=10000):
    if (n_links != 4):
        print("Cannot compute two last links' Ceff for nlinks not 4")
        return -1        
    
    # Get theta samples for one link
    theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=niter)

    # For the one correct solution, calculate the area of the parallelopiped
    det_vals = np.abs(np.cos(theta_vals - theta_0))

    # Multiply by folded Gaussian values
    # Add all vals and multiply by 2 pi
    ceff = 2 * np.pi * np.sum(eval_folded_grid(sigma, theta_vals - theta_0)**3/det_vals)/niter
    return ceff

def generate_last_two_samples(sigma, theta_0, nfor, L, niter):

    theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=(nfor, niter)) % (2*np.pi)
    final_theta_vals = np.sum(theta_vals,axis=0) % (2*np.pi)
    coords = get_coords_from_thetas(theta_vals) #, prepend_origin = False)

    return [final_theta_vals, coords[-1,0], coords[-1,1], coords[-2,0], coords[-2,1]]

    # theta_totals = np.zeros(niter)
    # x_totals = np.zeros(niter)
    # y_totals = np.zeros(niter)
    # x_prev = np.zeros(niter)
    # y_prev = np.zeros(niter)
    # for ii in range(nfor):
    #     theta_vals = np.random.normal(loc=theta_0, scale=sigma, size=niter)
    #     theta_totals += theta_vals
    #     theta_totals %= 2 * np.pi
    #     x_prev = np.copy(x_totals)
    #     y_prev = np.copy(y_totals)
    #     x_totals += L * np.cos(theta_totals)
    #     y_totals += L * np.sin(theta_totals)
    # return [theta_totals, x_totals, y_totals, x_prev, y_prev]

def last_link_2(sigma, theta_0, n_links, L=1, niter=1000):
    if (n_links < 4):
        print("Cannot compute two last links' Ceff for nlinks less than 4")
        return -1        
    
    # Get theta samples for one link
    [theta_totals, x_totals, y_totals, x_prev, y_prev] = \
        generate_last_two_samples(sigma, theta_0, n_links - 3, L, niter)
    
    # Get remaining points
    [(x1, y1), (x2, y2)] = get_triangles(x_totals, y_totals, L)
    
    # Get angles for probability calculation
    zeros = np.zeros(niter)
    angle1_1 = np.pi - get_angle([x_prev, y_prev], [x_totals, y_totals], [x1, y1])
    angle1_2 = np.pi - get_angle([x_prev, y_prev], [x_totals, y_totals], [x2, y2])
    angle2_1 = np.pi - get_angle([x_totals, y_totals], [x1, y1], [-L * np.ones(niter), zeros])
    angle2_2 = np.pi - get_angle([x_totals, y_totals], [x2, y2], [-L * np.ones(niter), zeros])
    angle3_1 = np.pi - get_angle([x1, y1], [-L * np.ones(niter), zeros], [zeros, zeros])
    angle3_2 = np.pi - get_angle([x2, y2], [-L * np.ones(niter), zeros], [zeros, zeros])

    # Vectorized determinant calculation 
    det_vals_1 = np.abs(get_J([[x_totals, y_totals], [x1, y1], [-L * np.ones(niter), zeros]]))
    det_vals_2 = np.abs(get_J([[x_totals, y_totals], [x2, y2], [-L * np.ones(niter), zeros]]))

    # Evaluate probability value with folded Gaussians
    soln_1 = np.nansum(eval_folded_grid(sigma, angle1_1 - theta_0) * \
                eval_folded_grid(sigma, angle2_1 - theta_0) * \
                eval_folded_grid(sigma, angle3_1 - theta_0)/det_vals_1)
    soln_2 = np.nansum(eval_folded_grid(sigma, angle1_2 - theta_0) * \
                eval_folded_grid(sigma, angle2_2 - theta_0) * \
                eval_folded_grid(sigma, angle3_2 - theta_0)/det_vals_2)

    # Add all vals and multiply by 2 pi
    ceff = 2 * np.pi * (soln_1 + soln_2)/niter
    return ceff
