import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde
import math
import random
from numpy import linalg as LA
from itertools import product
from utils import *

def mining_minima_explicit(sigma, n_links, cutoff=10000):
    if (n_links != 4):
        print("Explicit mining minima only implemented for 4 links")
        return 0

    theta_vals = 2 * np.pi * np.arange(-cutoff, cutoff)
    all_folded_gaussian_vals = np.exp(-theta_vals**2/(2 * sigma * sigma))
    Z = np.sum(all_folded_gaussian_vals)
    Z_unpaired = (Z/(math.sqrt(2 * np.pi) * sigma)) ** 4 # Z multiplied by normalization factor
    
    # Could instead approximate the Hessian numerically below
    dlogZ_dtheta2 = (1/Z) * (1/(sigma**2)) * (1 - np.sum(theta_vals**2 * all_folded_gaussian_vals)/(sigma**2))  
    if dlogZ_dtheta2 < 0:
        return 0 

    sigma_eff = 2/math.sqrt(dlogZ_dtheta2)
    Z_paired = math.sqrt(2 * np.pi) * sigma_eff * erf(np.pi/(math.sqrt(2) * sigma_eff)) 

    # Why are we multipying by Z_unpaired instead of dividing?
    # Not sure why I'm off by a factor of 4
    C_eff = 2 * np.pi * Z_unpaired * Z_paired/4
    return C_eff

# Gets energy of the configuration with the n_links-3 angles specified
# Finds the chain closure and uses this to calculate the energy 
# Chooses the closure solution that is closest to point_n2 (2nd to last point)

def get_energy_from_first_thetas(first_thetas, sigma, point_n2, theta_0, L):
    n_links = len(first_thetas) + 3 # not used?
    
    # Get x, y positions from first_thetas
    points = get_coords_from_thetas(first_thetas, L)
        
    # Get closure solutions and find the closure solution that is closest to the minimum
    [(x1, y1), (x2, y2)] = get_triangles(points[-1][0], points[-1][1], L)
    dist_1 = (point_n2[0] - x1[0])**2 + (point_n2[1] - y1[0])**2
    dist_2 = (point_n2[0] - x2[0])**2 + (point_n2[1] - y2[0])**2
    (x_n2, y_n2) = (x1[0], y1[0])
    if dist_2 < dist_1:
        (x_n2, y_n2) = (x2[0], y2[0])
    
    # Get angles for probability calculation
    thetas = np.copy(first_thetas)
    thetas = np.append(thetas,
                [np.pi - get_angle([points[-2][0], points[-2][1]], [points[-1][0], points[-1][1]], [x_n2, y_n2]),
                  np.pi - get_angle([points[-1][0], points[-1][1]], [x_n2, y_n2], [-L, 0]),
                  np.pi - get_angle([x_n2, y_n2], [-L, 0], [0, 0])])

    # Get Jacobian
    J = abs(get_J([[points[-1][0], points[-1][1]], [x_n2, y_n2], [-L, 0]]))

    # Compute energy
    ene = -math.log(np.prod(eval_folded_grid(sigma, thetas - theta_0))/J)
    
    return ene

# Gets energy of the configuration with one angle specified (this angle is repeated to close the chain)
# Returns the second-to-last points obtained by repeating the same angle n_links times
def get_energy_from_one_angle(theta, sigma, n_links, theta_0, L):
    if (n_links < 4):
        print("Cannot evaluate this function for n_links < 4")

    points = get_coords_from_thetas([theta]*n_links, L)
    
    Z = eval_folded(sigma, theta - theta_0)**n_links
    J = get_J(points[-4:-1])
    if (Z <= 0) or (J <= 1e-06):
        return [np.nan, points[-3]]
    ene = -math.log(Z/J)
    return [ene, points[-3]]

def mining_minima(sigma, theta_0, n_links, L=1, dtheta = 1e-5):

    # Minima have the form: all angles the same with all summing to 2*pi*k for integer k
    ceff = 0
    
    for i in range(n_links - 1):
        cur_angle = (i + 1) * 2 * np.pi/n_links
        first_thetas = np.array([cur_angle] * (n_links - 3))
        
        # Get energy of minimum and closure solution using get_energy_from_one_angle
        [ene, point_n2] = get_energy_from_one_angle(cur_angle, sigma, n_links, theta_0, L) 
        if np.isnan(ene):
            continue
        
        # Get Hessian: (f(x+dx, y+dy) - f(x+dx,y) - f(x,y+dy) + f(x,y))/(dx * dy)
        H = np.zeros((n_links - 3, n_links - 3))
        for ii in range(n_links - 3):
            for jj in range(n_links - 3):
                theta_dii = np.copy(first_thetas)
                theta_dii[ii] += dtheta
                ene_dii = get_energy_from_first_thetas(theta_dii, sigma, point_n2, theta_0, L)
                
                theta_djj = np.copy(first_thetas)
                theta_djj[jj] += dtheta
                ene_djj = get_energy_from_first_thetas(theta_djj, sigma, point_n2, theta_0, L)
                
                theta_dii_djj = np.copy(theta_dii)
                theta_dii_djj[jj] += dtheta
                ene_dii_djj = get_energy_from_first_thetas(theta_dii_djj, sigma, point_n2, theta_0, L)
                
                H[ii, jj] = (ene_dii_djj - ene_dii - ene_djj + ene)/(dtheta**2)
        
        # Get eigenvalue decomposition and check all eigenvalues positive
        w, _ = LA.eig(H)
        if np.sum(w < 0) > 0:
            continue
            
        # For each degree of freedom, add the contribution to Ceff
        cur_ceff = 2 * np.pi * math.exp(-ene)
        for ii in range(n_links - 3):
            sigma_eff = 2/math.sqrt(w[ii])
            cur_ceff *= (sigma_eff/2) * math.sqrt(2 * np.pi) * erf(math.sqrt(2) * np.pi/(sigma_eff))
        ceff += cur_ceff

    return ceff