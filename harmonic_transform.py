import numpy as np
from scipy.special import jv
from scipy.special import erf
import math
from numpy import linalg as LA
from utils import *

def compute_rhohat_mn(m, n, p, L, theta_0, sigma):
    return 1j**(n-m) * jv( m-n, p*L ) * np.exp(1j*n*theta_0 - (n*sigma)**2)

def compute_rhohat_matrix(p, L, theta_0, sigma, cutoff=10):
    rhohat_matrix = np.zeros([2*cutoff+1, 2*cutoff+1], dtype=complex)
    for i in range(rhohat_matrix.shape[0]):
        m = i - cutoff
        for j in range(rhohat_matrix.shape[1]):
            n = j - cutoff
            rhohat_matrix[i][j] = compute_rhohat_mn(m, n, p, L, theta_0, sigma)
            
    return rhohat_matrix

# populate an (m, n, p) matrix with all values of (m, n) rhohat matrices for various p's
def get_rhohat_ps(theta_0, sigma, L, cutoff, n_links, p_range):
    rhohat_all = np.zeros((p_range.shape[0], 2*cutoff+1, 2*cutoff+1), dtype=complex)
    for i_p, p in enumerate(p_range):
        rhohat = compute_rhohat_matrix(p, L, theta_0, sigma, cutoff)
        final_entry = np.identity(2*cutoff+1)
        for nl in range(n_links):
            final_entry = final_entry @ rhohat
        rhohat_all[i_p] = final_entry
    return rhohat_all

def se2_ceff(sigma, theta_0, n_links, L=1, cutoff=10, p_max=300, n_theta_points = 1500):
    # n_theta_points POORLY named, is the number of p points. For now doing this
    # to be consistent with other integration methods.
    # but it should really be n_p_points or something

    del_p = p_max / n_theta_points
    P = np.arange(0, p_max, del_p)    
    m_n_range = 2*cutoff+1
    n_vals = np.arange(-cutoff, -cutoff+m_n_range)
    
    rhohat_all = get_rhohat_ps(theta_0, sigma, L, cutoff, n_links, P)
    
    prob_value = 0
    for m_idx in n_vals:
        prob_value += np.sum( rhohat_all[:, m_idx, m_idx] * P * del_p * jv(0, 0)/(4 * np.pi * np.pi) )
    return 2 * np.pi * np.real(prob_value)