from semianalytical import *
from stochastic import *
from gaussian_prop import *
from harmonic_transform import *
from mining_minima import *

def eval_ceff(sigma, theta_0, n_links, method_name='naive', **kwargs):
    '''Evaluate c_eff using one of the available methods.
    
    Methods:
    
    'analytical'
    'analytical_sanity'
    'infsig'
    'Last Link 1'
    'Last Link 2'
    'RNAMake' or 'naive'
    'grid1'
    'grid2'
    'gaussian'    
    'mining_minima'
    
    Use keyword args for specific functions to pass other options.
    
    returns:
    Ceff (float)
    
    '''
    
    if method_name == 'analytical':
        return analytical_ceff(sigma, theta_0, n_links, **kwargs)

    if method_name == 'infsig':
        return analytical_inf_sigma(n_links, **kwargs)

    if method_name == 'gaussian':
        return gaussian_ceff(sigma, theta_0, n_links, **kwargs)

    if method_name == 'Last Link 1':
        return last_link_1(sigma, theta_0, n_links, **kwargs)

    if method_name == 'Last Link 2' and n_links >= 4:
        return last_link_2(sigma, theta_0, n_links, **kwargs)

    if method_name == 'RNAMake' or method_name == 'naive':
        return naive_ceff(sigma, theta_0, n_links, **kwargs)

    if method_name == 'mining minima':
        return mining_minima(sigma, theta_0, n_links, **kwargs)

    if method_name == 'grid1':
        return grid_ceff(sigma, theta_0, n_links - 1, 1, **kwargs)

    if method_name == 'grid2' and n_links >= 4:
        return grid_ceff(sigma, theta_0, n_links - 2, 2, **kwargs)

    if method_name == 'grid1 kde':
        return grid_ceff(sigma, theta_0, n_links - 1, 1, density_method='kde', **kwargs)

    if method_name == 'grid2 kde' and n_links >= 4:
        return grid_ceff(sigma, theta_0, n_links - 2, 2, density_method='kde', **kwargs)

    if method_name == 'SE(2) order 10' or method_name == 'SE(2)':
        return se2_ceff(sigma, theta_0, n_links, **kwargs)

    print("Failed to evaluate method with these configurations")
    return []

def eval_ceff_range(theta_0, n_links, method_name, sigma_min=-3, sigma_max=2, verbose=True, num_pts=10, **kwargs):
    sigma_range = np.logspace(sigma_min, sigma_max, num_pts)
   
    ceff_vals = []
    for sigma in sigma_range:
        if verbose: print(sigma)
        ceff_vals += [eval_ceff(sigma, theta_0, n_links, method_name, **kwargs)]
    return ceff_vals