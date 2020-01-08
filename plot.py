import matplotlib.pyplot as plt


style_dict = {
    'analytical': 'r',
    'infsig': 'k', 
    'gaussian': 'co', 
    'Last Link 1': 'gs', 
    'Last Link 2': 'ys', 
    'RNAMake': 'bo', 
    'mining minima': 'cs', 
    'grid1': 'go', 
    'grid2': 'mo', 
    'grid1 kde': 'bs',
    'grid2 kde': 'ms', 
    'SE(2) order 10': 'ko', 
    'naive': 'bo'
}

def plot_ceffs(sigma_min, sigma_max, num_pts, ceff_vals_lists, label_lists, fig_size=(8,5), cutoff=-100):
    sigma_range = np.logspace(sigma_min, sigma_max, num_pts)
    plt.figure(figsize=fig_size)
    for ii, ceff_list in enumerate(ceff_vals_lists):
        cur_label = label_lists[ii]
        if cur_label not in style_dict: 
            print("Please add style entry for the plot label: %s" % cur_label)
            continue
        ceff_list = np.array(ceff_list)
        ceff_list[ceff_list < 10**cutoff] = 0
        plt.plot(sigma_range, ceff_list, style_dict[cur_label], markersize=5, label=cur_label)
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
## PLOTTING PARAMETERS
SIGMA_MIN=-3
SIGMA_MAX=2
NUM_PTS=10

def eval_ceff(sigma, theta_0, n_links, method_name):
    if method_name == 'analytical':
        return analytical_ceff(sigma, n_links)
    if method_name == 'infsig':
        return analytical_inf_sigma(n_links)
    if method_name == 'gaussian':
        return gaussian_ceff(sigma, theta_0, n_links)
    if method_name == 'Last Link 1':
        return last_link_1(sigma, theta_0, n_links)
    if method_name == 'Last Link 2' and n_links >= 4:
        return last_link_2(sigma, theta_0, n_links)
    if method_name == 'RNAMake' or method_name == 'naive':
        return numerical_ceff(sigma, theta_0, n_links)
    if method_name == 'mining minima':
        return mining_minima(sigma, theta_0, n_links)
    if method_name == 'grid1':
        return grid_ceff(sigma, theta_0, n_links - 1, 1)
    if method_name == 'grid2' and n_links >= 4:
        return grid_ceff(sigma, theta_0, n_links - 2, 2)
    if method_name == 'grid1 kde':
        return grid_ceff_kde(sigma, theta_0, n_links - 1, 1)
    if method_name == 'grid2 kde' and n_links >= 4:
        return grid_ceff_kde(sigma, theta_0, n_links - 2, 2)
    if method_name == 'SE(2) order 10':
        return se2_ceff(sigma, theta_0, n_links)
    print("Failed to evaluate method with these configurations")
    return []

def eval_ceff_range(sigma_min, sigma_max, num_pts, theta_0, n_links, method_name):
    sigma_range = np.logspace(sigma_min, sigma_max, num_pts)
   
    ceff_vals = []
    for sigma in sigma_range:
        print(sigma)
        ceff_vals += [eval_ceff(sigma, theta_0, n_links, method_name)]
    return ceff_vals