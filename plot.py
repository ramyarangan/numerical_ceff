import matplotlib.pyplot as plt
import numpy as np

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
    'naive': 'bo',
    'SE(2)': 'ko'
}

def plot_ceffs(ceff_vals_lists, label_lists, fig_size=(8,5), cutoff=-100, sigma_min=-3, sigma_max=2, num_pts=10):
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