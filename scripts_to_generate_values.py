import pandas as pd
import numpy as np
from ceff_wrapper import eval_ceff
import datetime as dt
import os

## Full list to test:
#     'analytical'
#     'analytical_sanity'
#     'infsig'
#     'Last Link 1'
#     'Last Link 2'
#     'RNAMake' or 'naive'
#     'grid1'
#     'grid2'
#    'gaussian'    
#   'mining_minima'
#    'SE(2)'

## GLOBAL METHODS LIST
all_methods = ['naive', 'grid1', 'grid2', 'grid1 kde', 'grid2 kde', \
    'Last Link 1', 'Last Link 2', 'analytical', 'infsig', 'gaussian', 'mining minima', 'SE(2)']
all_df_options = ['method', 'log_sigma', 'log_iterations', 'log_accept_size', \
                'log_integration_points', 'log_k_max', 'p_max', 'se2_order', \
                'n_p_points', 'time', 'Ceff']
## OPTIONS
log_iter_list = [2,3,4,5] # [2,3,4,5,6,7]
log_accept_size_list = [-2] # [-3, -2, -1, -0.5]
log_theta_list = [4] # [3,4,5,6]
log_k_max_list = [5] # [3,4,5,6]
p_max_list = [300] # [100,300,500]
se2_order_list = [10,20]
n_p_points_list = [1500] # [500, 1500, 5000]
log_sigma_list = np.linspace(-3, 2, 10)
n_stoch_trials = 2 # 10

# Does a grid search on all parameters that are method-specific, and adds ceff values and
# runtimes to the dataframe.
def eval_ceff_all_opts_df(log_sigma, theta_0, n_links, method, df):
    df_dict = {'method': method, 'log_sigma': log_sigma}
    for option in all_df_options:
        if option not in df_dict.keys():
            df_dict[option] = np.nan

    if method == 'naive':
        for log_iter in log_iter_list:
            for log_accept_size in log_accept_size_list:
                n1 = dt.datetime.now()
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter), \
                          d_theta=10**log_accept_size, d_xy=10**log_accept_size)
                n2 = dt.datetime.now()
                time_elapsed = (n2 - n1).microseconds/1e6
                df_dict['log_iterations'] = log_iter
                df_dict['log_accept_size'] = log_accept_size
                df_dict['time'] = time_elapsed
                df_dict['Ceff'] = ceff
                df = df.append(df_dict, ignore_index=True)
    
    if (method == 'grid1') or (method == 'grid2') or \
        (method == 'grid1 kde') or (method == 'grid2 kde'):
        for log_iter in log_iter_list:
            for log_accept_size in log_accept_size_list:
                n1 = dt.datetime.now()
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter), \
                            d_theta=10**log_accept_size, d_xy=10**log_accept_size)
                n2 = dt.datetime.now()
                time_elapsed = (n2 - n1).microseconds/1e6
                df_dict['log_iterations'] = log_iter
                df_dict['log_accept_size'] = log_accept_size
                df_dict['time'] = time_elapsed
                df_dict['Ceff'] = ceff
                df = df.append(df_dict, ignore_index=True)

    if method == 'Last Link 1':
        for log_iter in log_iter_list:
            for log_accept_size in log_accept_size_list:
                n1 = dt.datetime.now()
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter), \
                            d_r=10**log_accept_size, d_theta=10**log_accept_size)
                n2 = dt.datetime.now()
                time_elapsed = (n2 - n1).microseconds/1e6
                df_dict['log_iterations'] = log_iter
                df_dict['log_accept_size'] = log_accept_size
                df_dict['time'] = time_elapsed
                df_dict['Ceff'] = ceff
                df = df.append(df_dict, ignore_index=True)

    if method == 'Last Link 2':
        for log_iter in log_iter_list:
            n1 = dt.datetime.now()
            ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter))
            n2 = dt.datetime.now()
            time_elapsed = (n2 - n1).microseconds/1e6
            df_dict['log_iterations'] = log_iter
            df_dict['time'] = time_elapsed
            df_dict['Ceff'] = ceff
            df = df.append(df_dict, ignore_index=True)        

    if method == 'analytical':
        for log_theta in log_theta_list:
            n1 = dt.datetime.now()
            ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, n_theta_points=int(10**log_theta))
            n2 = dt.datetime.now()
            time_elapsed = (n2 - n1).microseconds/1e6
            df_dict['log_integration_points'] = log_theta
            df_dict['time'] = time_elapsed
            df_dict['Ceff'] = ceff
            df = df.append(df_dict, ignore_index=True)
                    
    if method == 'infsig':
        for log_k_max in log_k_max_list:
            for log_theta in log_theta_list:
                n1 = dt.datetime.now()
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, \
                        k_max = int(10**log_k_max), n_theta_points=int(10**log_theta))
                n2 = dt.datetime.now()
                time_elapsed = (n2 - n1).microseconds/1e6
                df_dict['log_k_max'] = log_k_max
                df_dict['log_integration_points'] = log_theta
                df_dict['time'] = time_elapsed
                df_dict['Ceff'] = ceff
                df = df.append(df_dict, ignore_index=True)   
    
    if method == 'gaussian':
        n1 = dt.datetime.now()
        ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method)
        n2 = dt.datetime.now()
        time_elapsed = (n2 - n1).microseconds/1e6
        df_dict['time'] = time_elapsed
        df_dict['Ceff'] = ceff
        df = df.append(df_dict, ignore_index=True)
        
    if method == 'mining minima':
        n1 = dt.datetime.now()
        ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method)
        n2 = dt.datetime.now()
        time_elapsed = (n2 - n1).microseconds/1e6
        df_dict['time'] = time_elapsed
        df_dict['Ceff'] = ceff
        df = df.append(df_dict, ignore_index=True)
    
    if method == 'SE(2)':
        for p_max in p_max_list:
            for se2_order in se2_order_list:
                for n_p_points in n_p_points_list:
                    n1 = dt.datetime.now()
                    ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, 
                                     cutoff=se2_order, p_max=p_max, n_p_points=n_p_points)
                    n2 = dt.datetime.now()
                    time_elapsed = (n2 - n1).microseconds/1e6
                    df_dict['p_max'] = p_max
                    df_dict['se2_order'] = se2_order
                    df_dict['n_p_points'] = n_p_points
                    df_dict['time'] = time_elapsed
                    df_dict['Ceff'] = ceff
                    df = df.append(df_dict, ignore_index=True)
    return df
            
def generate_data_all_opts(theta_0 = np.pi*2/4, n_links = 4,
                        filename='all_opts_test', method_list = all_methods, save=True):    
    for method in method_list:
        df = pd.DataFrame(columns = all_df_options)
        # Here we iterate through the parameters that are common to all methods
        for log_sigma in log_sigma_list: #range of sigma vals selected
            print("log sigma: %f, method: %s" % (log_sigma, method))
            for _ in range(n_stoch_trials):
                df = eval_ceff_all_opts_df(log_sigma, theta_0, n_links, method, df)
        if save: 
            # Only write the DF header once
            if not os.path.isfile('%s.csv' % filename):
                df.to_csv('%s.csv' % filename, header='column_names', sep = '\t')
            else:
                df.to_csv('%s.csv' % filename, mode='a', header=False, sep = '\t')
    return df

def generate_stochastic_data(theta_0 = np.pi*2/4, n_links = 4,
                        filename='stochastic_test', method_list = ['naive', 'grid1', 'grid2', 'grid1 kde','grid2 kde',
                         'Last Link 1', 'Last Link 2'], save=True):

    df = pd.DataFrame(columns = ['method','log_sigma', 'C_eff', 'log_iterations'])
    
    for log_sigma in log_sigma_list: #range of sigma vals selected
        for method in method_list:
            print("log sigma: %d, method: %s" % (log_sigma, method))
            for _ in range(n_stoch_trials):
                for log_iter in log_iter_list:
                    ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter))
                    df = df.append({'method': method, 'log_sigma': log_sigma, 'C_eff': ceff, 'log_iterations': log_iter}, ignore_index=True)
                    if save: df.to_csv('%s.txt' % filename, sep = '\t') # Check that this is writing the right thing?
    return df

def generate_integration_data(theta_0 = np.pi*2/4, n_links = 4, filename='integration_test', 
                                 method_list = ['infsig', 'analytical'], save=True):
    
    df = pd.DataFrame(columns = ['method','log_sigma', 'C_eff', 'log_integration_points'])
    
    for log_sigma in log_sigma_list: # range of sigma vals selected
        for method in method_list:
            print("log sigma: %d, method: %s" % (log_sigma, method))
            for log_theta in log_theta_list:
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, n_theta_points=int(10**log_iter))
                df = df.append({'method': method, 'log_sigma': log_sigma, 'C_eff': ceff, 'log_integration_points': log_theta}, ignore_index=True)
                if save: df.to_csv('%s.txt' % filename, sep = '\t') # Check that this is writing the right thing?
    return df

if __name__=='__main__':
    
    #quick test case for all
#     generate_stochastic_data(log_iter_list=[1])
#     generate_integration_data(log_iter_list=[1])
    
#     generate_stochastic_data(theta_0 = np.pi*2/4, n_links = 4, filename='4link_stochastic.txt')
#     generate_integration_data(theta_0 = np.pi*2/4, n_links = 4, filename='4link_integration.txt')
    generate_data_all_opts(filename='4link_all_opts')