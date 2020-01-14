import pandas as pd
import numpy as np
from ceff_wrapper import eval_ceff

def generate_stochastic_data(theta_0 = np.pi*2/4, n_links = 4,
                        filename='stochastic_test', method_list = ['naive', 'grid1', 'grid2', 'grid1 kde','grid2 kde',
                         'Last Link 1', 'Last Link 2'], log_sigma_list = [-3,0,2], n_stoch_trials = 10, log_iter_list = [2,3,4,5,6,7], save=True):

    df = pd.DataFrame(columns = ['method','log_sigma', 'C_eff', 'log_iterations'])

    for log_sigma in log_sigma_list: #range of sigma vals selected
        for method in method_list:
            print("log sigma: %d, method: %s" % (log_sigma, method))
            for log_iter in log_iter_list:
                for _ in range(n_stoch_trials):
                    ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, niter=int(10**log_iter))
                    df = df.append({'method': method, 'log_sigma': log_sigma, 'C_eff': ceff, 'log_iterations': log_iter}, ignore_index=True)
                    if save: df.to_csv('%s.txt' % filename, sep = '\t')
    return df

def generate_integration_data(theta_0 = np.pi*2/4, n_links = 4, filename='integration_test', 
                                 method_list = ['infsig', 'analytical', 'SE(2) order 10'], 
                                 log_sigma_list = [-3,0,2], n_stoch_trials = 10, log_iter_list = [3,4,5,6], save=True):
    
    df = pd.DataFrame(columns = ['method','log_sigma', 'C_eff', 'log_integration_points'])

    for log_sigma in log_sigma_list: # range of sigma vals selected
        for method in method_list:
            print("log sigma: %d, method: %s" % (log_sigma, method))
            for log_iter in log_iter_list:
                ceff = eval_ceff(10**log_sigma, theta_0, n_links, method_name=method, n_theta_points=int(10**log_iter))
                df = df.append({'method': method, 'log_sigma': log_sigma, 'C_eff': ceff, 'log_integration_points': log_iter}, ignore_index=True)
                if save: df.to_csv('%s.txt' % filename, sep = '\t')
    return df

if __name__=='__main__':
    
    #quick test case for all
#     generate_stochastic_data(log_iter_list=[1])
#     generate_integration_data(log_iter_list=[1])
    
    generate_stochastic_data(theta_0 = np.pi*2/4, n_links = 4, filename='4link_stochastic.txt')
    generate_integration_data(theta_0 = np.pi*2/4, n_links = 4, filename='4link_integration.txt')