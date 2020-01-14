import numpy as np
import math
from utils import *


def gaussian_ceff(sigma, theta_0, n_links, L=1):
    start = np.array([[0, 0, 0], [0, 0, 0], [0, 0, sigma*sigma]]) # Covariance matrix
    rinv = np.array([[math.cos(theta_0), math.sin(theta_0), L*math.cos(theta_0)], \
        [-math.sin(theta_0), math.cos(theta_0), L*math.sin(theta_0)],[0, 0, 1]]) # Coordinate transform in SE2
    cur_mat = start
    for ii in range(n_links-1):
        cur_mat = np.matmul(np.matmul(rinv, cur_mat), rinv.transpose()) + start
    return 2 * np.pi/(math.sqrt(abs(np.linalg.det(cur_mat))) * (2 * np.pi)**(1.5))