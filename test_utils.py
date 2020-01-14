from utils import *

def test_get_J():
    J_ans_3 = math.sqrt(3)/2
    points0 = np.array([0, 0])
    points1 = np.array([-0.5, math.sqrt(3)/2])
    points2 = np.array([-1, 0])
    J_3 = get_J([points0, points1, points2])
    print(J_ans_3, J_3)
    
def test_J_four_links():
    # Checking Jacobian calculation for four links
    # Also checking vectorization for determinant calculation

    theta = np.arange(-np.pi, np.pi, np.pi/20)
    theta_0 = np.ones(len(theta)) * np.pi/2

    points1 = np.array([np.cos(theta_0 + theta), np.sin(theta_0 + theta)])
    points2 = points1 + np.array([-1 * np.ones(len(theta)), np.zeros(len(theta))])
    points3 = points2 + np.array([-np.cos(theta_0 + theta), -np.sin(theta_0 + theta)])
    points = [points1, points2, points3]
    J = np.abs(get_J(points))
    J_ans = np.abs(np.cos(theta))

    for ii in range(len(theta)):
        print("%f, %f" % (J[ii], J_ans[ii]))
        
def test_triangle():
    # Check geometry formulas
    print(get_triangles(0, 1, 1))
    print(get_triangles(-0.2, 1, 1))
    print(get_angle((0, 1), (-1, 1), (-1, 0)))
    print(get_angle((np.array([0, 0]), np.array([1, 1])), \
                   (np.array([-1, -1]), np.array([1, 1])), \
                  (np.array([-1, -1]), np.array([0, 0]))))
    
def test_get_thetas():
    thetas = [[0,0,0],[1,1,1]]
    print(get_coords_from_thetas(thetas,1))
    
def test_gen_for_samples():
    print(generate_for_samples(1, theta_0=0, n_links=3, L=1, n_iter=10))

def test_gen_back_samples():
    print(generate_back_samples(1, theta_0=0, n_links=3, L=1, n_iter=10))
    
if __name__ == '__main__':
    
    #test_get_J()
    #test_J_four_links()
    #test_triangle()
    #test_get_thetas()
    test_gen_for_samples()
    
    

