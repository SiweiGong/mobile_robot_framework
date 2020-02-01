"""
The basic framework of the Iterative Closest Points Matching is provided by Albert-Ludwigs-Universität Freiburg,
the course Introduction to Mobile Robotics (engl.) - Autonomous Mobile Systems
Lecturer: Prof. Dr. Wolfram Burgard, Dr. Michael Tangermann, Dr. Daniel Büscher, Lukas Luft
Co-organizers: Marina Kollmitz, Iman Nematollahi
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def plot_icp(X, P, P0, i, rmse):
  plt.cla()
  plt.scatter(X[0,:], X[1,:], c='k', marker='o', s=50, lw=0)
  plt.scatter(P[0,:], P[1,:], c='r', marker='o', s=50, lw=0)
  plt.scatter(P0[0,:], P0[1,:], c='b', marker='o', s=50, lw=0)
  plt.legend(('X', 'P', 'P0'), loc='lower left')
  plt.plot(np.vstack((X[0,:], P[0,:])), np.vstack((X[1,:], P[1,:])) ,c='k')
  plt.title("Iteration: " + str(i) + "  RMSE: " + str(rmse))
  plt.axis([-10, 15, -10, 15])
  plt.gca().set_aspect('equal', adjustable='box')
  plt.draw()
  plt.pause(0.5)
  return
  
def generate_data():
  
  # create reference data  
  X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,-1,-2,-3,-4,-5]])
  
  # add noise
  P = X + 0.05 * np.random.normal(0, 1, X.shape)
  
  # translate
  P[0,:] = P[0,:] + 1
  P[1,:] = P[1,:] + 1
  
  # rotate
  theta1 = ( 10.0 / 360) * 2 * np.pi
  theta2 = (110.0 / 360) * 2 * np.pi
  rot1 = np.array([[math.cos(theta1), -math.sin(theta1)],
                   [math.sin(theta1),  math.cos(theta1)]])
  rot2 = np.array([[math.cos(theta2), -math.sin(theta2)],
                   [math.sin(theta2),  math.cos(theta2)]])
  
  # sets with known correspondences
  P1 = np.dot(rot1, P)
  P2 = np.dot(rot2, P)
  
  # sets with unknown correspondences
  P3 = np.random.permutation(P1.T).T
  P4 = np.random.permutation(P2.T).T
  
  return X, P1, P2, P3, P4

def closest_point_matching(X, P):
  """
  Performs closest point matching of two point sets.
  
  Arguments:
  X -- reference point set
  P -- point set to be matched with the reference
  
  Output:
  P_matched -- reordered P, so that the elements in P match the elements in X
  """

  P_matched = np.empty(np.shape(P))
  num_pts = np.shape(X)[1]
  dist_mat = np.empty((num_pts,num_pts))
  for i in range(num_pts):
    for j in range(num_pts):
      dist_mat[i,j] = np.linalg.norm(X[:,i] - P[:,j])

  rows, cols = [], []
  while(len(rows) != num_pts):
    min_ind = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
    if (min_ind[0] in rows) or (min_ind[1] in cols):
      dist_mat[min_ind] = math.inf
    else:
      P_matched[:,min_ind[0]] = P[:,min_ind[1]]
      dist_mat[min_ind] = math.inf
      rows.append(min_ind[0])
      cols.append(min_ind[1])

  return P_matched
  
def icp(X, P, do_matching):
  
  P0 = P  
  for i in range(10):  
    # calculate RMSE
    rmse = 0
    for j in range(P.shape[1]):
      rmse += math.pow(P[0,j] - X[0,j], 2) + math.pow(P[1,j] - X[1,j], 2)
    rmse = math.sqrt(rmse / P.shape[1])

    # print and plot
    print("Iteration:", i, " RMSE:", rmse)
    plot_icp(X, P, P0, i, rmse)
    
    # data association
    if do_matching:
      P = closest_point_matching(X, P)
    
    # substract center of mass
    mx = np.transpose([np.mean(X, 1)])
    mp = np.transpose([np.mean(P, 1)])
    X_prime = X - mx
    P_prime = P - mp
    
    # singular value decomposition
    W = np.dot(X_prime, P_prime.T)
    U, _, V = np.linalg.svd(W)
    
    # calculate rotation and translation
    R = np.dot(U, V.T)
    t = mx - np.dot(R, mp)
    
    # apply transformation
    P = np.dot(R, P) + t

  return
    
def main():
  
  X, P1, P2, P3, P4 = generate_data()
  
  # icp(X, P1, False)
  # icp(X, P2, False)
  # icp(X, P3, True)
  icp(X, P4, True)

  plt.waitforbuttonpress()
    
if __name__ == "__main__":
  main()
