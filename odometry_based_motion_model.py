import sample_function as sf
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def sample_function(mean,std):
    return sf.sample_normal_distribution_numpy(mean,std)

def odometry_based_motion_model(x,u,alpha):
    measured_delta_rot1 = u[0] + sample_function(0,alpha[0]*np.abs(u[0])+alpha[1]*np.abs(u[-1]))
    measured_delta_rot2 = u[1] + sample_function(0,alpha[0]*np.abs(u[1])+alpha[1]*np.abs(u[-1]))
    measured_delta_trans = u[-1] + sample_function(0,alpha[2]*np.abs(u[-1])+alpha[3]*(np.abs(u[0])+np.abs(u[1])))
    theta_post = x[-1] + measured_delta_rot1 + measured_delta_rot2
    x_post = x[0] + measured_delta_trans*np.cos(x[-1]+measured_delta_rot1)
    y_post = x[1] + measured_delta_trans*np.sin(x[-1]+measured_delta_rot1)
    return x_post, y_post, theta_post

pose_start = np.array([2.0,4.0,0.0])
odometry_reading_start = np.array([np.pi/2,0.0,1.0])
noise_param = np.array([0.1,0.1,0.01,0.01])
num_eval = 5000
new_pose = np.empty((num_eval,3))
plt.figure()
for i in range(num_eval):
    new_pose[i,:] = odometry_based_motion_model(pose_start,odometry_reading_start,noise_param)

plt.scatter(new_pose[:,0],new_pose[:,1],s=0.1)
plt.quiver(pose_start[0],pose_start[1],np.cos(pose_start[-1]),np.sin(pose_start[-1]))
plt.show()