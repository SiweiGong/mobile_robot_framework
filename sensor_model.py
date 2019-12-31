import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def prob_density_func(x,mean,std):
    return scipy.stats.norm.pdf(x,mean,std)

def calc_likelyhood_measurements_given_position(position,landmarks,measurements):
    distance2landmarks = np.empty((np.shape(landmarks)[0],))
    likelyhood_measurements_given_position = np.empty((np.shape(landmarks)[0],))
    for i in range(np.shape(landmarks)[0]):
        distance2landmarks[i] = np.linalg.norm(position-landmarks[i,:])
        likelyhood_measurements_given_position[i] = prob_density_func(measurements[i,0],measurements[i,1]+distance2landmarks[i],measurements[i,-1])
    likelyhood = 1.0
    for i in range(np.shape(likelyhood_measurements_given_position)[0]):
        likelyhood = likelyhood*likelyhood_measurements_given_position[i]
    return likelyhood

def plot_likelyhood_function(plot_vicinity,step,landmarks,measurements,plot_location):
    x = np.arange(plot_vicinity[0,0],plot_vicinity[1,0],step)
    y = np.arange(plot_vicinity[0,1],plot_vicinity[1,1],step)
    xx, yy = np.meshgrid(x,y)
    p = np.empty(np.shape(xx))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(y)[0]):
            p[j,i] = calc_likelyhood_measurements_given_position(np.array([x[i],y[j]]),landmarks,measurements)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx,yy,p,rstride=1,cstride=1,cmap=cm.coolwarm,alpha=0.5)
    if plot_location.any():
        print(calc_likelyhood_measurements_given_position(plot_location[0,:],landmarks,measurements))
        print(calc_likelyhood_measurements_given_position(plot_location[1,:],landmarks,measurements))
        for i in range(np.shape(plot_location)[0]):
            ax.scatter(plot_location[i,0],plot_location[i,1],calc_likelyhood_measurements_given_position(plot_location[i,:],landmarks,measurements),c='b',marker='o',s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('likelihood')
    plt.show()
    return 

landmarks = np.array([[12.0,4.0],
                      [ 5.0,7.0]])
measurements = np.array([[3.9,0.0,1.0],
                         [4.5,0.0,np.sqrt(1.5)]])
location = np.array([[10.0,8.0],
                     [ 6.0,3.0]])
plot_vicinity = np.array([[0,0],
                          [17,15]])
step = 0.5
plot_likelyhood_function(plot_vicinity,step,landmarks,measurements,np.append(location,landmarks,axis=0))


