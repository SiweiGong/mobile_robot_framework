# The basic framework of the kalman filter is provided by Albert-Ludwigs-Universität Freiburg,
# the course Introduction to Mobile Robotics (engl.) - Autonomous Mobile Systems
# Lecturer: Prof. Dr. Wolfram Burgard, Dr. Michael Tangermann, Dr. Daniel Büscher, Lukas Luft
# Co-organizers: Marina Kollmitz, Iman Nematollahi

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def odometry_motion_model_mean(odometry, mu):
    mu_est = np.empty(np.shape(mu))
    mu_est[0] = mu[0] + odometry['t']*np.cos(odometry['r1']+mu[2])
    mu_est[1] = mu[1] + odometry['t']*np.sin(odometry['r1']+mu[2])
    mu_est[2] = odometry['r1'] + odometry['r2']+ mu[2]

    # Jacobian matrices 
    Jacobian_location = np.array([[1.0, 0.0, -odometry['t']*np.sin(odometry['r1']+mu[2])],\
                                  [0.0, 1.0, odometry['t']*np.cos(odometry['r1']+mu[2])],\
                                  [0.0, 0.0, 1]])
    Jacobian_control = np.array([[-odometry['t']*np.sin(odometry['r1']+mu[2]), 0.0, np.cos(odometry['r1']+mu[2])],\
                                 [ odometry['t']*np.cos(odometry['r1']+mu[2]), 0.0, np.sin(odometry['r1']+mu[2])],\
                                 [ 1.0, 1.0, 0.0]])     
    return mu_est, Jacobian_location, Jacobian_control

def sensor_model_mean(sensor_data, mu, landmarks, noise_scala):

    estimated_measurements = np.empty(np.shape(sensor_data['range']))
    Jacobian_sensor = np.empty((len(sensor_data['range']),len(mu)))
    noise = np.zeros((len(sensor_data['range']),len(sensor_data['range'])))
    i = 0
    for id in sensor_data['id']:
        estimated_measurements[i] = np.linalg.norm(mu[0:-1]-landmarks[id])
        Jacobian_sensor[i,:] = np.array([(mu[0]-landmarks[id][0])/estimated_measurements[i],\
                                         (mu[1]-landmarks[id][1])/estimated_measurements[i],\
                                         0])
        noise[i,i] = noise_scala
        i += 1
    return estimated_measurements, Jacobian_sensor, noise

def prediction_step(odometry, mu, sigma, noise):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    mu_estimated, Jacob_loca, Jacob_control = odometry_motion_model_mean(odometry, mu)
    sigma_estimated = Jacob_loca @ sigma @ Jacob_loca.T + Jacob_control @ noise @ Jacob_control.T
    return mu_estimated, sigma_estimated

def correction_step(sensor_data, mu, sigma, landmarks, noise):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    estimated_mesurements, Jacob_sensor, noise = sensor_model_mean(sensor_data, mu, landmarks, noise)
    
    # Kalman gain
    K = sigma @ Jacob_sensor.T @ np.linalg.inv(Jacob_sensor @ sigma @Jacob_sensor.T + noise)
    mu = mu + K @ (sensor_data['range'] - estimated_mesurements)
    sigma = (np.identity(np.shape(sigma)[0]) - K @ Jacob_sensor) @ sigma
    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("src/ekf_data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("src/ekf_data/sensor_data.dat")

    # initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    # motion model noise
    Q = np.array([[0.2, 0.0, 0.0],
                  [0.0, 0.2, 0.0],
                  [0.0, 0.0, 0.2]])

    # noise in the sensor model
    R_scala = 0.5

    #plot preferences, interactive plotting mode
    plt.figure()
    plt.axis([-1, 12, 0, 10])
    plt.ion()
    plt.show()

    #run kalman filter
    for timestep in range(int(len(sensor_readings)/2)):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma, Q)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks, R_scala)

    plt.show('hold')

if __name__ == "__main__":
    main()