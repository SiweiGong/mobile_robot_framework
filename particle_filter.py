# The basic framework of the particle filter is provided by Albert-Ludwigs-Universität Freiburg,
# the course Introduction to Mobile Robotics (engl.) - Autonomous Mobile Systems
# Lecturer: Prof. Dr. Wolfram Burgard, Dr. Michael Tangermann, Dr. Daniel Büscher, Lukas Luft
# Co-organizers: Marina Kollmitz, Iman Nematollahi

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from odometry_based_motion_model import odometry_based_motion_estimate
from sensor_model import calc_likelyhood_measurements_given_position

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.002)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (probabilistic motion models slide 27)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    new_particles = []
    
    for particle in particles:
        estimated_pose = odometry_based_motion_estimate(np.array([particle['x'],particle['y'],particle['theta']]),np.array([delta_rot1,delta_rot2,delta_trans]),noise)
        new_particle = dict()
        new_particle['x'],new_particle['y'],new_particle['theta'] = estimated_pose[0],estimated_pose[1],estimated_pose[2]
        new_particles.append(new_particle)
    return new_particles

def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide 33)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2
    mu_r = 0
    err = np.array([mu_r,sigma_r])
    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    error_measurements = np.empty((len(ids),len(err)))
    for i in range(len(ids)):
        error_measurements[i] = err
    weights = []
    for particle in particles:
        involved_landmarks = np.array([landmarks[b] for b in ids])
        weight = calc_likelyhood_measurements_given_position(np.array([particle['x'],particle['y']]),involved_landmarks,ranges,error_measurements)
        weights.append(weight)
    #normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer
    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []
    # generate cumulative density function
    cdf_particles = np.empty((len(weights),))
    cdf_particles[0] = weights[0]
    for i in range(len(weights)-1):
        cdf_particles[i+1] = cdf_particles[i]+weights[i+1]
    thresholds = []
    for p in range(len(weights)):
        thresholds.append([p/len(weights)])
    i = 0
    for j in range(len(weights)):
        while(thresholds[j] > cdf_particles[i]):
            i += 1
        new_particles.append(particles[i])
    
    return new_particles

def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("src/pf_data/world.dat")
    
    print("Reading sensor data")
    sensor_readings = read_sensor_data("src/pf_data/sensor_data.dat")

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #add random seed for generating comparable pseudo random numbers
    np.random.seed(123)

    #plot preferences, interactive plotting mode
    plt.axis([-1, 12, 0, 10])
    plt.ion()
    plt.show()

    #run particle filter
    for timestep in range(int(len(sensor_readings)/2)):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show('hold')

if __name__ == "__main__":
    main()