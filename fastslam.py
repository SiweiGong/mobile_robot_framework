"""
The basic framework of the FastSLAM is provided by Albert-Ludwigs-Universität Freiburg,
the course Introduction to Mobile Robotics (engl.) - Autonomous Mobile Systems
Lecturer: Prof. Dr. Wolfram Burgard, Dr. Michael Tangermann, Dr. Daniel Büscher, Lukas Luft
Co-organizers: Marina Kollmitz, Iman Nematollahi
"""
from read_data import read_world, read_sensor_data
from misc_tools import *
from odometry_based_motion_model import odometry_based_motion_estimate
import numpy as np
import math
import copy
import scipy.stats

def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    for particle in particles:
        particle['history'].append([particle['x'],particle['y'],particle['theta']])
        particle['x'], particle['y'], particle['theta'] = odometry_based_motion_estimate(np.array([particle['x'],particle['y'],particle['theta']]),np.array([delta_rot1,delta_rot2,delta_trans]),noise)
        

def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.1, 0],\
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']
    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']
        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                lx = particle['x'] + meas_range*np.cos(particle['theta']+meas_bearing)
                ly = particle['y'] + meas_range*np.sin(particle['theta']+meas_bearing)
                landmark['mu'] = [lx,ly]
                h, meas_Jacob = measurement_model(particle,landmark)
                meas_Jacob_inv = np.linalg.inv(meas_Jacob)
                landmark['sigma'] = meas_Jacob_inv @ Q_t @ meas_Jacob_inv.T
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                estimated_meas, meas_Jacob = measurement_model(particle,landmark)
                delta = np.array([meas_range - estimated_meas[0],angle_diff(meas_bearing, estimated_meas[1])])
        
                # Kalman gain
                Q = meas_Jacob@landmark['sigma']@meas_Jacob.T+Q_t
                K = landmark['sigma']@ meas_Jacob.T @ np.linalg.inv(Q)
                
                # EKF-Update
                landmark['mu'] = landmark['mu'] + K @ delta
                landmark['sigma'] = (np.eye(np.shape(landmark['sigma'])[0])-K @ meas_Jacob)@landmark['sigma']

                # update weight
                weight = scipy.stats.multivariate_normal.pdf(delta,mean=np.array([0,0]),cov=Q)
                particle['weight'] = particle['weight'] * weight
        
    #normalize weights
    normalizer = sum([p['weight'] for p in particles])
    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

    return

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    # generate cumulative density function
    cdf = np.empty((len(particles),))
    cdf[0] = particles[0]['weight']
    for cdf_indx in range(len(particles)-1):
        cdf[cdf_indx+1] = cdf[cdf_indx] + particles[cdf_indx]['weight']
    thresholds = []
    rand = np.random.uniform(0,1.0/len(particles)/3)
    for p in range(len(particles)):
        thresholds.append([p/len(particles)+rand])
    i = 0
    for j in range(len(particles)):
        while(thresholds[j] > cdf[i]):
            i += 1
        new_particle = copy.deepcopy(particles[i])
        new_particle['weight'] = 1.0/len(particles)
        new_particles.append(new_particle)
    
    return new_particles

def main():

    print ("Reading landmark positions")
    landmarks = read_world("src/fastSLAM_data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("src/fastSLAM_data/sensor_data.dat")

    #plot preferences, interactive plotting mode
    plt.axis([-1, 12, 0, 10])
    plt.ion()
    plt.show()

    num_particles = 100
    num_landmarks = len(landmarks)

    #create particle set
    particles = initialize_particles(num_particles, num_landmarks)

    #run FastSLAM
    for timestep in range(int(len(sensor_readings)/2)):

        #predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

        #plot filter state
        plot_state(particles, landmarks)

        #calculate new set of equally weighted particles
        particles = resample_particles(particles)

    plt.show('hold')

if __name__ == "__main__":
    main()