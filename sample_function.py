import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import timeit

def sample_normal_distribution_sum_uniforms(mean,std):
    sum = np.sum(np.random.uniform(-1,1,12))/2
    std_normal_distribution_number = mean+std*(sum/2)
    return mean+std*std_normal_distribution_number

def sample_normal_distribution_rejection(mean,std):
    interval = 5*std
    norm_object = scipy.stats.norm(mean, std)
    maxf = norm_object.pdf(mean)
    while(True):
        x = np.random.uniform(mean-interval,mean+interval)
        y = np.random.uniform(0,maxf)
        if y < norm_object.pdf(x):
            return x

def sample_normal_distribution_Box_Muller(mean,std):
    u = np.random.uniform(0,1,2)
    return np.cos(2*np.pi*u[0])*np.sqrt(-2*np.log(u[1]))

def sample_normal_distribution_numpy(mean,std):
    return np.random.normal(mean,std)

def evaluate_time_per_sample(sample_function,mean,std,num_samples):
    tic = timeit.default_timer()
    y = np.empty((num_samples,),)
    for i in range(num_samples):
        y[i] = sample_function(mean,std)
    toc = timeit.default_timer()
    time_per_sample = (toc - tic) / num_samples * 1e6
    print("%30s : %.3f us" % (sample_function.__name__, time_per_sample))

mean = 0
std = 2
num_samples = 1000
sample_functions = [
    sample_normal_distribution_sum_uniforms,
    sample_normal_distribution_rejection,
    sample_normal_distribution_Box_Muller,
    sample_normal_distribution_numpy
]
# for f in sample_functions:
    # evaluate_time_per_sample(f,mean,std,num_samples)


