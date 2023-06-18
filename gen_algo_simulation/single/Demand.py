import numpy as np
from numba import njit

from random import random

@njit
def choice(probabilities):
    cumulative = 0
    s = np.random.random()
    for i in range(len(probabilities)):
        cumulative += probabilities[i]
        if s < cumulative:
            return i
    return len(probabilities) - 1



@njit
def standard_deviation(data: np.array) -> float:
    data = np.array(data,dtype=np.float64)
    mean_data = 0
    for i in range(len(data)):
        mean_data += data[i]
    mean_data = mean_data / len(data)
    sum_of_squares = np.sum((data - mean_data)**2)
    return np.sqrt(sum_of_squares / (len(data) - 1))



@njit
def attributes(pmf: np.ndarray, x: np.ndarray) -> tuple:
    mean = np.sum(pmf * x)
    std = np.sqrt(np.sum(x**2 * pmf) - np.sum(pmf * x)**2)
    return mean, std


@njit
def min_max(data:np.ndarray):
    min_val = data[0]
    max_val = data[0]
    
    for val in data:
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    return min_val, max_val


@njit
def histogram(data:np.ndarray, bins:np.ndarray):
    min_data, max_data = min_max(data)
    bin_width = (max_data - min_data) / bins
    bin_counters = np.zeros(bins,)
    bin_edges = np.zeros(bins+1)
    # initialize bin counters
    for i in range(bins):
        bin_counters[i] = 0

    # calculating the bin each data point belongs to
    for val in data:
        bin_index = min(int((val - min_data) / bin_width), bins - 1)
        bin_counters[bin_index] += 1

    # defining bin edges
    for i in range(bins+1):
        bin_edges[i] = min_data + i*bin_width
    
    return bin_counters, bin_edges



@njit
def estimate_pmf(x:np.ndarray, bins:int=100):


    hist, bin_edges = histogram(x, bins)   
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    pmf = hist / np.sum(hist)
    return pmf, bin_centers


@njit
def get_range_values(x: np.array) -> tuple:
        bandwidth = standard_deviation(x) / (len(x)**(1/5))
        #find min of x 
        min_x = 0
        for i in range(len(x)):
            if x[i] < min_x:
                min_x = x[i]
        max_x = 0
        for i in range(len(x)):
            if x[i] > max_x:
                max_x = x[i]
        lower = np.floor(min_x - 3 * bandwidth)
        upper = np.ceil(max_x + 3 * bandwidth)
        return lower, upper



@njit
def model_demand(data: np.ndarray, time: int) -> tuple:
    # Calculate bandwidth

    # Get x values for density estimation
    lower, upper = get_range_values(data)
    x_kde = np.linspace(lower, upper, 1000)

    pmf, x_kde = estimate_pmf(data, bins=1000)

    # Compute mean and standard deviation
    demand_mu, demand_std = attributes(pmf, x_kde)

    # Generate demand using the estimated pmf
    indices = np.array([choice(pmf) for _ in range(time)])
    demand = x_kde[indices]

    # Ensure positive demand
    demand = np.maximum(0, demand)
    
    return demand, demand_mu, demand_std, pmf


    

    