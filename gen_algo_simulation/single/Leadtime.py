from numba import njit
import pandas as pd
import numpy as np
import numba 

def leadtime_generated_params():
    L_x = np.array([3,4,5])
    L_pmf = np.array([0.1,0.7,0.2])
    L_mu, L_std = attributes_numba(L_pmf, L_x)
    L_median = 4
    L_max = 5
  
    return L_x, L_pmf, L_mu, L_std, L_median, L_max


@njit
def attributes_numba(pmf: np.ndarray, x: np.ndarray) -> tuple:
    """
    Returns the mean and standard deviation of a distribution.

    Parameters
    ----------
    pmf : distribution
        The distribution as a pmf (probability mass function).
    x : array
        The values for which the distribution is defined.

    Returns
    -------
    mean : float
        The mean of the distribution.
    std : float
        The standard deviation of the distribution.

    """
    mean = np.sum(pmf * x)
    std = np.sqrt(np.sum(x ** 2 * pmf) - np.sum(pmf * x) ** 2)
    return mean, std




@numba.jit(nopython=True)
def compute_leadtime_data(L_x_orig):
    """
    The function will find the probability mass function of the lead time data and return the mean, standard deviation, median and max of the lead time data.
    """
    # Create a new L_x array that includes all integers within the range of L_x_orig
    L_x = np.arange(np.min(L_x_orig), np.max(L_x_orig) + 1)

    bins = np.arange(int(np.min(L_x_orig)), int(np.max(L_x_orig)) + 2)
    hist, bin_edges = np.histogram(L_x_orig, bins=bins)

    # Create a new L_pmf array that is the same length as L_x. For the bins not present in hist, set the value to 0
    L_pmf = np.zeros_like(L_x, dtype=np.float64)
    L_pmf[:len(hist)] = hist

    # Normalize the histogram manually since density argument is not supported in Numba
    L_pmf = L_pmf / L_pmf.sum()

    bin_mids = bin_edges[:-1] + np.diff(bin_edges)/2
    L_mu, L_std = attributes_numba(L_pmf, bin_mids)
    
    L_median = np.median(L_x_orig)
    L_max = np.max(L_x_orig)
    
    return L_x, L_pmf, L_mu, L_std, L_median, L_max


def get_leadtime_from_df(lead_time_df_path:str, colname_target:str):
    df = pd.read_csv(lead_time_df_path)
    L_x_orig = np.array(df[colname_target])
    L_x, L_pmf, L_mu, L_std, L_median, L_max = compute_leadtime_data(L_x_orig)

    return L_x, L_pmf, L_mu, L_std, L_median, L_max

@njit
def get_leadtime_from_array(lead_time_array:np.ndarray):
    
    L_x_orig = np.array(lead_time_array)
    L_x, L_pmf, L_mu, L_std, L_median, L_max = compute_leadtime_data(L_x_orig)

    return L_x, L_pmf, L_mu, L_std, L_median, L_max
