# General utility library
import numpy as np



##===============================================##
##      Basic functions to manipulate data       ##
##===============================================##

##  function to average data    ##
def average(x: np.ndarray, axis: int = 0, N: int = 10, nan = False):

    """
    average in either frequency or time

    Parameters
    ----------
    x: ndarray
       data to average over
    axis: int 
        axis to average over
    N: int
        Averaging/donwsampling factor
    nan : bool, optional
        If True, using nanmean to ignore NaN values in array 'x', by default False

    Returns
    -------
    x: ndarray 
       Averaged data
    
    """

    # if nan if true, will use numpy function that ignores nans in array x
    if nan:
        func = np.nanmean
    else:
        func = np.mean


    if N == 1:
        return x
    

    # either dynamic spectra or time series
    ndims = x.ndim
    if ndims == 1:
        N_new = int(x.size / N) * N
        return func(x[:N_new].reshape(int(N_new / N), N),axis = 1).flatten()
    
    elif ndims == 2:
        if axis == 0:
            #frequency scrunching
            N_new = int(x.shape[0] / N) * N
            return func(x[:N_new].T.reshape(x.shape[1],int(N_new / N), N),axis = 2).T
        
        elif axis == 1 or axis == -1:
            #time scrunching
            N_new = int(x.shape[1] / N) * N
            return func(x[:,:N_new].reshape(x.shape[0],int(N_new / N), N),axis = 2)
        
        else:
            print("axis must be 1[-1] or 0")
            return x
        
    else:
        print("ndims must equal 1 or 2..")
        return x