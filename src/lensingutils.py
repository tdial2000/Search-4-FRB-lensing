##################################################
# Author:   Tyson Dial                           #
# Email:    tdial@swin.edu.au                    #
# Date (created):     20/02/2025                 #
# Date (updated):     20/02/2025                 #
##################################################
#                                                #
#                                                #
# Search for lensing in FRB voltage data         #
##################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, next_fast_len, ifft
import argparse
from utils import average
from math import ceil
import sys, warnings, os
from make_dynspec import make_ds, baseline_correction, flag_chan
import multiprocessing as mp 
import time


class _empty:
    pass



#############################################

# Utility functions

#############################################

def get_sort_index(arr):

    return sorted(range(len(arr)), key = lambda k: arr[k])

def progressBar(Niter, iteration, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        Niter       - Required  : Number of iterations
        iteration   - Required  : Iteration number (progress)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(Niter)))
        filledLength = int(length * iteration // Niter)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

    # Call
    printProgressBar(iteration)



def get_optimal_fft_width(nsamp, nFFT):
    """
    Get optimal width of 1D crop that will enable the cooley-turkey algorithm after channelisation
    """

    nchan = nsamp // nFFT
    nchan = next_fast_len(nchan)

    return nchan * nFFT



def get_freqs(cfreq, bw, nchan):
    df = bw / nchan
    return np.linspace(cfreq + bw/2 - df/2, cfreq - bw/2 + df/2, nchan)


def make_complex_ds(x, nFFT = 336):
    """
    Info:
        Make Stokes Dynamic spectra with specified stft length 

    Args:
        x: time series polarisation 
        nFFT: number of freq channels

    Returns:
        ds (ndarray): Complex Dynamic spectrum

    """

    # pre-processing for iterative 
    BLOCK_SIZE = 200e6 # block size in B
    BIT_SIZE = 8       # Bit size in B

    # First need to chop data so that an integer number of FFT windows can
    # be performed. Afterwards, this data will be split into coarse BLOCKS
    # with specific memory constraints. 

    # define parameters
    nsamps  = x.size                  # original number of samples in loaded dataset
    fnsamps = (nsamps // nFFT) * nFFT    # number of samples after chopping 
    nwind   = fnsamps // nFFT            # number of fft windows along time series

    # memeory block paramters
    nwinb = int(BLOCK_SIZE // (nFFT * BIT_SIZE))    # num windows in BLOCK
    nsinb = int(nwinb * nFFT)                       # num samples in BLOCK
    nblock= int(nwind // nwinb)                     # num BLOCKS

    # create empty array
    ds = np.zeros((nFFT, nwind), dtype = x.dtype)

    b_arr = np.empty((0,2), dtype = int)
    i = -1
    for i in range(nblock):
        b_arr = np.append(b_arr, [[i*nwinb,(i+1)*nwinb]], axis = 0)
    # append extra block at end
    if nblock*nsinb < nsamps:
        b_arr = np.append(b_arr, [[(i+1)*nwinb,nwind]], axis = 0)

    # need to add normalising constant to fft (1/sqrt(N))

    # loop over blocks
    for i, b in enumerate(b_arr): # b is bounds of block in nFFT windows
        sb = b * nFFT
        wind_w = b[1] - b[0]
        ds[:,b[0]:b[1]] = fft(x[sb[0]:sb[1]].copy().reshape(wind_w, nFFT), axis = 1, norm = "ortho").T
       

    return ds

def apply_gaussian_smooth(y, stDev: int = 3):
    """
    Apply a gaussian as a smoothing function to [y]

    Parameters
    ----------
    y : np.ndarray or array-like
        Data to smooth
    stDev : int
        Standard deviation of gaussian in number of samples
    """

    x = np.linspace(-6*stDev, 6*stDev, 12*stDev + 1)
    gaussian = np.exp(-x**2/(2*stDev**2))

    return np.convolve(y, gaussian/np.sum(gaussian), mode = "same")




# def get_flaggedchans(chanflag, nFFT: int = 336):
#     """
#     Get channels to flag from str

#     Parameters
#     ----------
#     chanflag : str
#         channels to flag (indicies)


#     """

#     chans2flag = chanflag.split(',')
#     flaggedchans = []
#     for i, chans in enumerate(chans2flag):
#         if ":" in chans:
#             lchan = int(float(chans.split(':')[0]) * (nFFT-1))
#             rchan = int(float(chans.split(':')[1]) * (nFFT-1))
#             flaggedchans += list(range(lchan,rchan+1))
        
#         else:
#             chan = int(float(chans) * (nFFT-1))
#             flaggedchans += [chan]

    
#     return np.abs(np.array(flaggedchans) - nFFT + 1)




def phasor_DM(f, DM: float, f0: float):
    """
    Calculate Phasor Rotator for DM dispersion

    Parameters
    ----------
    f : np.ndarray
        Frequency array [MHz]
    DM : float
        Dispersion Measure [pc/cm^3]
    f0 : float
        Reference Frequency [MHz]

    Returns
    -------
    phasor_DM : np.ndarray
        Phasor Rotator array in frequency domain
    """
    # constants
    kDM = 4.14938e3         # DM constant

    return np.exp(2j*np.pi*kDM*DM*(f-f0)**2/(f*f0**2)*1e6)



def coherent_desperse(t, cfreq, bw, f0, DM, fast = False, 
                      DM_iter = 50):
    """
    Apply Coherent dedespersion on Complex Polarisation time series data

    Parameters
    ----------
    t : np.mmap or np.ndarray
        Complex Polarisation time series data
    cfreq : float
        Central Frequency of observing band [MHz]
    bw : float
        Bandwidth of observation [MHz]
    f0 : float
        Reference Frequency
    DM : float
        Dispersion Measure [pc/cm^3]
    fast : bool, optional
        Apply FFT and IFFT quickly by zero-padding data to optimal length. , by default False \n
        Note: This shouldn't affect results too much assuming full CELEBI HTR data, however, the longer 
        the padding relative to the original size of dataset, the worse the data, so use wisely.
    DM_iter : int, optional
        Number of iterations to split Dispersion into, by default 50

    Returns
    -------
    t_d : np.ndarray
        De-dispersed Complex Polarisation times series data
    """

    # constants
    kDM = 4.14938e3         # DM constant

    next_len = t.size
    if fast:
        next_len = next_fast_len(next_len)

    # ifft
    t_d = fft(t, next_len, norm = "ortho")

    # apply dispersion
    uband = cfreq + bw/2                            # upper band
    bw = bw*(next_len/t.size)                       # updated bandwidth due to zero padding in time domain

    BLOCK_size = int(next_len / DM_iter)            # number of samples per DM iteration
    BLOCK_end = next_len - BLOCK_size * DM_iter     # in case number of samples don't divide into DM_iter evenly, usually
                                                    # a handful of samples are left at the end, this is BLOCK_end
    BLOCK_bw = float(bw*BLOCK_size/next_len)        # amount of bandwidth being covered per DM iteration

    # iterate over chuncks to save memory
    for i in range(DM_iter):
        freqs = (np.linspace(uband - i*BLOCK_bw, uband - (i+1)*BLOCK_bw, BLOCK_size, endpoint = False) 
                +bw/next_len/2)

        # disperse part of t series
        t_d[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs, DM, f0)

    # apply last chunck if needed
    if BLOCK_end > 0:
        freqs = np.linspace(uband - (i+1)*BLOCK_bw,uband - bw, BLOCK_end, endpoint = False) +bw/next_len/2

        # disperse
        t_d[(i+1)*BLOCK_size:] *= phasor_DM(freqs, DM, f0)

    # inverse fourier tranform back to time domain
    t_d = ifft(t_d, next_len, norm = "ortho")[:t.size]

    return t_d





def auto_channel_flag(dsI, flag_thresh, tN):
    """
    Use off pulse crop to do auto channel flagging
    """

    # create boolean array
    chanmask = np.arange(0, dsI.shape[0]).astype(int)

    ds_avg = average(dsI, axis = 1, N = tN, nan = True)
    f_std = np.nanstd(ds_avg, axis = 1)
    med_rms = np.nanmedian(f_std)
    mad_rms = 1.48 * np.nanmedian(np.abs(f_std - med_rms))

    chan2flag = np.where(f_std > (med_rms + flag_thresh*mad_rms))[0]
    return chanmask[chan2flag]





def get_best_deltaDM(args, corr_vals, dms):

    idx = get_sort_index(dms)
    dms = np.array(dms)[idx]
    corr_vals = np.array(corr_vals)[idx]

    # get max
    idx_max = np.nanargmax(corr_vals)
    peak_dm = dms[idx_max]     
    return peak_dm, dms, corr_vals


def split_tasks_into_processes(tasks, nproc):
    """
    Split n tasks into m processes

    """
    task_arr = []

    if tasks.size <= nproc:
        print(f"Number of tasks less then or equal to the number of processes (CPUS), Only using {tasks.size} cpus...")
        for i in range(tasks.size):
            task_arr += [np.array([tasks[i]])]
        
        return task_arr

    
    task_per_proc = tasks.size // nproc
    for i in range(nproc):
        task_arr += [np.array(tasks[i*task_per_proc:(i+1)*task_per_proc])]
    
    rem_tasks = tasks.size - task_per_proc * nproc
    for j in range(rem_tasks):
        task_arr[j] = np.append(task_arr[j], tasks[(i+1)*task_per_proc + j])
    
    return task_arr






# create a function to zap channels
def zap_chan(f, zap_str):
    """
    Zap channels, assumes contiguous frequency array

    Parameters
    ----------
    f : np.ndarray
        frequency array used for zapping
    zap_str : str
        string used for zapping channels, in format -> "850, 860, 870:900" \n
        each element seperated by a ',' is a seperate channel. If ':' is used, user can specify a range of values \n
        i.e. 870:900 -> from channel 870 to 900 inclusive of both.

    Returns
    -------
    y : np.ndarray
        zapped indicies in frequency
    
    """

    if zap_str is None:
        return []
    if len(zap_str) == 0:
        return []

    # vals
    df = f[1] - f[0]
    f_min = np.min(f)
    f_max = np.max(f)

    if df < 0:
        # upperside band
        fi = f_max
        df_step = -1

    else:
        # lowerside band
        fi = f_min
        df_step = 1

    df = abs(df)
    

    # split segments
    zap_segments = zap_str.split(',')
    print(zap_str)
    print(zap_segments)
    seg_idx = []

    # for each segment, check for delimiter :, else float cast
    for i, zap_seg in enumerate(zap_segments):

        # if segment is a range of frequencies
        if ":" in zap_seg:
            zap_range = zap_seg.strip().split(':')
            zap_0 = round(df_step * (float(zap_range[0]) - fi)/df)
            zap_1 = round(df_step * (float(zap_range[1]) - fi)/df)

            # check if completely outside bounds
            if (zap_0 < 0 and zap_1 < 0) or (zap_0 > f.size -1 and zap_1 > f.size -1):
                print(f"zap range [{zap_range[0]}, {zap_range[1]}] MHz out of range of bandwidth [{f_min}, {f_max}] MHz")
                continue            
            
            # check bounds
            crop_zap = False

            if zap_0 < 0:
                crop_zap = True
                zap_0 = 0
            elif zap_0 > f.size - 1:
                crop_zap = True
                zap_0 = f.size - 1

            if zap_1 < 0:
                crop_zap = True
                zap_1 = 0
            elif zap_1 > f.size - 1:
                crop_zap = True
                zap_1 = f.size - 1

            if crop_zap:
                print(f"zap range cropped from [{zap_range[0]}, {zap_range[1]}] MHz -> [{f[zap_0]}, {f[zap_1]}] MHz")

            seg_idx += list(range(zap_0,zap_1+(1*df_step),df_step))[::df_step]

        
        # if segment is just a single frequency
        else:
            _idx = round(df_step * (float(zap_seg.strip()) - fi)/df)
            if (_idx < 0) or (_idx > f.size - 1):
                print(f"zap channel {zap_seg.strip()} MHz out of bounds of bandwidth [{f_min}, {f_max}] MHz")
            else:
                seg_idx += [_idx]

    return seg_idx






def get_zapstr(chan, freq):
    """
    Create string of channels to zap based on given nan frequencies in 
    stokes I dynamic spectrum

    Parameters
    ----------
    chan : np.ndarray or array-like
        Stokes I freq array
    freq : np.ndarray or array-like
        Array of frequency values in [MHz]

    Returns
    -------
    zap_str: str
        string of frequencies to zap using zap_chan function
    
    """

    # could be improved later with a smarter algorithm, but not nessesary for ilex.

    if freq[0] > freq[1]:
        chan = chan[::-1]
        freq = freq[::-1]

    zap_str = ""

    chan2zap = np.argwhere(np.isnan(chan)).flatten()
    
    i = 0
    while i < chan2zap.size:
        j = 0
        while i + j + 1 < chan2zap.size:
            if chan2zap[i + 1 + j] - chan2zap[i + j] == 1:
                j += 1
            else:
                break

        if j > 3:
            zap_str += "," + str(freq[chan2zap[i]]) + ":" + str(freq[chan2zap[i + j]])
        else:
            for k in range(j+1):
                zap_str += f",{freq[chan2zap[i+k]]}"
        
        i += j + 1

        
    if zap_str != "":
        zap_str = zap_str[1:]   # remove ','

    if freq[0] > freq[1]:
        chan = chan[::-1]
        freq = freq[::-1]
    
    return zap_str






def combine_zapchan(chan1, chan2):
    """
    Combine two zapchan strings together without duplication. If chan1 is NoneType, return chan2

    Parameters
    ----------
    chan1 : str
        String of current channels
    chan2 : str
        String of channels to add

    Returns
    -------

    zapstr: str
        combined String of channels to zap

    """

    if chan1 is None:
        if chan2 is None:
            return ""
        else:
            return chan2
    
    if chan2 is None:
        if chan1 is None:
            return ""
        else:
            return chan1
            

    if len(chan2) == 0:
        return chan1

    chan1_list = chan1.split(',')
    chan2_list = chan2.split(',')

    zapstr = chan1

    for i, chan in enumerate(chan2_list):
        if chan not in chan1_list:
            if zapstr == "":
                zapstr += f"{chan}"
            else:
                zapstr += f",{chan}"
            
    return zapstr