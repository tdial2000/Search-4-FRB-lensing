# imports
import numpy as np 
from scipy.fft import next_fast_len, fft, ifft
import sys




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

    return np.exp(-2j*np.pi*kDM*DM*(f-f0)**2/(f*f0**2)*1e6)





def fft_time_shift(X, bw, cfreq, t, DM, f0):
    """
    Apply a time shift to dataset X via rotation in the frequency domain through FFT

    Parameters
    ----------
    X: 1D np.ndarray
        Dataset 
    bw: float
        Bandwidth (sampling frequency) in MHz
    cfreq: float
        Central frequency in MHz
    t: float
        time shift in milliseconds
    DM: float
        Dedipsersion to apply in pc cm^-3 
    f0: float
        Reference frequency for dedispersion in MHz

    Returns
    -------
    X: 1D np.ndarray
    time shifted Dataset X
    
    """

    # move to frequency domain
    X_shift = fft(X)

    BLOCK_size = int(X.size / 50)            # number of samples per DM iteration
    BLOCK_end = X.size - BLOCK_size * 50     # in case number of samples don't divide into DM_iter evenly, usually
                                                    # a handful of samples are left at the end, this is BLOCK_end
    BLOCK_bw = float(bw*BLOCK_size/X.size)        # amount of bandwidth being covered per iteration

    # iterate over chuncks to save memory
    for i in range(50):
        freqs = (np.linspace(bw/2 - i*BLOCK_bw, bw/2 - (i+1)*BLOCK_bw, BLOCK_size, endpoint = False) 
                +bw/X.size/2)

        X_shift[i*BLOCK_size:(i+1)*BLOCK_size] *= np.exp(2j* np.pi * t * freqs * 1e3)        # constant shift in time

        X_shift[i*BLOCK_size:(i+1)*BLOCK_size] *= phasor_DM(freqs + cfreq, DM, f0)


        print(f"Phase Shifting:    [Progress] = {(i+1)/(50+1)*100:3.3f}%", end = "\r")
    
    if BLOCK_end > 0:
        freqs = np.linspace(bw/2 - (i+1)*BLOCK_bw, bw/2 - bw, BLOCK_end, endpoint = False) + bw/X.size/2

        X_shift[(i+1)*BLOCK_size:] *= np.exp(2j*np.pi*t*freqs*1e3)
    
    return ifft(X_shift)





def make_frb_bbv_white_noise(nsamp: int = 100_000_000):
    """
    Create mock complex voltage baseband dataset with white noise

    Parameters
    ----------
    nsamp: int
        Number of samples in baseband dataset, by default 100_000_000
    """

    # create white noise
    V = (np.random.normal(loc = 0, scale = np.sqrt(2)/2, 
                          size = 2*nsamp)).astype(np.float32).view(np.complex64)
    
    return V




def make_frb_bbv_single_gaussian(amp: float = 1, sig: float = 1, mu: float = 100.0, 
                                nsamp: int = 100_000_000, dt: float = 2.97619e-6):
    """
    Create a mock complex voltage baseband dataset with a single gaussian pulse

    Parameters
    ----------
    amp : float, optional
        amplitude of gaussian pulse, by default 1
    sig : float, optional
        width of gaussian in [milliseconds], by default 1
    mu : float, optional
        position of gaussian in [milliseconds], by default 100.0
    nsamp : int, optional
        Number of samples of baseband dataset, by default 100_000_000
    dt : float, optional
        time resolution of each sample in [milliseconds]
    """

    # create white noise
    V = make_frb_bbv_white_noise(nsamp)
    
    # apply Gaussian envelope function
    V *= amp * np.exp(-(np.linspace(0, nsamp*dt, nsamp)-mu)**2/(2*sig**2))

    return V





def get_gaussian_amp_from_flux_density(flux: float, sig: float, dt: float):
    """
    Assumes power = sum((amp* e^-x^2/2)^2)
    """

    x = np.arange(-6, 6, dt/sig)**2

    amp = np.sqrt(flux)/np.sqrt(dt * np.sum(np.exp(-x)))

    return amp
    

def get_gaussian_peak_from_flux_density(flux: float, sig: float, dt: float):
    """
    Peak of power spectrum
    
    Assumes power = sum(amp * e^-x^2)
    """

    x = np.arange(-6, 6, dt/sig)**2
    
    amp = flux / np.sum(dt * np.exp(-x))

    return amp




def make_bbv_square_wave(amp: float = 1, center: float = 10, width: float = 10,
                        nsamp: int = 100_000_000, dt: float = 2.97619e-6):
    """
    Simulate a complex square wave

    Parameters
    ----------
    amp: float
        Amplitude of square wave
    center: float
        Center of square wave in milliseconds
    width: float
        Width of sqaure wave in milliseconds
    nsamp: int
        Number of samples making simulated voltage data
    dt: float
        Time resolution of each sample in milliseconds
    
    Returns:
    V: np.ndarray or array-like (1D)
        Simulated complex voltage with square wave
    """


    # create empty array
    V = np.zeros(nsamp, dtype = np.complex64)

    # add square wave 
    width_samp = int(width / dt)
    center_samp = int(center / dt)

    V[center_samp - width_samp//2:center_samp + width_samp//2] += amp * make_frb_bbv_white_noise(
                                                            width_samp//2 * 2)
    
    return V



