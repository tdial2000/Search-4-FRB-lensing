# imports

import numpy as np
import argparse
from scipy.fft import fft, ifft, next_fast_len
from simulate_utils import *






def get_args():

    desc = """
        This script will simulate an FRB with full baseband data.

        STEPS to simulate FRB
        1. Simulate White noise
        2. Convolve with Gaussian profile
        3. Create lensed copy (optional)
        4. Apply a delta DM to lensed copy (optional)
    
    """


    parser = argparse.ArgumentParser(description = desc)

    # arguments
    parser.add_argument("--nsamp", help = "Number of samples", type = int, default = 100_000_000)
    parser.add_argument("--dt", help = "Time resolution per sample in [milliseconds]", type = float, default = 2.97619e-6)

    # pulse properties
    parser.add_argument('--peak_power', help = "Flux density of signal in [Jy]", type = float, default = 5.0)
    parser.add_argument('--system', help = "System SEFD in [Jy]", type = float, default = 300.0)
    parser.add_argument('--sig', help = "Pulse width [ms]", type = float, default = 1.0)
    parser.add_argument('--mu', help = "Position of pulse relative to first sample in [ms]", type = float, default = 50.0)
    
    # lensed copy (single copy)
    parser.add_argument("--lensedcopy", help = "Make lensed copy", action = "store_true")
    parser.add_argument("--tshift", help = "Time to shift lensed copy in [milliseconds]", type = float, default = 5.0)
    parser.add_argument("--mag", help = "Magnification of lensed copy", type = float, default = 0.5)
    parser.add_argument("--deltaDM", help = "Delta DM to apply to lensed copy", type = float, default = 0.0000)

    # frequency properties
    parser.add_argument("--cfreq", help = "Central frequency of FRB [MHz]", type = float, default = 919.5)
    parser.add_argument("--bw", help = "Bandwidth of FRB [MHz]", type = float, default = 336.0)

    # outputs
    parser.add_argument('-o', help = "Output filenames <filename>_X.npy and <fileame>_Y.npy", type = str, default = "")
    args = parser.parse_args()

    args.sig *= 2**0.5

    # set nsamp to optmial length to enable Cooley-Turkey algorithm 
    args.nsamp = next_fast_len(args.nsamp)
    print(f"[Simulate FRB] Setting [nsamp] to {args.nsamp} for optimal FFT algorithm...")

    # calculate gaussian amplitude in complex voltage buffer
    # NOTE: added sqrt(2) since power is X**2 + Y**2
    args.amp = np.sqrt(args.peak_power/2)
    print(f"[Simulate FRB] Amplitude of complex gaussian with a peak power of [{args.peak_power}] Jy and width of [{args.sig/np.sqrt(2)}] ms is [{args.amp}]...")


    # calculate antenna noise SEFD in complex X and Y baseband data
    args.system_amp = (args.system/np.sqrt(2))**0.5
    print(f"[Simulate FRB] noise amplitude per sample in complex voltage data given an SEFD of [{args.system}] Jy is [{args.system_amp}]...")


    return args










def simulate_frb(args, pol = "X"):
    """
    Simulate FRB
    
    
    """

    # Create Gaussian pulse
    print(f"[pol {pol}] Creating Gaussian pulse using white noise...")
    V = make_frb_bbv_single_gaussian(amp = args.amp, sig = args.sig, mu = args.mu,
                                     nsamp = args.nsamp, dt = args.dt)
    

    # Create Lensed copy
    if args.lensedcopy:
        print(f"[pol {pol}] Applying phase time shift of [{args.tshift}] ms...")
        print(f"[pol {pol}] Applying a delta DM of {args.deltaDM} pc cm^-3 to lensed copy...")
        f0 = args.cfreq - args.bw/2
        V += args.mag * fft_time_shift(V, args.bw, args.cfreq, args.tshift, args.deltaDM, f0)

    
    # add white noise
    print(f"[pol {pol}] Adding white noise...")
    V += args.system_amp * make_frb_bbv_white_noise(args.nsamp)



    # Apply Delta DM to lensed copy



    return V 










if __name__ == "__main__":
    # main block of code

    args = get_args()

    # get X polarisation
    X = simulate_frb(args, pol = "X")
    print("Saving X pol baseband data...")
    with open(args.o + "_X.npy", 'wb') as file:
        np.save(file, X)
    
    Y = simulate_frb(args, pol = "Y")
    print("Saving Y pol basebad data...")
    with open(args.o + "_Y.npy", 'wb') as file:
        np.save(file, Y)

    print("[simulate_frb.py] Completed!")