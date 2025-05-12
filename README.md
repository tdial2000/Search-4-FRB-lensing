# Searching for FRB lensing

This repo provides the tools nessesary to search for potential lensing in FRB baseband data. The code utilises a channelized ACF algorithm to
find correlations in the complex voltage data. 

## Data Format

The expected data format is a set of `X` and `Y` complex time-series datasets. However, since the `X` and `Y` products are simply added together
after the channelized ACF is performed, simply having one polarisation should work. Note: This has not been tested.

## Code requirements

This code requires python version >=3.9 along with the following packages:
1. Numpy
2. Matplotlib
3. Scipy
4. multiprocessing
5. ILEX (to be removed in later version) [https://github.com/tdial2000/ILEX]


## How to use

. To Do


##
