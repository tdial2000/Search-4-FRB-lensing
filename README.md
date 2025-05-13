# Searching for FRB lensing

This repo provides the tools nessesary to search for potential lensing in FRB baseband data. The code utilises a channelized ACF algorithm to
find correlations in the complex voltage data. See this [Guthub Repo Wiki](https://github.com/tdial2000/Search-4-FRB-lensing/wiki) for a how to guide!

## Data Format

The expected data format is a set of `X` and `Y` complex time-series datasets. However, since the `X` and `Y` products are simply added together
after the channelized ACF is performed, simply having one polarisation should work. Note: This has not been tested.

## Code requirements

This code requires python version >=3.9 along with the following packages:
1. Numpy (For general data manipulation)
2. Matplotlib (For plotting, of course!)
3. Scipy (For signal processing algorithms including autocorrelation and FFT)
4. multiprocessing (Enable multi-processing)

