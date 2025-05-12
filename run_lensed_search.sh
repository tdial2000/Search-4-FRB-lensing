#!/bin/bash

#SBATCH --job-name=correlated_lensed_search
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=64

# parameters
sig=0.15
dm=0.1
x="/fred/oz002/tdial/frb_lensing/testing/sig${sig}_dm${dm}_X.npy"
y="/fred/oz002/tdial/frb_lensing/testing/sig${sig}_dm${dm}_Y.npy"
dsI="/fred/oz002/tdial/frb_lensing/testing/sig${sig}_dm${dm}_I.npy"

t=8
nFFT=1680
tN=20

DMmin=0
DMmax=0.2
DMstep=0.00002

rms_w=5
rms_g=1
stDev=2
Wsigma=2

o="/fred/oz002/tdial/frb_lensing/testing/"
n="sig${sig}_dm${dm}_del${DMstep}_nFFT${nFFT}"

source ~/vpython/bin/activate
cd /fred/oz002/tdial/frb_lensing/src/

python3 search4frblensing.py -x $x -y $y --nFFT $nFFT -t $t --cfreq 919.5 --bw 336 --DMmin $DMmin --DMmax $DMmax --DMstep $DMstep --rms_w $rms_w --rms_g $rms_g --stDev $stDev --Wsigma $Wsigma --tN $tN --dsI $dsI -o $o -n $n --cpus 64 --showplots