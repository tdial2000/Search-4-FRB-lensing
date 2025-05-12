#!/bin/bash
config=$1

cd /fred/oz002/tdial/frb_lensing/askap

source ~/vpython/bin/activate

python3 /fred/oz002/tdial/frb_lensing/src/search4frblensing.py -c $config 