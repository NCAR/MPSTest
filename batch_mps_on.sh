#!/bin/bash
#PBS -l select=1:ncpus=64:ngpus=4:mps=1
#PBS -l walltime=00:05:00
#PBS -A SCSG0001
#PBS -q main
#PBS -N TestMPS
#PBS -k eod
#PBS -j oe

./test_mps.sh >& test_mps_on.`hostname`.log
