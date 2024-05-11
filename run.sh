#!/bin/bash
# Run 6 copies of the comp_kernel program on a user supplied device
export CUDA_VISIBLE_DEVICES=$1
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null 
