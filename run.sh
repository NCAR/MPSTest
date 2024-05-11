#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null &
./comp_kernel.exe $2 > /dev/null 
