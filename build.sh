#!/bin/bash

# load a CUDA capable environment
module --force purge
module load ncarenv/23.09 craype/2.7.23 nvhpc/23.7 ncarcompilers/1.0.0 cuda/12.2.1 cray-mpich/8.1.27

# build the test executable
nvcc -o comp_kernel.exe comp_kernel.cu

if [ "$?" == "0" ]; then
	echo "Build complete"
else
	echo "Problem building executable"
fi
