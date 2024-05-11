#ifndef _COMP_KERNEL_H_
#define _COMP_KERNEL_H_
#include<string>
#include<cuda_runtime.h>

// utility funtion to convert a device UUID to a string in nvidia-smi format
std::string uuid_to_str(const cudaDeviceProp *dev_prop);

#endif
