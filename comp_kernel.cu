#include<iostream>
#include<string>
#include <sstream>
#include <vector>
#include <tuple>
#include <iomanip>
#include<cmath>
#include <cuda_runtime.h>
#include "comp_kernel.h"

__global__ void comp_kernel(int* result, int loop_max){
  int i,j,k,l;
  *result=1;
  for(i=0; i<loop_max;i++){
     for(j=0; j<loop_max; j++){
        for(k=0; k<loop_max; k++){
           for(l=0; l<loop_max; l++){
              *result += *result%std::abs(l-k+j-i);
              if(*result%25 == 0) *result=1;  // reset if divisible by 5
	      if(*result < 0) *result=1;     // this should not happen
	   }
	}
     }
  }
  *result=(*result%137);
}

// utility funtion to convert a device UUID to a string in nvidia-smi format
std::string uuid_to_str(const cudaDeviceProp *dev_prop){
   std::ostringstream uuid_ostr;
   std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
   uuid_ostr << "GPU";
   for (auto t : r){
      uuid_ostr << "-";
      for (int i = std::get<0>(t); i < std::get<1>(t); i++)
         uuid_ostr << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)dev_prop->uuid.bytes[i];
   }
   std::string uuid_str = uuid_ostr.str();
   //char *rstr = new char [uuid_str.length()+1];
   //std::strcpy(rstr, uuid_str.c_str());
   return(uuid_str);
}

int main(int argc, char *argv[]){
  int loop_max = 73;               // loop iterations
  int* h_res;                      // host result, not actually used
  int* d_res;                      // device result, not actually used
  int dev_id=0;                    // Assume this is called with CUDA_VISIBLE_DEVICES set to a single device ID, so "0" is the only visible device
  cudaSetDevice(dev_id);
  cudaDeviceProp *dev_prop = new cudaDeviceProp();
  cudaGetDeviceProperties(dev_prop, dev_id);
  std::string dev_uuid = uuid_to_str(dev_prop); // UUID from device quert
  std::string input_uuid = argv[1];             // expected UUID from command line argument

  // Expected UUID should be supplied as an argument
  if (argc != 2) {
     std::cout << "Usage: ./" << argv[0] << " <UUID-to-compare>" << std::endl;
     return -1;
  }

  // check to make sure we are running on the expected device
  if(dev_uuid == input_uuid){
	  std::cout << "Running on expected CUDA device";
  } else {
	  std::cout << "Error: Not running on expected device: FALSE";
	  return(-137);
  }

  // allocate storage for the (unused) results
  h_res = (int*)malloc(sizeof(int));
  cudaMalloc((void**)&d_res,sizeof(int));
  
  // this only purpose of this kernel is to consume wallclock time on the device
  comp_kernel<<<1,1>>>(d_res,loop_max);
  cudaDeviceSynchronize();
  cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
  // we only use the result to guard against the kernel being optimized out
  std::cout << " -- result = " << *h_res << std::endl;
  return 0;
}
