# MPSTest
A test program to determine if NVIDIA MPS is enabled and functioning on GPU compute nodes. Includes a test program and shell scripts to drive it, a simple build script, and sample PBS batch scripts to demonstrate behavior with MPS on/off. Currently targeted at the Derecho A100 GPU nodes.

## Build 
Use the included script to load a CUDA environment and build the executable
```
> ./build.sh
Build complete
```

## Run
Submit to run on a single GPU node to determine if MPS is enabled and functioning on that node. Two example batch scripts are included:
1. `batch_mps_on.sh` to demonstrate test output when MPS is working
   <br>Example output
   ```
   Testing MPS on host deg0036
   Found 4 GPU devices as expected
   MPS was requested in job script: TRUE
   Proceeding with individual GPU device tests
     -- Report for GPU 0 (GPU-6fc66dd4-e01b-f5f7-bac0-c7c12c97277a)
        MPS server found: TRUE
        MPS test status: PASS
     -- Report for GPU 1 (GPU-84100388-688c-ed28-01ff-df9b1dae79d9)
        MPS server found: TRUE
        MPS test status: PASS
     -- Report for GPU 2 (GPU-45228571-dfef-894b-a76d-4600b29a3c50)
        MPS server found: TRUE
        MPS test status: PASS
     -- Report for GPU 3 (GPU-3634c579-2862-f11c-fadc-5157b954d49f)
        MPS server found: TRUE
        MPS test status: PASS
   ```
2. `batch_mps_off.sh` to demonstrate test output when MPS is disabled
   <br>Example output
   ```
   Testing MPS on host deg0034
   Found 4 GPU devices as expected
   MPS was requested in job script: FALSE
   Proceeding with individual GPU device tests
     -- Report for GPU 0 (GPU-3f500276-5883-11c3-b677-e4cb9cfba732)
        MPS server found: FALSE
        MPS test status: FAIL
     -- Report for GPU 1 (GPU-567682dd-c40d-b8c6-b921-1a5166af81e4)
        MPS server found: FALSE
        MPS test status: FAIL
     -- Report for GPU 2 (GPU-d5b60f1d-c5f6-217c-4b85-a92704d9a535)
        MPS server found: FALSE
        MPS test status: FAIL
     -- Report for GPU 3 (GPU-fd98b3e4-9472-ba5d-cb13-19df847ff06a)
        MPS server found: FALSE
        MPS test status: FAIL
   ```

The test program can also be run from the command line in an interactive sessions via
```
./test_mps.sh
```

