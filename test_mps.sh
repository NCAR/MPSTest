#!/bin/bash
# A test driver with three main tasks
#  1. Query the environment to see if MPS has been requested
#  2. Query the available GPU devices to see if MPS is configured
#  3. Run multiple instances of a CUDA program and examine the elapsed
#     wallclock time to determine if MPS is functioning properly

### test if the job script requested that MPS be enabled
mps_resource=`echo $PBS_SELECT | egrep -o "mps=1"`
if [ "$mps_resource" == "mps=1" ]; then
	mps_req_stat="TRUE"
else
	mps_req_stat="FALSE"
fi

### Test to make sure we find all 4 expected GPUs
ngpus=`nvidia-smi -q | sed -nr 's/Attached\s+GPUs\s+:\s+([0-9]+)/\1/p'`
if [ "$ngpus" != "4" ]; then
	echo "Did not detect 4 GPU devices. Aborting MPS test, fix this issue first"
	exit
fi

# Report global results
echo "Testing MPS on host $(hostname)"
echo "Found $ngpus GPU devices as expected"
echo "MPS was requested in job script: $mps_req_stat"
echo "Proceeding with individual GPU device tests"
### Test each individual GPU 
for i in `seq 0 $((ngpus - 1))`; do
	# get UUID for sanity check
	dev_uuid=`nvidia-smi -q -i $i | grep UUID | awk '{print $4}'`
	
	# Test to see if this GPU expects MPS to work
        gpu_has_mps_server=`nvidia-smi -q -i $i | grep -o nvidia-cuda-mps-server`
	if [ "$gpu_has_mps_server" == "nvidia-cuda-mps-server" ]; then
		mps_server_found="TRUE"
	else
		mps_server_found="FALSE"
	fi

	# Run a performance test to see if MPS is actually working
	# expected run time is ~5 sec w/ MPS working, and 30 sec if MPS is broken or disabled
	t_elapsed=$(/usr/bin/time --format="%e" -- ./run.sh $i $dev_uuid 2>&1)
        if (( $(echo "$t_elapsed < 10" |bc -l) )); then
           mps_is_working="PASS"
        else
           mps_is_working="FAIL"
        fi
	echo "  -- Report for GPU $i ($dev_uuid)"
	echo "       MPS server found: $mps_server_found"
        echo "       MPS test status: $mps_is_working"
done
