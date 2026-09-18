#!/bin/bash
# Runs inside the CUDA devel container; /w is the mounted volfi tree.
# Invoked as:  docker run --rm --gpus all -v $HOME/volfi:/w \
#                nvidia/cuda:12.4.1-devel-ubuntu22.04 bash /w/gpu/run_gpu.sh
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd /w/gpu

echo "=== toolchain / device ==="
nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current \
           --format=csv,noheader

echo
echo "=== build (sm_90, --fmad=false is required: device analogue of -ffp-contract=off) ==="
nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false -Xptxas -v -I../include/volfi volfi_gpu_book.cu -o volfi_gpu_book 2> build.log
echo "=== host self-check of the device mirrors (no GPU needed) ==="
nvcc -O3 -std=c++17 --fmad=false -DVGB_HOST_CHECK=1 -x c++ -I../include/volfi volfi_gpu_book.cu -o vgb_hostcheck && ./vgb_hostcheck
echo "BUILD OK"
echo "--- occupancy gate: wing/left/right MUST be 0 bytes stack frame ---"
grep -E "Compiling entry|stack frame|Used .* registers" build.log
grep -iE "error" build.log | head && echo "(build errors above)" || true

echo
echo "########## RUN 1 (5,000,000 quotes x 300 iters) ##########"
./volfi_gpu_book 5000000 300

echo
echo "########## RUN 2 (repeat; the two must agree to ~3 digits) ##########"
./volfi_gpu_book 5000000 300
