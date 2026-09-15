#!/bin/bash
# run_pde_gpu.sh -- the PDE method (Matic/Radoicic/Stefanica, OpenCL) on our market feed, on the
# same H100 and in the same container session as run_near_gpu.sh.  Invoked inside the CUDA devel
# container with the bundle mounted at /w:
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $R:/w \
#     nvidia/cuda:12.4.1-devel-ubuntu22.04 bash /w/gpu/run_pde_gpu.sh
# Expects /w/pde/ = the authors' sources + loadPartition.txt (NOT redistributed: clone
# https://github.com/maticivan/PDE-method-for-implied-volatility, MIT, and fetch its 46 MB table
# as its README says), /w/gpu/market_feed.csv (not redistributed either), and the harnesses
# reproduce/book/pde_compare.cpp and pde_branch.cpp copied to /w/gpu; our headers in /w/include/volfi.
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd /w/pde
echo "=== OpenCL loader + NVIDIA ICD (the CUDA image ships neither) ==="
apt-get update -qq > /dev/null && apt-get install -y -qq ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers clinfo > /dev/null
[ -f /etc/OpenCL/vendors/nvidia.icd ] || { mkdir -p /etc/OpenCL/vendors && echo libnvidia-opencl.so.1 > /etc/OpenCL/vendors/nvidia.icd; }; cat /etc/OpenCL/vendors/nvidia.icd
clinfo -l 2>&1 | head -5
clinfo 2>/dev/null | grep -E "Device Name|Device Version|Double" | head -4

echo "=== build: the authors' objects at their flags (-O2 -fopenmp, as compile_aws), our driver at ours ==="
g++ -c cImpVolBasic.cpp -std=c++11 -fopenmp -O2 -w
g++ -c indexStructure.cpp -std=c++11 -fopenmp -O2 -w
g++ -c GCOpenCL.cpp -std=c++11 -fopenmp -O2 -w
g++ -c gcImpVolBasic.cpp -std=c++11 -fopenmp -O2 -w
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -march=native -fopenmp -w -DPDE_GPU \
    -I/w/include/volfi -I/w/pde -c /w/gpu/pde_compare.cpp -o pde_compare.o
g++ -o pde_compare pde_compare.o GCOpenCL.o cImpVolBasic.o indexStructure.o gcImpVolBasic.o -fopenmp -lOpenCL
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -march=native -fopenmp -w -DPDE_GPU -I/w/include/volfi -I/w/pde -c /w/gpu/pde_branch.cpp -o pde_branch.o
g++ -o pde_branch pde_branch.o GCOpenCL.o cImpVolBasic.o indexStructure.o gcImpVolBasic.o -fopenmp -lOpenCL
echo "BUILD OK"
cp /w/gpu/market_feed.csv .
echo "=== run: 4,980,000-quote tile, 20 iterations ==="
./pde_compare 4980000 20
echo "=== repeat ==="
./pde_compare 4980000 20
echo "=== branch-wise: PDE on region-filtered tiles of 4,980,000 (Near / Far / Wing), 20 iterations ==="
./pde_branch 4980000 20
echo "=== repeat ==="
./pde_branch 4980000 20
