#!/bin/bash
# Builds and runs the CUDA twins of the book kernel (Section "intrinsic coordinates" of the
# paper): the recurrence kernel, the template-emitted 12-cell table and the whole-book kernel,
# inside the CUDA devel container on an H100 host.  Mirrors gpu/run_gpu.sh (the routed charts).
#
# Invoked as (the volfi tree mounted at /w):
#   docker run --rm --gpus all -v $HOME/volfi:/w \
#     nvidia/cuda:12.4.1-devel-ubuntu22.04 bash /w/gpu/run_near_gpu.sh | tee gpu_near_run.txt
#
# Expects: /w/include/volfi (the headers), /w/gpu (this script, the .cu, the generated .cuh
# files from gpu/make_near_cuda.py), and in /w/gpu the run inputs market_feed.csv (not
# redistributed, see reproduce/README.md) and near_truth.bin (copy from reproduce/data).
#
# Three builds, one per book layout (-DNB_LAYOUT=1,2,3).  L1 is the design: one
# cell over the whole a-range, straight-line, nothing to diverge on.  L2 and L3
# exist to measure what a compile-time cell switch costs on mixed warps.
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd /w/gpu

echo "=== toolchain / device ==="
nvcc --version | tail -1
nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current \
           --format=csv,noheader

# Layouts: L1 is the design and the default.  L2/L3 (the cell-switch cost study) only on
# request: LAYOUTS="1 2 3".  Each layout is two nvcc builds and two device runs (~5-8 min).
for L in ${LAYOUTS:-1}; do
  if [ ! -f volfi_near_book_tables_L${L}_cuda.cuh ]; then
    echo; echo "(layout L$L: no device table, skipped)"; continue
  fi
  echo
  echo "################################################################"
  echo "###  BOOK LAYOUT L$L"
  echo "################################################################"
  echo "=== build (sm_90; --fmad=false is required, it is the device analogue of"
  echo "    -ffp-contract=off, and without it the bit-identity contract is lost) ==="
  nvcc -O3 -arch=sm_90 -std=c++17 --fmad=false -Xptxas -v -DNB_LAYOUT=$L \
       -I../include/volfi volfi_near_certified_gpu.cu -o volfi_near_gpu_L$L 2> build_L$L.log
  echo "BUILD OK"
  echo "--- occupancy gate: EVERY kernel must report 0 bytes stack frame ---"
  grep -E "Compiling entry|stack frame|Used .* registers" build_L$L.log | sed 's/_Z[0-9]*//'
  grep -iE "error" build_L$L.log | head || true

  echo
  echo "=== host self-check first: the device functions compiled for the CPU must"
  echo "    reproduce the CPU reference bit for bit before any device run ==="
  nvcc -O3 -std=c++17 --fmad=false -DNCG_HOST_CHECK=1 -DNB_LAYOUT=$L -x c++ \
       -I../include/volfi volfi_near_certified_gpu.cu -o ncheck_L$L
  ./ncheck_L$L

  echo
  echo "########## DEVICE RUN 1, L$L (5,000,000 quotes x 300 iters) ##########"
  ./volfi_near_gpu_L$L 5000000 300

  echo
  echo "########## DEVICE RUN 2, L$L (repeat; the two must agree to ~3 digits) ##########"
  ./volfi_near_gpu_L$L 5000000 300
done

echo
echo "Reference points, same H100 PCIe host class, from the earlier sessions:"
echo "  shipped v0.2.4: NEAR 0.037, FAR 0.135, UPPER 0.163, WING 0.661 ns/quote"
echo "  12-cell certified chart: self-routing 0.062, bucketed 0.077 ns/quote"
echo "The feed tile's 'book kernel alone' line against shipped NEAR is the like-for-like."
echo
echo "Section [1] must read 'mismatches = 0' in every layout.  If it does not, the timings mean nothing."
