# Verification of volfi. No external dependencies. Binaries go to out/ and run from data/,
# because every harness loads its truth files by relative path.
#
#   make check       smoke, accuracy + scalar==batch identity, warm start, book kernel, seams (native build)
#   make check-isa   the identity gates on AVX-512, AVX2 and the build without SIMD
#   make smoke | verify | warm | book | seam      one gate
#   make clean
#
# The head-to-head benchmarks need Let's Be Rational and are built by bench/build_all.sh and
# bench/run_cpu_all.sh, see README.md.

CXX   ?= g++
FLAGS ?= -std=c++17 -O3 -ffp-contract=off -fno-fast-math -funroll-loops -w
INC   := -I../include/volfi
ARCH  ?= -march=native
AVX2  := -mavx2 -mfma -mno-avx512f
NOSIMD := -mno-avx2 -mno-avx512f

out:
	mkdir -p out

out/%: check/%.cpp | out
	$(CXX) $(FLAGS) $(ARCH) $(INC) $< -o $@

smoke: out/smoke_test
	cd data && ../out/smoke_test | tail -2
verify: out/verify_vec
	cd data && ../out/verify_vec
warm: out/warm_test
	cd data && ../out/warm_test
book: out/wb_vec_gate
	cd data && ../out/wb_vec_gate
seam: out/seam_gate
	cd data && ../out/seam_gate

check: smoke verify warm book seam

check-isa: | out
	for g in verify_vec wb_vec_gate seam_gate; do \
	  $(CXX) $(FLAGS) -march=native $(INC) check/$$g.cpp -o out/$${g}_512 && \
	  $(CXX) $(FLAGS) $(AVX2)       $(INC) check/$$g.cpp -o out/$${g}_256 && \
	  $(CXX) $(FLAGS) $(NOSIMD)     $(INC) check/$$g.cpp -o out/$${g}_scl || exit 1; \
	  for isa in 512 256 scl; do echo "== $$g $$isa"; (cd data && ../out/$${g}_$$isa | grep -E "IDENTITY|mismatch|contradiction|pts>1e-15") || exit 1; done; \
	done

clean:
	rm -rf out

.PHONY: smoke verify warm book seam check check-isa clean
