# Standalone verification and timing for the volfi v0.2.0 engine.
# No external dependencies. Run from this directory (oracle_*.bin load by relative path).
# The LBR head-to-head (benchmark_vec / accuracy_vs_lbr) needs Jäckel's sources and is not
# wired here — see BENCHMARK_PROTOCOL.md.

CXX   ?= g++
STD   ?= -std=c++17
ARCH  ?= -march=native
FLAGS ?= -O3 -ffp-contract=off -fno-fast-math
INC   := -I../include/volfi

BIN := smoke vv warm fb

all: $(BIN)

smoke: smoke_test.cpp
	$(CXX) $(STD) $(FLAGS) $(ARCH) $(INC) $< -o $@
vv: verify_vec.cpp
	$(CXX) $(STD) $(FLAGS) $(ARCH) $(INC) $< -o $@
warm: warm_test.cpp
	$(CXX) $(STD) $(FLAGS) $(ARCH) $(INC) $< -o $@
fb: fixed_bench.cpp
	$(CXX) $(STD) $(FLAGS) $(ARCH) $(INC) $< -o $@

# build + run the host-independent checks (accuracy + scalar==batch bit-identity)
check: smoke vv warm
	./smoke | tail -2
	./vv
	./warm

clean:
	rm -f $(BIN) vv_avx2 vv_scalar bench_512 bench_256 mismatch_dump.csv

.PHONY: all check clean
