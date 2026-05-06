CXX ?= clang++
STD ?= -std=c++17
INC := -Iinclude
TEST_FLAGS ?= -O2
BENCH_FLAGS ?= -Ofast -march=native -ffp-contract=fast -fno-math-errno
LBR_INCLUDE ?=
LBR_LIBDIR ?=
LBR_LIBNAME ?= LetsBeRational
LBR_INCFLAGS := $(if $(LBR_INCLUDE),-I$(LBR_INCLUDE),)
LBR_LDFLAGS := $(if $(LBR_LIBDIR),-L$(LBR_LIBDIR),)
LBR_LDLIBS := -l:$(LBR_LIBNAME).so

all: test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_low_delta test_random_true_otm test_fastpatch test_atm bench_otm_grid

test_otm_grid: tests/test_otm_grid.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_true_otm_grid: tests/test_true_otm_grid.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_edge_grid: tests/test_edge_grid.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_delta: tests/test_random_delta.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_v_delta: tests/test_random_v_delta.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_low_delta: tests/test_random_low_delta.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_true_otm: tests/test_random_true_otm.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_fastpatch: tests/test_fastpatch.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_atm: tests/test_atm.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

bench_otm_grid: bench/bench_otm_grid.cpp include/volfi/volfi.hpp include/volfi/volfi_fastpatch.hpp
	$(CXX) $(STD) $(BENCH_FLAGS) $(INC) $< -o $@

bench_lbr_compare: bench/bench_lbr_compare.cpp include/volfi/volfi.hpp
ifeq ($(strip $(LBR_INCLUDE)),)
	$(error LBR_INCLUDE is not set. Example: make bench_lbr_compare LBR_INCLUDE=/path/to/LetsBeRational LBR_LIBDIR=/path/to/LetsBeRational/Linux)
endif
ifeq ($(strip $(LBR_LIBDIR)),)
	$(error LBR_LIBDIR is not set. Example: make bench_lbr_compare LBR_INCLUDE=/path/to/LetsBeRational LBR_LIBDIR=/path/to/LetsBeRational/Linux)
endif
	$(CXX) $(STD) $(BENCH_FLAGS) $(INC) $(LBR_INCFLAGS) $< $(LBR_LDFLAGS) $(LBR_LDLIBS) -Wl,-rpath,$(LBR_LIBDIR) -o $@

test: test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_low_delta test_random_true_otm test_fastpatch test_atm
	./test_otm_grid
	./test_true_otm_grid
	./test_edge_grid
	./test_random_delta
	./test_random_v_delta
	./test_random_low_delta
	./test_random_true_otm
	./test_fastpatch
	./test_atm

bench: bench_otm_grid
	./bench_otm_grid

clean:
	rm -f test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_low_delta test_random_true_otm test_fastpatch test_atm bench_otm_grid bench_lbr_compare
