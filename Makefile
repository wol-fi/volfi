CXX ?= clang++
STD ?= -std=c++17
INC := -Iinclude
LBR_INC := -Ithird_party/LetsBeRational
LBR_DEF := -DNO_XL_API
LBR_SRC := third_party/LetsBeRational/lets_be_rational.cpp third_party/LetsBeRational/normaldistribution.cpp third_party/LetsBeRational/rationalcubic.cpp third_party/LetsBeRational/erf_cody.cpp
TEST_FLAGS ?= -O2
BENCH_FLAGS ?= -Ofast -march=native -ffp-contract=fast -fno-math-errno

all: test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_true_otm test_atm bench_otm_grid bench_lbr_compare

test_otm_grid: tests/test_otm_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_true_otm_grid: tests/test_true_otm_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_edge_grid: tests/test_edge_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_delta: tests/test_random_delta.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_v_delta: tests/test_random_v_delta.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_true_otm: tests/test_random_true_otm.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_atm: tests/test_atm.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

bench_otm_grid: bench/bench_otm_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(BENCH_FLAGS) $(INC) $< -o $@

bench_lbr_compare: bench/bench_lbr_compare.cpp include/volfi/volfi.hpp $(LBR_SRC)
	$(CXX) $(STD) $(BENCH_FLAGS) $(INC) $(LBR_INC) $(LBR_DEF) $< $(LBR_SRC) -o $@

test: test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_true_otm test_atm
	./test_otm_grid
	./test_true_otm_grid
	./test_edge_grid
	./test_random_delta
	./test_random_v_delta
	./test_random_true_otm
	./test_atm

bench: bench_otm_grid bench_lbr_compare
	./bench_otm_grid
	./bench_lbr_compare

clean:
	rm -f test_otm_grid test_true_otm_grid test_edge_grid test_random_delta test_random_v_delta test_random_true_otm test_atm bench_otm_grid bench_lbr_compare
