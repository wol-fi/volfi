CXX ?= clang++
STD ?= -std=c++17
INC := -Iinclude
TEST_FLAGS ?= -O2
BENCH_FLAGS ?= -Ofast -march=native -ffp-contract=fast -fno-math-errno

all: test_otm_grid test_edge_grid test_random_delta test_random_v_delta test_atm bench_otm_grid

test_otm_grid: tests/test_otm_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_edge_grid: tests/test_edge_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_delta: tests/test_random_delta.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_random_v_delta: tests/test_random_v_delta.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

test_atm: tests/test_atm.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(TEST_FLAGS) $(INC) $< -o $@

bench_otm_grid: bench/bench_otm_grid.cpp include/volfi/volfi.hpp
	$(CXX) $(STD) $(BENCH_FLAGS) $(INC) $< -o $@

test: test_otm_grid test_edge_grid test_random_delta test_random_v_delta test_atm
	./test_otm_grid
	./test_edge_grid
	./test_random_delta
	./test_random_v_delta
	./test_atm

bench: bench_otm_grid
	./bench_otm_grid

clean:
	rm -f test_otm_grid test_edge_grid test_random_delta test_random_v_delta test_atm bench_otm_grid
