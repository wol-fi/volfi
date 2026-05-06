cmake_minimum_required(VERSION 3.15)
project(volfi LANGUAGES CXX)

add_library(volfi INTERFACE)
target_include_directories(volfi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(volfi INTERFACE cxx_std_17)

add_executable(test_otm_grid tests/test_otm_grid.cpp)
target_link_libraries(test_otm_grid PRIVATE volfi)

add_executable(test_true_otm_grid tests/test_true_otm_grid.cpp)
target_link_libraries(test_true_otm_grid PRIVATE volfi)

add_executable(test_edge_grid tests/test_edge_grid.cpp)
target_link_libraries(test_edge_grid PRIVATE volfi)

add_executable(test_random_delta tests/test_random_delta.cpp)
target_link_libraries(test_random_delta PRIVATE volfi)

add_executable(test_random_v_delta tests/test_random_v_delta.cpp)
target_link_libraries(test_random_v_delta PRIVATE volfi)

add_executable(test_random_low_delta tests/test_random_low_delta.cpp)
target_link_libraries(test_random_low_delta PRIVATE volfi)

add_executable(test_random_true_otm tests/test_random_true_otm.cpp)
target_link_libraries(test_random_true_otm PRIVATE volfi)

add_executable(test_fastpatch tests/test_fastpatch.cpp)
target_link_libraries(test_fastpatch PRIVATE volfi)

add_executable(test_atm tests/test_atm.cpp)
target_link_libraries(test_atm PRIVATE volfi)

add_executable(bench_otm_grid bench/bench_otm_grid.cpp)
target_link_libraries(bench_otm_grid PRIVATE volfi)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
 target_compile_options(test_otm_grid PRIVATE -O2)
 target_compile_options(test_true_otm_grid PRIVATE -O2)
 target_compile_options(test_edge_grid PRIVATE -O2)
 target_compile_options(test_random_delta PRIVATE -O2)
 target_compile_options(test_random_v_delta PRIVATE -O2)
 target_compile_options(test_random_low_delta PRIVATE -O2)
 target_compile_options(test_random_true_otm PRIVATE -O2)
 target_compile_options(test_fastpatch PRIVATE -O2)
 target_compile_options(test_atm PRIVATE -O2)
 target_compile_options(bench_otm_grid PRIVATE -Ofast -march=native -ffp-contract=fast -fno-math-errno)
endif()

enable_testing()
add_test(NAME otm_grid COMMAND test_otm_grid)
add_test(NAME true_otm_grid COMMAND test_true_otm_grid)
add_test(NAME edge_grid COMMAND test_edge_grid)
add_test(NAME random_delta COMMAND test_random_delta)
add_test(NAME random_v_delta COMMAND test_random_v_delta)
add_test(NAME random_low_delta COMMAND test_random_low_delta)
add_test(NAME random_true_otm COMMAND test_random_true_otm)
add_test(NAME fastpatch COMMAND test_fastpatch)
add_test(NAME atm COMMAND test_atm)

# The LetsBeRational comparison benchmark is intentionally built through the Makefile,
# because it depends on a separately obtained local LetsBeRational build.
