cmake_minimum_required(VERSION 3.15)
project(volfi VERSION 0.3.1 LANGUAGES CXX)

# Header-only interface library. Consumers: target_link_libraries(mytarget PRIVATE volfi)
# then #include <volfi/volfi_annulus_all.hpp>.
add_library(volfi INTERFACE)
target_include_directories(volfi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(volfi INTERFACE cxx_std_17)

option(VOLFI_BUILD_TESTS "Build the verification checks in reproduce/check" ON)
if(VOLFI_BUILD_TESTS)
  enable_testing()
  # The reproduce/ sources include the engine headers by name, so add include/volfi to the
  # search path; the oracle_*.bin golden vectors load by relative path, so each test runs
  # from reproduce/data. The LBR head-to-head (benchmark_vec, accuracy_vs_lbr) is not wired here
  # because it needs external Let's Be Rational sources -- see reproduce/README.md.
  foreach(chk smoke_test verify_vec warm_test)
    add_executable(${chk} reproduce/check/${chk}.cpp)
    target_link_libraries(${chk} PRIVATE volfi)
    target_include_directories(${chk} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/volfi)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(${chk} PRIVATE -O3 -ffp-contract=off -fno-fast-math)
    endif()
  endforeach()
  add_test(NAME smoke  COMMAND smoke_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reproduce/data)
  add_test(NAME verify COMMAND verify_vec WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reproduce/data)
  add_test(NAME warm   COMMAND warm_test  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reproduce/data)
  # v0.3.0 book kernel: bit-identity of the SIMD twin and accuracy on the 40-digit truth set
  # (wb_vec_gate needs no market feed; wb_gate and the rec gates do, so they are not wired here).
  add_executable(wb_vec_gate reproduce/check/wb_vec_gate.cpp)
  target_link_libraries(wb_vec_gate PRIVATE volfi)
  target_include_directories(wb_vec_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/volfi)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(wb_vec_gate PRIVATE -O3 -ffp-contract=off -fno-fast-math -march=native)
  endif()
  add_test(NAME book COMMAND wb_vec_gate WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/reproduce/data)
endif()
