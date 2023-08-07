cmake_minimum_required(VERSION 3.12)
project(fastcheb)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp")

set(CHEBTOOLS_STATIC_LIBRARY true CACHE INTERNAL "Build ChebTools as a static library")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools")

add_executable(fitcheb fastcheb.cpp main.cpp)
target_link_libraries(fitcheb teqpinterface ChebTools teqpcpp)
target_include_directories(fitcheb
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp/externals/autodiff"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)
target_compile_definitions(fitcheb PUBLIC -DTEQP_MULTIPRECISION_ENABLED)
target_compile_definitions(fitcheb PUBLIC -DTEQP_MULTICOMPLEX_ENABLED)

add_executable(bench bench.cpp)
target_link_libraries(bench teqpinterface ChebTools teqpcpp Catch2WithMain)
target_include_directories(bench
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp/externals/autodiff"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)
target_compile_definitions(bench PUBLIC -DTEQP_MULTIPRECISION_ENABLED)
target_compile_definitions(bench PUBLIC -DTEQP_MULTICOMPLEX_ENABLED)
