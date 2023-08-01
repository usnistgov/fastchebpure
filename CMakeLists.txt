cmake_minimum_required(VERSION 3.12)
project(fastcheb)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp")

#set(COOLPROP_STATIC_LIBRARY true CACHE INTERNAL "Build CoolProp as a static library")
#add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp")
## set_property(TARGET CoolProp PROPERTY POSITION_INDEPENDENT_CODE ON)

set(CHEBTOOLS_STATIC_LIBRARY true CACHE INTERNAL "Build ChebTools as a static library")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools")
# set_property(TARGET ChebTools PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(checkcheb checkcheb.cpp fastcheb.cpp)
target_link_libraries(checkcheb teqpinterface ChebTools teqpcpp)
target_include_directories(checkcheb
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp/externals/autodiff"
#    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools/externals/Eigen"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)
target_compile_definitions(checkcheb PUBLIC -DTEQP_MULTIPRECISION_ENABLED)
target_compile_definitions(checkcheb PUBLIC -DTEQP_MULTICOMPLEX_ENABLED)

add_executable(bench bench.cpp)
target_link_libraries(bench teqpinterface ChebTools teqpcpp Catch2WithMain)
target_include_directories(bench
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp/externals/autodiff"
#    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools/externals/Eigen"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)
target_compile_definitions(bench PUBLIC -DTEQP_MULTIPRECISION_ENABLED)
target_compile_definitions(bench PUBLIC -DTEQP_MULTICOMPLEX_ENABLED)
