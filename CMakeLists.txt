cmake_minimum_required(VERSION 3.12)
project(fastcheb)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp")

set(COOLPROP_STATIC_LIBRARY true CACHE INTERNAL "Build CoolProp as a static library")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/CoolProp")
# set_property(TARGET CoolProp PROPERTY POSITION_INDEPENDENT_CODE ON)

set(CHEBTOOLS_STATIC_LIBRARY true CACHE INTERNAL "Build ChebTools as a static library")
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools")
# set_property(TARGET ChebTools PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(fastcheb fastcheb.cpp)
target_link_libraries(fastcheb teqpinterface ChebTools CoolProp)

target_include_directories(fastcheb
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/teqp/externals/autodiff"
#    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/ChebTools/externals/Eigen"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals"
)
add_definitions(-DTEQP_MULTIPRECISION_ENABLED)

# add_executable(timeREFPROP timeREFPROP.cpp)