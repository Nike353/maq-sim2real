cmake_minimum_required(VERSION 2.8.3)
project(go1_sdk_python_wrapper)


include_directories(include)

#add_compile_options(-std=c++14)

# check arch and os
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
    set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
    set(ARCH arm64)
endif()


link_directories(lib/cpp/${ARCH})


set(EXTRA_LIBS -pthread rt m libunitree_legged_sdk.a)
# set(CMAKE_CXX_FLAGS "-O3 -no-pie")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -fPIC")
set(CMAKE_CXX_STANDARD 14)
# python


add_subdirectory(pybind11)

pybind11_add_module(robot_interface python_interface.cpp)
target_link_libraries(robot_interface ${EXTRA_LIBS} unitree_legged_sdk)
pybind11_add_module(go1_interface go1_interface.cpp)
target_link_libraries(go1_interface ${EXTRA_LIBS} unitree_legged_sdk)

