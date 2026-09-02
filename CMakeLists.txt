cmake_minimum_required(VERSION 3.24)
project(cuda_programming_exercises LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOLUTIONS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/solutions/*.cu")
foreach(src ${SOLUTIONS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
