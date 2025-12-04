cmake_minimum_required(VERSION 3.18)
project(cuda_apriltag LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

option(BUILD_PYTHON_WRAPPER "Build Python wrapper with pybind11" ON)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_PYTHON_WRAPPER)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 QUIET)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    "-O3"
    "--use_fast_math"
    "-lineinfo")

set(SRC_FILES
    src/gpu_context.cpp
    src/image_preprocessor.cu
    src/apriltag_gpu.cu)

add_library(cuda_apriltag SHARED ${SRC_FILES})
target_link_libraries(cuda_apriltag PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES})
set_target_properties(cuda_apriltag PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

add_executable(apriltag_demo src/main.cpp src/config.cpp src/camera_intrinsics.cpp src/camera_controls.cpp)
target_link_libraries(apriltag_demo PRIVATE cuda_apriltag ${OpenCV_LIBS})

add_executable(quad_extraction_benchmark src/quad_extraction_benchmark.cpp)
target_link_libraries(quad_extraction_benchmark PRIVATE cuda_apriltag ${OpenCV_LIBS})

add_executable(tune_camera_with_tag src/tune_camera_with_tag.cpp src/camera_intrinsics.cpp)
target_link_libraries(tune_camera_with_tag PRIVATE cuda_apriltag ${OpenCV_LIBS})

if(BUILD_PYTHON_WRAPPER AND pybind11_FOUND)
  pybind11_add_module(cuda_apriltag_py python/binding.cpp)
  target_link_libraries(cuda_apriltag_py PRIVATE cuda_apriltag)
endif()


