cmake_minimum_required(VERSION 3.10)
project(XMP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src/include)

# Source files
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)

# Add CUDA support if needed
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++14)
    cuda_add_library(xmp STATIC ${SRC_FILES})
else()
    add_library(xmp STATIC ${SRC_FILES})
endif()

# Add executable (if applicable)
add_executable(simple_rsa ${PROJECT_SOURCE_DIR}/samples/simple_rsa/rsa.cc)

# Link libraries
target_link_libraries(simple_rsa xmp)
