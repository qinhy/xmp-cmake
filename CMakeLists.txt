cmake_minimum_required(VERSION 3.10)
project(XMP)

# Set C++ standard
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_ITERATOR_DEBUG_LEVEL=2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_ITERATOR_DEBUG_LEVEL=0")

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
add_executable(simple_rsa ${PROJECT_SOURCE_DIR}/samples/simple_rsa/rsa.cpp)
add_executable(crt_rsa ${PROJECT_SOURCE_DIR}/samples/crt_rsa/rsa.cpp)
add_executable(sample01 ${PROJECT_SOURCE_DIR}/samples/sample01/sample01.cpp)

# Link libraries
target_link_libraries(simple_rsa xmp)
target_link_libraries(crt_rsa xmp)
target_link_libraries(sample01 xmp)
