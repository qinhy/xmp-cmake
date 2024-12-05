#include <iostream>   // For input/output operations
#include <vector>     // For std::vector
#include <string>     // For std::string
#include <sstream>    // For std::stringstream
#include <iomanip>    // For std::setprecision and formatting
#include <cstdlib>    // For std::rand, std::srand, and general utilities
#include <cmath>      // For mathematical functions
#include <chrono>     // For time-related utilities
#include <bitset>     // For binary manipulation with std::bitset

#include <cuda_runtime_api.h>
#include "xmp.h"

#define XMP_CHECK_ERROR(fun) \
{                             \
  xmpError_t error=fun;     \
  if(error!=xmpErrorSuccess){ \
    if(error==xmpErrorCuda)   \
      printf("CUDA Error %s, %s:%d\n",cudaGetErrorString(cudaGetLastError()),__FILE__,__LINE__); \
    else  \
      printf("XMP Error %s, %s:%d\n",xmpGetErrorString(error),__FILE__,__LINE__); \
    exit(EXIT_FAILURE); \
  } \
}

// Function to convert binary data to a hexadecimal string
std::string binaryToHexString(const std::vector<uint32_t>& data) {
    size_t size = data.size();
    std::ostringstream hexStream;
    for (size_t i = 0; i < size; ++i) {
        hexStream << std::hex << std::setw(8) << std::setfill('0') << data[size - i - 1];
    }
    return hexStream.str();
}
std::string binaryToBitString(const std::vector<uint32_t>& data) {
    size_t size = data.size();
    std::ostringstream bitStream;
    for (size_t i = 0; i < size; ++i) {
        // Convert each uint32_t value to a 32-bit binary string
        bitStream << std::bitset<32>(data[size - i - 1]);
    }
    return bitStream.str();
}

int main() {
    int bits = 64; // Bit size of the integers
    uint32_t limbs = bits / 8 / sizeof(uint32_t); // Number of 32-bit words per integer

    xmpHandle_t handle;
    xmpIntegers_t a, b, c;

    // Initialize CUDA device
    cudaSetDevice(0);

    // Create XMP handle
    XMP_CHECK_ERROR(xmpHandleCreate(&handle));

    // Create XMP integers
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &a, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &b, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &c, bits, 1));

    // Initialize data for a and b
    std::vector<uint32_t> a_data(limbs, 0x00f00000); // Example value for a
    std::vector<uint32_t> b_data(limbs, 0x00000000); // Example value for b
    b_data[0] = 0x00000002;
    std::vector<uint32_t> c_data(limbs, 0);          // To store the result

    // Import values into XMP integers
    XMP_CHECK_ERROR(xmpIntegersImport(handle, a, limbs, -1, sizeof(uint32_t), 0, 0, a_data.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, b, limbs, -1, sizeof(uint32_t), 0, 0, b_data.data(), 1));

    // Perform multiplication: c = a * b
    XMP_CHECK_ERROR(xmpIntegersMul(handle, c, a, b, 1));

    // Export result back to the host
    XMP_CHECK_ERROR(xmpIntegersExport(handle, c_data.data(), &limbs, -1, sizeof(uint32_t), 0, 0, c, 1));

    // Print result
    std::cout << "Result: ";
    for (auto word : c_data) {
        std::cout << std::hex << word << " ";
    }
    std::cout << std::endl;

    // Free XMP integers
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, a));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, b));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, c));

    // Destroy XMP handle
    XMP_CHECK_ERROR(xmpHandleDestroy(handle));
    
    std::cout << binaryToBitString(a_data) << std::endl;
    std::cout << binaryToBitString(b_data) << std::endl;
    std::cout << binaryToBitString(c_data) << std::endl;

    std::cout << "Multiplication completed successfully." << std::endl;
    return 0;
}
