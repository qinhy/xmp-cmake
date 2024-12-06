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

std::vector<uint32_t> bitStringToBinary(const std::string& bitString) {
    const size_t bitsPerElement = 32; // Number of bits per uint32_t
    std::vector<uint32_t> data;

    // Ensure the bitString's size is a multiple of 32 by padding with zeros if necessary
    size_t paddedSize = ((bitString.size() + bitsPerElement - 1) / bitsPerElement) * bitsPerElement;
    std::string paddedBitString = std::string(paddedSize - bitString.size(), '0') + bitString;

    // Convert each 32-bit chunk of the bit string into a uint32_t value
    for (size_t i = 0; i < paddedSize; i += bitsPerElement) {
        std::string chunk = paddedBitString.substr(i, bitsPerElement);
        uint32_t value = static_cast<uint32_t>(std::bitset<32>(chunk).to_ulong());
        data.push_back(value);
    }

    // Reverse the vector because the input bit string represents numbers in reverse order
    std::reverse(data.begin(), data.end());

    return data;
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
    auto a_data = bitStringToBinary("0000000011110000000000000000000000000000111100000000000000000000");
    auto b_data = bitStringToBinary("0000000000000000000000000000000000000000000000000000000000000010");
    auto c_data = bitStringToBinary("0000000000000000000000000000000000000000000000000000000000000000");

    // Import values into XMP integers
    XMP_CHECK_ERROR(xmpIntegersImport(handle, a, limbs, -1, sizeof(uint32_t), 0, 0, a_data.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, b, limbs, -1, sizeof(uint32_t), 0, 0, b_data.data(), 1));

    // Perform multiplication: c = a * b
    XMP_CHECK_ERROR(xmpIntegersMul(handle, c, a, b, 1));

    // Export result back to the host
    XMP_CHECK_ERROR(xmpIntegersExport(handle, c_data.data(), &limbs, -1, sizeof(uint32_t), 0, 0, c, 1));

    // Free XMP integers
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, a));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, b));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, c));

    // Destroy XMP handle
    XMP_CHECK_ERROR(xmpHandleDestroy(handle));
    
    std::cout << "a: \n";
    std::cout << binaryToBitString(a_data) << std::endl;
    std::cout << "b: \n";
    std::cout << binaryToBitString(b_data) << std::endl;
    std::cout << "a * b = c: \n";
    std::cout << binaryToBitString(c_data) << std::endl;

    std::cout << "Multiplication completed successfully." << std::endl;
    return 0;
}
