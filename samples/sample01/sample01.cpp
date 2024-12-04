/***
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
***/
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <bitset>

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
    int bits = 64;
    uint32_t i, w;
    uint32_t N = 15 * bits * 4;
    xmpIntegers_t base, mod, exp, out;
    uint32_t limbs = bits / 8 / sizeof(uint32_t);

    size_t bytes = N * bits / 8;

    // Use std::vector instead of malloc
    std::vector<uint32_t> b(N * limbs, 0);
    std::vector<uint32_t> o(N * limbs, 0);
    std::vector<uint32_t> m(limbs, 0);
    std::vector<uint32_t> e(limbs, 0);

    xmpHandle_t handle;

    cudaSetDevice(0);
    // Allocate handle
    XMP_CHECK_ERROR(xmpHandleCreate(&handle));

    // Allocate integers
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &base, bits, N));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &out, bits, N));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &exp, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &mod, bits, 1));

    // Initialize base, exp, and mod
    for (i = 0; i < N; i++) {
        for (w = 0; w < limbs; w++) {
            b[i * limbs + w] = 0xffffffff;
        }
    }

    for (w = 0; w < limbs; w++) {
        m[w] = 0xffffffff;
        e[w] = 0xffffffff;
    }
    // Make sure modulus is odd
    m[0] |= 1;

    // Import 
    XMP_CHECK_ERROR(xmpIntegersImport(handle, base, limbs, -1, sizeof(uint32_t), 0, 0, b.data(), N));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, exp, limbs, -1, sizeof(uint32_t), 0, 0, e.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, mod, limbs, -1, sizeof(uint32_t), 0, 0, m.data(), 1));

    // Call powm
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, out, base, exp, mod, N));

    // Export
    // XMP_CHECK_ERROR(xmpIntegersExport(handle, o.data(), &limbs, -1, sizeof(uint32_t), 0, 0, out, N));
    XMP_CHECK_ERROR(xmpIntegersExport(handle, o.data(), &limbs, -1, sizeof(uint32_t), 0, 0, exp, 1));

    
    // Convert the first result to a decimal string
    {
        std::string resultDecimal = binaryToBitString(e);
        std::cout << "Result (Decimal): " << resultDecimal << std::endl;
    }
    {
        std::string resultDecimal = binaryToBitString(m);
        std::cout << "Result (Decimal): " << resultDecimal << std::endl;
    }

    // Free integers
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, base));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, out));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, exp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, mod));

    // Free handle
    XMP_CHECK_ERROR(xmpHandleDestroy(handle));

    std::cout << "sample01 executed successfully" << std::endl;
    return 0;
}