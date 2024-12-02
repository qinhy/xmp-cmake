#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include "xmp.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Timer function
static double wallclock() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) / static_cast<double>(freq.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1e6;
#endif
}

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

uint32_t rand32() {
    uint32_t lo = rand() & 0xffff;
    uint32_t hi = rand() & 0xffff;
    return (hi << 16) | lo;
}

#include <iostream>
#include <cstdlib>
#include "xmp.h" // Assuming the necessary header

int main() {
    // Constants
    const int totalMessages = 1000;
    const int keyBits = 2048;
    const int halfKeyBits = keyBits / 2;

    // Timing variables
    double startTime, endTime;

    // Memory size calculations
    uint32_t fullKeySizeWords = keyBits / 8 / sizeof(uint32_t);
    uint32_t halfKeySizeWords = (fullKeySizeWords + 1) / 2; // +1 to round up
    size_t fullKeySizeBytes = fullKeySizeWords * sizeof(uint32_t);
    size_t halfKeySizeBytes = fullKeySizeBytes / 2;

    // Handle and integer objects
    xmpHandle_t xmpHandle;
    xmpIntegers_t publicModulus, publicExponent;     // Public key
    xmpIntegers_t primeP, primeQ, privateExponent;   // Private key
    xmpIntegers_t plaintext, ciphertext, result;     // Messages
    xmpIntegers_t dp, dq, coefficientP, coefficientQ;
    xmpIntegers_t partialCipherP, partialCipherQ, partialPlainP, partialPlainQ;
    xmpIntegers_t scratchSpaceSum, scratchSpaceResult;

    // Host data for initialization
    uint32_t* hostModulus = (uint32_t*)calloc(1, fullKeySizeBytes);
    uint32_t* hostExponent = (uint32_t*)calloc(1, sizeof(uint32_t));
    uint32_t* hostPrimeP = (uint32_t*)calloc(1, halfKeySizeBytes);
    uint32_t* hostPrimeQ = (uint32_t*)calloc(1, halfKeySizeBytes);
    uint32_t* hostMessages = (uint32_t*)calloc(totalMessages, fullKeySizeBytes);
    uint32_t* hostDP = (uint32_t*)calloc(1, halfKeySizeBytes);
    uint32_t* hostDQ = (uint32_t*)calloc(1, halfKeySizeBytes);
    uint32_t* hostCoefficientP = (uint32_t*)calloc(1, fullKeySizeBytes);
    uint32_t* hostCoefficientQ = (uint32_t*)calloc(1, fullKeySizeBytes);
    int32_t* validationResults = (int32_t*)calloc(totalMessages, sizeof(int32_t));

    // Create XMP handle and integer objects
    XMP_CHECK_ERROR(xmpHandleCreate(&xmpHandle));

    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &publicModulus, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &publicExponent, 32, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &primeP, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &primeQ, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &privateExponent, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &coefficientP, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &coefficientQ, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &plaintext, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &ciphertext, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &result, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &partialCipherP, keyBits + halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &partialCipherQ, keyBits + halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &partialPlainP, halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &partialPlainQ, halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &dp, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &dq, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle, &scratchSpaceResult, halfKeyBits, totalMessages));

    // Hardcoded keys for testing
    hostModulus[0] = 17460671;
    hostExponent[0] = 65537;
    hostPrimeP[0] = 4931;
    hostPrimeQ[0] = 3541;
    hostDP[0] = 1063;
    hostDQ[0] = 113;
    hostCoefficientP[0] = 15212136;
    hostCoefficientQ[0] = 2248536;

    // Generate random plaintext
    for (int i = 0; i < totalMessages; i++) {
        hostMessages[i * fullKeySizeWords] = rand32() % hostModulus[0];
    }

    // Import keys and plaintext
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, publicModulus, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostModulus, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, publicExponent, 1, -1, sizeof(uint32_t), 0, 0, hostExponent, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, primeP, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostPrimeP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, primeQ, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostPrimeQ, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, plaintext, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostMessages, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, dp, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostDP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, dq, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostDQ, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, coefficientP, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostCoefficientP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, coefficientQ, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostCoefficientQ, 1));

    // Encryption
    startTime = wallclock();
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, ciphertext, plaintext, publicExponent, publicModulus, totalMessages));
    endTime = wallclock();
    std::cout << "Encryption time: " << endTime - startTime
              << ", " << keyBits << "-bit throughput: "
              << totalMessages / (endTime - startTime) << " encryptions/second" << std::endl;

    // Decryption
    startTime = wallclock();
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, scratchSpaceResult, ciphertext, primeP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, partialPlainP, scratchSpaceResult, dp, primeP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, scratchSpaceResult, ciphertext, primeQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, partialPlainQ, scratchSpaceResult, dq, primeQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMul(xmpHandle, partialCipherP, partialPlainP, coefficientP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMul(xmpHandle, partialCipherQ, partialPlainQ, coefficientQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersAdd(xmpHandle, partialCipherP, partialCipherP, partialCipherQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, result, partialCipherP, publicModulus, totalMessages));
    endTime = wallclock();
    std::cout << "Decryption time: " << endTime - startTime
              << ", " << keyBits << "-bit throughput: "
              << totalMessages / (endTime - startTime) << " decryptions/second" << std::endl;

    // Validation
    XMP_CHECK_ERROR(xmpIntegersCmp(xmpHandle, validationResults, plaintext, result, totalMessages));
    std::cout << "Validating results..." << std::endl;
    for (int i = 0; i < totalMessages; i++) {
        if (validationResults[i] != 0) {
            std::cerr << "  Error at index " << i << std::endl;
            exit(1);
        }
    }

    // Clean up
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, publicModulus));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, publicExponent));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, primeP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, primeQ));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, plaintext));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, ciphertext));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, scratchSpaceResult));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, partialCipherP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, partialCipherQ));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, dp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, dq));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, coefficientP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, coefficientQ));
    XMP_CHECK_ERROR(xmpHandleDestroy(xmpHandle));

    free(hostModulus);
    free(hostExponent);
    free(hostPrimeP);
    free(hostPrimeQ);
    free(hostMessages);
    free(hostDP);
    free(hostDQ);
    free(hostCoefficientP);
    free(hostCoefficientQ);
    free(validationResults);

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}
