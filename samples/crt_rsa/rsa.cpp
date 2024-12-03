#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>

#include <cuda_runtime_api.h>
#include "xmp.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Timer function
static double wallclock()
{
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

#define XMP_CHECK_ERROR(fun)                                                                                  \
    {                                                                                                         \
        xmpError_t error = fun;                                                                               \
        if (error != xmpErrorSuccess)                                                                         \
        {                                                                                                     \
            if (error == xmpErrorCuda)                                                                        \
                printf("CUDA Error %s, %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
            else                                                                                              \
                printf("XMP Error %s, %s:%d\n", xmpGetErrorString(error), __FILE__, __LINE__);                \
            exit(EXIT_FAILURE);                                                                               \
        }                                                                                                     \
    }

// Constants
constexpr int KEY_BITS = 2048;
constexpr int HALF_KEY_BITS = KEY_BITS / 2;

// Memory size calculations
constexpr uint32_t FULL_KEY_SIZE_WORDS = KEY_BITS / 8 / sizeof(uint32_t);
constexpr uint32_t HALF_KEY_SIZE_WORDS = (FULL_KEY_SIZE_WORDS + 1) / 2; // Rounded up
constexpr size_t FULL_KEY_SIZE_BYTES = FULL_KEY_SIZE_WORDS * sizeof(uint32_t);
constexpr size_t HALF_KEY_SIZE_BYTES = FULL_KEY_SIZE_BYTES / 2;

class HostAndCudaVariableManager {
public:
    // Constructor: initializes variables and allocates memory
    HostAndCudaVariableManager(size_t fullKeySizeBytes)
        : fullKeySizeBytes_(fullKeySizeBytes),
          fullKeySizeWords_(fullKeySizeBytes / sizeof(uint32_t)),
          hostVar_(std::make_unique<uint32_t[]>(fullKeySizeWords_)) {
            xmpHandleCreate(&xmpHandle_);
          }

    // Destructor: ensures proper cleanup
    ~HostAndCudaVariableManager() {
        xmpIntegersDestroy(xmpHandle_, cudaVar_);
        xmpHandleDestroy(xmpHandle_);
    }

private:
    xmpHandle_t xmpHandle_;                       // Handle for xmp functions
    size_t fullKeySizeBytes_;                    // Size in bytes
    size_t fullKeySizeWords_;                    // Size in words
    std::unique_ptr<uint32_t[]> hostVar_;        // Managed hostVar array
    xmpIntegers_t cudaVar_;                      // Encapsulated xmpIntegers_t
};

// Example usage
int main() {
    size_t fullKeySizeBytes = 1024; // Example size
    std::string varString = "123456789"; // Example input

    // Create and manage host variable and cudaVar
    HostAndCudaVariableManager manager(fullKeySizeBytes);

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}
