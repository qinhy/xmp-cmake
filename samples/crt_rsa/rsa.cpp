#include <iostream>
#include <cstdlib>
#include <cstring>
#include <map>
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

class HostAndCudaVariableManager
{
public:
    // Constructor: initializes variables and allocates memory
    HostAndCudaVariableManager(size_t fullKeySizeBytes)
        : fullKeySizeWords_(fullKeySizeBytes / sizeof(uint32_t))
    {
        xmpHandleCreate(&xmpHandle_);
    }

    // Destructor: ensures proper cleanup
    ~HostAndCudaVariableManager()
    {
        // Clean up all managed CUDA variables
        for (auto it = cudaVars_.begin(); it != cudaVars_.end(); ++it) {
            xmpIntegersDestroy(xmpHandle_, *it->second);
        }

        xmpHandleDestroy(xmpHandle_);
    }

    // Add a new variable (allocates memory for host and CUDA)
    void addVariable(const std::string &key, const std::string &varString)
    {
        if (cudaVars_.find(key) != cudaVars_.end())
        {
            throw std::runtime_error("Variable already exists: " + key);
        }

        // Calculate the size dynamically
        // Each 32-bit chunk can represent up to 10 digits (2^32 - 1 = 4294967295)
        size_t size = (varString.length() + 9) / 10; // Round up to next whole chunk
        size = fullKeySizeWords_;

        // Resize host temporary buffer
        hostTmp_.resize(size);
        auto array = hostTmp_.data();
        auto arraySize = hostTmp_.size();
        auto decimalString = varString.c_str();

        // Initialize the array
        memset(array, 0, arraySize * sizeof(uint32_t));

        // Arbitrary-precision storage for processing
        char *number = strdup(decimalString); // Duplicate string for modification
        size_t len = strlen(number);

        // Iterate and extract 32-bit chunks in little-endian order
        for (size_t i = 0; i < arraySize; i++)
        {
            if (len == 0)
                break;

            // Process the number string as a whole, extracting 32-bit chunks
            uint64_t remainder = 0;

            // Simulate division of the large number string by 2^32 (4294967296)
            for (size_t j = 0; j < len; j++)
            {
                remainder = remainder * 10 + (number[j] - '0');
                number[j] = (remainder / 4294967296) + '0'; // Update number with quotient
                remainder %= 4294967296;                    // Update remainder
            }

            // Find the new length (trim leading zeros)
            char *newStart = number;
            while (*newStart == '0' && *newStart != '\0')
                newStart++;
            len = strlen(newStart);
            memmove(number, newStart, len + 1);

            // Store the least significant 32 bits of remainder in the array
            array[i] = (uint32_t)remainder;
        }

        free(number);

        // Allocate CUDA variable
        auto cudaVar = std::make_shared<xmpIntegers_t>();
        // Import the number into the CUDA variable
        // xmpIntegersImport(xmpHandle_, *cudaVar, fullKeySizeWords_, -1, sizeof(uint32_t), 0, 0, hostTmp_.data(), 1);
        cudaVars_[key] = cudaVar;
    }

    // Get CUDA variable handle
    auto getCudaVariable(const std::string &key) const
    {
        auto it = cudaVars_.find(key);
        if (it == cudaVars_.end())
        {
            throw std::runtime_error("CUDA variable not found: " + key);
        }
        return it->second;
    }

    // Remove a variable (deallocates memory)
    void removeVariable(const std::string &key)
    {
        // Remove CUDA variable
        auto cudaIt = cudaVars_.find(key);
        if (cudaIt != cudaVars_.end())
        {
            xmpIntegersDestroy(xmpHandle_, *cudaIt->second);
            cudaVars_.erase(cudaIt);
        }
    }

private:
    xmpHandle_t xmpHandle_;                         // Handle for xmp functions
    size_t fullKeySizeWords_;                       // Size in words
    std::vector<uint32_t> hostTmp_;                 // Managed hostVar array
    std::map<std::string, std::shared_ptr<xmpIntegers_t>> cudaVars_; // CUDA variables
};

// Example usage
int main()
{
    size_t fullKeySizeBytes = 1024;      // Example size
    std::string varString = "123456789"; // Example input

    // Create and manage host variable and cudaVar
    HostAndCudaVariableManager manager(fullKeySizeBytes);

    // Import data
    manager.addVariable("test", varString);

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}
