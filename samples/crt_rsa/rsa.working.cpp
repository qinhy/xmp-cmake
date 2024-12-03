#include <iostream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdint>
#include <algorithm> // For std::fill

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

class CudaVariableManager
{
public:
    // Constructor: initializes variables and allocates memory
    CudaVariableManager(size_t fullKeySizeBits)
        : fullKeySizeBits_(fullKeySizeBits),
          fullKeySizeBytes_(fullKeySizeBits_ / 8),
          fullKeySizeWords_(fullKeySizeBytes_ / sizeof(uint32_t))
    {
        XMP_CHECK_ERROR(xmpHandleCreate(&xmpHandle_));
    }

    // Destructor: ensures proper cleanup
    ~CudaVariableManager()
    {
        
        std::cout << "Clean up all managed CUDA variables" << std::endl;
        // Clean up all managed CUDA variables
        for (auto it = cudaVars_.begin(); it != cudaVars_.end(); ++it) {
            XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle_, *it->second));
        }

        std::cout << "xmpHandleDestroy" << std::endl;
        XMP_CHECK_ERROR(xmpHandleDestroy(xmpHandle_));
    }

    void importDecimalString(std::vector<uint32_t>& array, const std::string& decimalString) {
        // Resize the array to fit the required number of 32-bit chunks (if not already sized appropriately)
        array.assign(array.size(), 0);

        // Arbitrary-precision storage for processing
        std::string number = decimalString; // Make a copy of the input string
        size_t len = number.size();

        // Iterate and extract 32-bit chunks in little-endian order
        for (size_t i = 0; i < array.size(); i++) {
            if (len == 0)
                break;

            // Process the number string as a whole, extracting 32-bit chunks
            uint64_t remainder = 0;

            // Simulate division of the large number string by 2^32 (4294967296)
            for (size_t j = 0; j < len; j++) {
                remainder = remainder * 10 + (number[j] - '0');
                number[j] = (remainder / 4294967296) + '0'; // Update number with quotient
                remainder %= 4294967296;                    // Update remainder
            }

            // Find the new length (trim leading zeros)
            size_t newStart = number.find_first_not_of('0');
            if (newStart == std::string::npos) {
                len = 0;
                number.clear();
            } else {
                number = number.substr(newStart);
                len = number.size();
            }

            // Store the least significant 32 bits of remainder in the array
            array[i] = static_cast<uint32_t>(remainder);
        }
    }
    // Add a new variable (allocates memory for host and CUDA)
    void addVariable(const std::string &key, const std::string &varString, size_t size=0)
    {
        if (cudaVars_.find(key) != cudaVars_.end())
        {
            throw std::runtime_error("Variable already exists: " + key);
        }

        // Calculate the size dynamically
        // Each 32-bit chunk can represent up to 10 digits (2^32 - 1 = 4294967295)
        if(size==0)size = fullKeySizeWords_;
        hostTmp_.resize(size,0);
        importDecimalString(hostTmp_,varString);

        // Allocate CUDA variable
        auto cudaVar = std::make_shared<xmpIntegers_t>();
        XMP_CHECK_ERROR(xmpIntegersCreate(xmpHandle_, &(*cudaVar), fullKeySizeBits_, 1));
        XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle_, *cudaVar, fullKeySizeWords_, -1, sizeof(uint32_t), 0, 0, hostTmp_.data(), 1));
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

    std::string binaryToDecimalString(const std::vector<uint32_t> &data)
    {
        std::vector<uint8_t> decimal; // Vector to store the decimal digits

        for (const auto &value : data)
        {
            for (int j = 0; j < 32; j += 8)
            {
                uint8_t byte = (value >> j) & 0xFF; // Extract each byte
                size_t carry = byte;
                for (size_t k = 0; k < decimal.size(); ++k)
                {
                    carry += decimal[k] * 256;
                    decimal[k] = carry % 10;
                    carry /= 10;
                }
                while (carry)
                {
                    decimal.push_back(carry % 10);
                    carry /= 10;
                }
            }
        }

        // Convert the vector of digits into a string
        std::string result;
        for (auto it = decimal.rbegin(); it != decimal.rend(); ++it)
        {
            result += ('0' + *it);
        }
        return result.empty() ? "0" : result;
    }

    auto exportCudaVariable(const std::string &key)
    {
        auto var = getCudaVariable(key);
        hostTmp_.resize(fullKeySizeWords_,0);
        std::fill(hostTmp_.begin(), hostTmp_.end(), 0);
        auto array = hostTmp_.data();
        auto arraySize = hostTmp_.size();
        
        for (int val : hostTmp_) {
            std::cout << val << " ";
        }

        uint32_t fullKeySizeWords = fullKeySizeWords_;
        XMP_CHECK_ERROR(xmpIntegersExport(xmpHandle_, hostTmp_.data(), &fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, *var, 1));

        for (int val : hostTmp_) {
            std::cout << val << " ";
        }
        std::string varString = binaryToDecimalString(hostTmp_);
        std::cout <<  varString << std::endl;
        return varString;
    }

    // Remove a variable (deallocates memory)
    void removeVariable(const std::string &key)
    {
        // Remove CUDA variable
        auto cudaIt = cudaVars_.find(key);
        if (cudaIt != cudaVars_.end())
        {
            XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle_, *cudaIt->second));
            cudaVars_.erase(cudaIt);
        }
    }

private:
    xmpHandle_t xmpHandle_;                         // Handle for xmp functions
    size_t fullKeySizeBits_,fullKeySizeBytes_,fullKeySizeWords_;                       // Size in words
    std::vector<uint32_t> hostTmp_;                 // Managed hostVar array
    std::map<std::string, std::shared_ptr<xmpIntegers_t>> cudaVars_; // CUDA variables
};

// Example usage
int main()
{
    size_t fullKeySizeBits = 2048;      // Example size
    std::string varString = "123456789"; // Example input

    // Create and manage host variable and cudaVar
    CudaVariableManager manager(fullKeySizeBits);

    // Import data
    manager.addVariable("test", varString);
    std::cout <<  manager.exportCudaVariable("test") << std::endl;

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}
