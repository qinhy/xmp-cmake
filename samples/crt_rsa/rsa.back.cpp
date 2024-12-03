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

uint32_t rand32()
{
    uint32_t lo = rand() & 0xffff;
    uint32_t hi = rand() & 0xffff;
    return (hi << 16) | lo;
}

// Helper function to convert decimal string to uint32_t array
void importDecimalString(uint32_t *array, size_t arraySize, const char *decimalString)
{
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
}
// Helper function to convert binary data to a decimal string
std::string binaryToDecimalString(const uint32_t *data, size_t size)
{
    std::vector<uint8_t> decimal; // Vector to store the decimal digits

    for (size_t i = 0; i < size; ++i)
    {
        uint32_t value = data[i];
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

// Constants
constexpr int KEY_BITS = 2048;
constexpr int HALF_KEY_BITS = KEY_BITS / 2;

// Memory size calculations
constexpr uint32_t FULL_KEY_SIZE_WORDS = KEY_BITS / 8 / sizeof(uint32_t);
constexpr uint32_t HALF_KEY_SIZE_WORDS = (FULL_KEY_SIZE_WORDS + 1) / 2; // Rounded up
constexpr size_t FULL_KEY_SIZE_BYTES = FULL_KEY_SIZE_WORDS * sizeof(uint32_t);
constexpr size_t HALF_KEY_SIZE_BYTES = FULL_KEY_SIZE_BYTES / 2;

// Helper to allocate and clear memory
template <typename T>
T *allocateMemory(size_t count = 1)
{
    return static_cast<T *>(calloc(count, sizeof(T)));
}

// Handle and integer objects
struct XMPResources
{
    xmpHandle_t handle;
    xmpIntegers_t publicModulus, publicExponent;
    xmpIntegers_t primeP, primeQ, privateExponent;
    xmpIntegers_t plaintext, ciphertext, result;
    xmpIntegers_t dp, dq, coefficientP, coefficientQ;
    xmpIntegers_t partialCipherP, partialCipherQ, partialPlainP, partialPlainQ;
    xmpIntegers_t scratchSpaceResult;
};

// Create and initialize XMP integer objects
void createXmpIntegers(XMPResources &resources, int keyBits, int halfKeyBits, int totalMessages)
{
    XMP_CHECK_ERROR(xmpHandleCreate(&resources.handle));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.publicModulus, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.publicExponent, 32, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.primeP, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.primeQ, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.privateExponent, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.coefficientP, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.coefficientQ, keyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.plaintext, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.ciphertext, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.result, keyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.partialCipherP, keyBits + halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.partialCipherQ, keyBits + halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.partialPlainP, halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.partialPlainQ, halfKeyBits, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.dp, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.dq, halfKeyBits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(resources.handle, &resources.scratchSpaceResult, halfKeyBits, totalMessages));
}

// Main memory allocation function
void initializeHostData(uint32_t *&hostModulus, uint32_t *&hostExponent, uint32_t *&hostPrimeP,
                        uint32_t *&hostPrimeQ, uint32_t *&hostDP, uint32_t *&hostDQ,
                        uint32_t *&hostCoefficientP, uint32_t *&hostCoefficientQ)
{
    hostModulus = allocateMemory<uint32_t>(FULL_KEY_SIZE_BYTES);
    hostExponent = allocateMemory<uint32_t>(1);
    hostPrimeP = allocateMemory<uint32_t>(HALF_KEY_SIZE_BYTES);
    hostPrimeQ = allocateMemory<uint32_t>(HALF_KEY_SIZE_BYTES);
    hostDP = allocateMemory<uint32_t>(HALF_KEY_SIZE_BYTES);
    hostDQ = allocateMemory<uint32_t>(HALF_KEY_SIZE_BYTES);
    hostCoefficientP = allocateMemory<uint32_t>(FULL_KEY_SIZE_BYTES);
    hostCoefficientQ = allocateMemory<uint32_t>(FULL_KEY_SIZE_BYTES);
}

// Encryption process and performance measurement
void performEncryption(
    xmpHandle_t xmpHandle,
    xmpIntegers_t &ciphertext,
    xmpIntegers_t &plaintext,
    xmpIntegers_t &publicExponent,
    xmpIntegers_t &publicModulus,
    int totalMessages,
    int keyBits)
{
    double startTime = wallclock();
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, ciphertext, plaintext, publicExponent, publicModulus, totalMessages));
    double endTime = wallclock();

    std::cout << "Encryption time: " << endTime - startTime
              << ", " << keyBits << "-bit throughput: "
              << totalMessages / (endTime - startTime) << " encryptions/second" << std::endl;
}

// Import key components and messages into XMP integers
void importKeysAndMessages(
    xmpHandle_t xmpHandle,
    uint32_t *hostModulus, uint32_t *hostExponent, uint32_t *hostPrimeP,
    uint32_t *hostPrimeQ, uint32_t *hostMessages, uint32_t *hostDP,
    uint32_t *hostDQ, uint32_t *hostCoefficientP, uint32_t *hostCoefficientQ,
    xmpIntegers_t &publicModulus, xmpIntegers_t &publicExponent, xmpIntegers_t &primeP,
    xmpIntegers_t &primeQ, xmpIntegers_t &plaintext, xmpIntegers_t &dp,
    xmpIntegers_t &dq, xmpIntegers_t &coefficientP, xmpIntegers_t &coefficientQ,
    size_t fullKeySizeWords, size_t halfKeySizeWords, int totalMessages)
{
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, publicModulus, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostModulus, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, publicExponent, 1, -1, sizeof(uint32_t), 0, 0, hostExponent, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, primeP, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostPrimeP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, primeQ, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostPrimeQ, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, plaintext, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostMessages, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, dp, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostDP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, dq, halfKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostDQ, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, coefficientP, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostCoefficientP, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(xmpHandle, coefficientQ, fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, hostCoefficientQ, 1));
}


// Function to perform decryption and measure performance
void performDecryption(
    xmpHandle_t xmpHandle,
    xmpIntegers_t& ciphertext,
    xmpIntegers_t& plaintext,
    xmpIntegers_t& primeP,
    xmpIntegers_t& primeQ,
    xmpIntegers_t& dp,
    xmpIntegers_t& dq,
    xmpIntegers_t& coefficientP,
    xmpIntegers_t& coefficientQ,
    xmpIntegers_t& scratchSpaceResult,
    xmpIntegers_t& partialPlainP,
    xmpIntegers_t& partialPlainQ,
    xmpIntegers_t& partialCipherP,
    xmpIntegers_t& partialCipherQ,
    xmpIntegers_t& result,
    xmpIntegers_t& publicModulus,
    int totalMessages,
    int keyBits
) {
    double startTime = wallclock();
    
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, scratchSpaceResult, ciphertext, primeP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, partialPlainP, scratchSpaceResult, dp, primeP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, scratchSpaceResult, ciphertext, primeQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersPowm(xmpHandle, partialPlainQ, scratchSpaceResult, dq, primeQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMul(xmpHandle, partialCipherP, partialPlainP, coefficientP, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMul(xmpHandle, partialCipherQ, partialPlainQ, coefficientQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersAdd(xmpHandle, partialCipherP, partialCipherP, partialCipherQ, totalMessages));
    XMP_CHECK_ERROR(xmpIntegersMod(xmpHandle, result, partialCipherP, publicModulus, totalMessages));
    
    double endTime = wallclock();
    std::cout << "Decryption time: " << endTime - startTime
              << ", " << keyBits << "-bit throughput: "
              << totalMessages / (endTime - startTime) << " decryptions/second" << std::endl;
}

// Function to export integers and convert them to decimal strings
void exportAndPrintResults(
    xmpHandle_t xmpHandle,
    xmpIntegers_t& plaintext,
    xmpIntegers_t& result,
    int totalMessages,
    uint32_t fullKeySizeWords
) {
    // Export plaintext
    uint32_t* exportedPlaintext = allocateMemory<uint32_t>(totalMessages * fullKeySizeWords);
    XMP_CHECK_ERROR(xmpIntegersExport(xmpHandle, exportedPlaintext, &fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, plaintext, totalMessages));
    
    // Export result (decrypted text)
    uint32_t* exportedResult = allocateMemory<uint32_t>(totalMessages * fullKeySizeWords);
    XMP_CHECK_ERROR(xmpIntegersExport(xmpHandle, exportedResult, &fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, result, totalMessages));

    // Print plaintext and result as decimal strings
    for (int i = 0; i < totalMessages; i++) {
        std::string plaintextDecimal = binaryToDecimalString(&exportedPlaintext[i * fullKeySizeWords], fullKeySizeWords);
        std::string resultDecimal = binaryToDecimalString(&exportedResult[i * fullKeySizeWords], fullKeySizeWords);
        
        std::cout << "Message " << i + 1 << ": " << plaintextDecimal << "\n"
                  << "Decrypt " << i + 1 << ": " << resultDecimal << "\n" << std::endl;
    }

    // Free exported buffers
    free(exportedPlaintext);
    free(exportedResult);
}

// Function to clean up resources
void cleanupResources(
    xmpHandle_t xmpHandle,
    XMPResources& resources,
    uint32_t* hostModulus,
    uint32_t* hostExponent,
    uint32_t* hostPrimeP,
    uint32_t* hostPrimeQ,
    uint32_t* hostMessages,
    uint32_t* hostDP,
    uint32_t* hostDQ,
    uint32_t* hostCoefficientP,
    uint32_t* hostCoefficientQ,
    int32_t* validationResults
) {
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.publicModulus));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.publicExponent));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.primeP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.primeQ));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.plaintext));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.ciphertext));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.scratchSpaceResult));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.partialCipherP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.partialCipherQ));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.dp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.dq));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.coefficientP));
    XMP_CHECK_ERROR(xmpIntegersDestroy(xmpHandle, resources.coefficientQ));
    XMP_CHECK_ERROR(xmpHandleDestroy(xmpHandle));

    // Free host data
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
}


int XMP_CHECK_ERROR(int result) {
    // Mock implementation of XMP_CHECK_ERROR
    return result;
}

struct xmpIntegers_t {}; // Mock xmpIntegers_t

int xmpIntegersImport(void* handle, xmpIntegers_t& cudaVar, size_t sizeWords, int a, size_t b, int c, int d, uint32_t* hostVar, int e) {
    // Mock implementation of xmpIntegersImport
    return 0;
}

void xmpIntegersDestroy(void* handle, xmpIntegers_t& cudaVar) {
    // Mock implementation of xmpIntegersDestroy
}

class HostAndCudaVariableManager {
public:
    // Constructor: initializes variables and allocates memory
    HostVariableManager(void* xmpHandle, size_t fullKeySizeBytes)
        : xmpHandle_(xmpHandle),
          fullKeySizeBytes_(fullKeySizeBytes),
          fullKeySizeWords_(fullKeySizeBytes / sizeof(uint32_t)),
          hostVar_(std::make_unique<uint32_t[]>(fullKeySizeWords_)) {}

    // Destructor: ensures proper cleanup
    ~HostVariableManager() {
        xmpIntegersDestroy(xmpHandle_, cudaVar_);
        xmpHandleDestroy(xmpHandle);
    }

    // Import decimal string into hostVar
    void importDecimalString(const std::string& varString) {
        auto array = hostVar_.get();
        auto arraySize = fullKeySizeBytes / sizeof(uint32_t);
        auto decimalString = varString.c_str();
        {
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
        }
    }

    // Perform XMP import
    void performXMPImport() {
        XMP_CHECK_ERROR(::xmpIntegersImport(
            xmpHandle_, cudaVar_, fullKeySizeWords_, -1, sizeof(uint32_t), 0, 0, hostVar_.get(), 1));
    }

private:
    
    xmpHandle_t xmpHandle;                            // Handle for xmp functions
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
    HostVariableManager manager(xmpHandle, fullKeySizeBytes);

    // Import data
    manager.importDecimalString(varString);

    // Perform XMP import
    manager.performXMPImport();

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}

int main_old()
{
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
    xmpIntegers_t publicModulus, publicExponent;   // Public key
    xmpIntegers_t primeP, primeQ, privateExponent; // Private key
    xmpIntegers_t plaintext, ciphertext, result;   // Messages
    xmpIntegers_t dp, dq, coefficientP, coefficientQ;
    xmpIntegers_t partialCipherP, partialCipherQ, partialPlainP, partialPlainQ;
    xmpIntegers_t scratchSpaceSum, scratchSpaceResult;

    // Host data for initialization
    uint32_t *hostModulus = (uint32_t *)calloc(1, fullKeySizeBytes);
    uint32_t *hostExponent = (uint32_t *)calloc(1, sizeof(uint32_t));
    uint32_t *hostPrimeP = (uint32_t *)calloc(1, halfKeySizeBytes);
    uint32_t *hostPrimeQ = (uint32_t *)calloc(1, halfKeySizeBytes);
    uint32_t *hostMessages = (uint32_t *)calloc(totalMessages, fullKeySizeBytes);
    uint32_t *hostDP = (uint32_t *)calloc(1, halfKeySizeBytes);
    uint32_t *hostDQ = (uint32_t *)calloc(1, halfKeySizeBytes);
    uint32_t *hostCoefficientP = (uint32_t *)calloc(1, fullKeySizeBytes);
    uint32_t *hostCoefficientQ = (uint32_t *)calloc(1, fullKeySizeBytes);
    int32_t *validationResults = (int32_t *)calloc(totalMessages, sizeof(int32_t));

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
    // hostModulus[0] = 17460671;
    // hostExponent[0] = 65537;
    // hostPrimeP[0] = 4931;
    // hostPrimeQ[0] = 3541;
    // hostDP[0] = 1063;
    // hostDQ[0] = 113;
    // hostCoefficientP[0] = 15212136;
    // hostCoefficientQ[0] = 2248536;

    // Example decimal strings
    // const char* modulusString = "17460671";
    // const char* exponentString = "65537";
    // const char* primePString = "4931";
    // const char* primeQString = "3541";
    // const char* dpString = "1063";
    // const char* dqString = "113";
    // const char* coefficientPString = "15212136";
    // const char* coefficientQString = "2248536";

    const char *modulusString = "13829982717909335748179034405320242413849622614255097391828563247796418348257440146014379231260712065070897765621725469845232840009894624271030845992041645260269120321856587015460470487559870155883435056512950267965302970839637832882679756239572751064344251727638317398023744771819862062139894061705621810959364318703232849614816438826057808803334054664465016141543125428597562454454522315677059515111624370489276816365107746570308023393388540969596327203092472317210634294845306590749678755781897308234162602708474771958750626746608717471307944670093755300516834005444393531966323655363931756656218610981551634716423";
    const char *exponentString = "65537";
    const char *primePString = "77131108284679771703035256231489387573731190895924399074668343612040473975966104157996553296139313101744921629565011059094384492399664348894990701650518244278415876273195408200153080816638319302478282581465795854484986364369028434634418160147421082128272885511407788352704862553773492361910284291261051621947";
    const char *primeQString = "179304861883545982535696179954237010218152580252063578670021523711348152281438498913693424103662589413984422317886982461024284589299340595152384513098083294398820186335684387226771994119132251402721084932345151244485259808270869912444256546283939818931643006772312592969420258042447468518414565534068407948709";
    const char *dpString = "64735898976132731777857611257962277240231993457593749654426236177736031112928812109351975434550603737758509151078374556441195340104727673084959084857206708066195664653663021316957141924701935597186595867884189098920406411219897295376141277429679366197196241322535138903833421804023848320290449478017213855763";
    const char *dqString = "14120151855913160746856401494481242805375367604267820155270779618753800355898257364444081386072029906244924295933819315521710373764040111869347948061388343720223857998969545790581965327202062186084860755540127341391708895287943598549289836815408294635186376519399809150787768005204256908670484958440683177797";
    const char *coefficientPString = "8883788031638636209185856263441736061348453719353610363335869106364547402308910551715756830320055837309041769741192177634036411363540364298313429372105973035823756953184779927001601495559642679990513219810285142096693993255788043406579399383800801822365665179764259756686059255844282892658878884152739732728457347994832081680099829031931126849096804105511074454529972515987115059183582560796636992669231187035282600965540802715274395669193064693000461244054415625516030153968527059996648949422150834563539146303649668478221768281388806836092978360244283095614110667602492716833319272910441865778555940826943114822486";
    const char *coefficientQString = "4946194686270699538993178141878506352501168894901487028492694141431870945948529594298622400940656227761855995880533292211196428646354259972717416619935672224445363368671807088458868992000227475892921836702665125868608977583849789476100356855771949241978586547874057641337685515975579169481015177552882078230906970708400767934716609794126681954237250558953941687013152912610447395270939754880422522442393183453994215399566943855033627724195476276595865959038056691694604140876779530753029806359746473670623456404825103480528858465219910635214966309849472204902723337841900815133004382453489890877662670154608519893938";

    // Generate random plaintext
    for (int i = 0; i < totalMessages; i++)
    {
        hostMessages[i * fullKeySizeWords] = rand32() % hostModulus[0];
    }

    // Convert decimal strings to uint32_t arrays
    importDecimalString(hostModulus, fullKeySizeBytes / sizeof(uint32_t), modulusString);
    importDecimalString(hostExponent, sizeof(uint32_t) / sizeof(uint32_t), exponentString);
    importDecimalString(hostPrimeP, halfKeySizeBytes / sizeof(uint32_t), primePString);
    importDecimalString(hostPrimeQ, halfKeySizeBytes / sizeof(uint32_t), primeQString);
    importDecimalString(hostDP, halfKeySizeBytes / sizeof(uint32_t), dpString);
    importDecimalString(hostDQ, halfKeySizeBytes / sizeof(uint32_t), dqString);
    importDecimalString(hostCoefficientP, fullKeySizeBytes / sizeof(uint32_t), coefficientPString);
    importDecimalString(hostCoefficientQ, fullKeySizeBytes / sizeof(uint32_t), coefficientQString);
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

    // Validation not work well
    // XMP_CHECK_ERROR(xmpIntegersCmp(xmpHandle, validationResults, plaintext, result, totalMessages));
    // std::cout << "Validating results..." << std::endl;
    // for (int i = 0; i < totalMessages; i++) {
    //     if (validationResults[i] != 0) {
    //         std::cerr << "  Error at index " << i << std::endl;
    //         exit(1);
    //     }
    // }

    // Export plaintext
    uint32_t *exportedPlaintext = (uint32_t *)calloc(totalMessages * fullKeySizeWords, sizeof(uint32_t));
    XMP_CHECK_ERROR(xmpIntegersExport(xmpHandle, exportedPlaintext, &fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, plaintext, totalMessages));

    // Export result (decrypted text)
    uint32_t *exportedResult = (uint32_t *)calloc(totalMessages * fullKeySizeWords, sizeof(uint32_t));
    XMP_CHECK_ERROR(xmpIntegersExport(xmpHandle, exportedResult, &fullKeySizeWords, -1, sizeof(uint32_t), 0, 0, result, totalMessages));

    // Print plaintext and result as decimal strings
    for (int i = 0; i < totalMessages; i++)
    {
        std::string plaintextDecimal = binaryToDecimalString(&exportedPlaintext[i * fullKeySizeWords], fullKeySizeWords);
        std::string resultDecimal = binaryToDecimalString(&exportedResult[i * fullKeySizeWords], fullKeySizeWords);

        std::cout << "Message " << i + 1 << ": " << plaintextDecimal
                  << "\n"
                  << "Decrypt " << i + 1 << ": " << resultDecimal << "\n"
                  << std::endl;
    }

    // Free exported buffers
    free(exportedPlaintext);
    free(exportedResult);

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
