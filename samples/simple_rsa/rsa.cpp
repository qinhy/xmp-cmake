#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
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

// Error-checking macro for XMP
#define XMP_CHECK_ERROR(fun)                                      \
    {                                                             \
        xmpError_t error = fun;                                   \
        if (error != xmpErrorSuccess) {                           \
            if (error == xmpErrorCuda) {                          \
                std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) \
                          << " (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            } else {                                              \
                std::cerr << "XMP Error: " << xmpGetErrorString(error) \
                          << " (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            }                                                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

// Generate a random 32-bit integer
uint32_t rand32() {
    uint32_t lo = rand() & 0xffff;
    uint32_t hi = rand() & 0xffff;
    return (hi << 16) | lo;
}

int main() {
    const int count = 1000;
    const int bits = 2048;

    double start, end;

    uint32_t limbs = bits / 8 / sizeof(uint32_t);
    size_t bytes = limbs * sizeof(uint32_t);

    xmpHandle_t handle;

    // Public key (n, e), private key (d), and messages
    xmpIntegers_t n, e, d, m, c, r;

    // Host data for initialization
    std::vector<uint32_t> h_n(limbs, 0), h_e(1, 0), h_d(limbs, 0);
    std::vector<uint32_t> h_m(count * limbs, 0);
    std::vector<int32_t> res(count, 0);

    // Allocate handle
    XMP_CHECK_ERROR(xmpHandleCreate(&handle));

    // Allocate integers
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &n, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &e, 32, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &d, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &m, bits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &c, bits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &r, bits, count));

    // Hardcoded keys and messages for testing
    h_n[0] = 17460671;
    h_e[0] = 65537;
    h_d[0] = 16156673;

    for (int i = 0; i < count; ++i) {
        h_m[i * limbs] = rand32() % h_n[0];
    }

    // Import keys and messages into XMP
    XMP_CHECK_ERROR(xmpIntegersImport(handle, n, limbs, -1, sizeof(uint32_t), 0, 0, h_n.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, e, 1, -1, sizeof(uint32_t), 0, 0, h_e.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, d, limbs, -1, sizeof(uint32_t), 0, 0, h_d.data(), 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, m, limbs, -1, sizeof(uint32_t), 0, 0, h_m.data(), count));

    // Encryption: c = m^e mod n
    start = wallclock();
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, c, m, e, n, count));
    end = wallclock();
    std::cout << "Encryption time: " << (end - start)
              << ", " << bits << "-bit throughput: " << count / (end - start) << " encryptions/second\n";

    // Decryption: r = c^d mod n
    start = wallclock();
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, r, c, d, n, count));
    end = wallclock();
    std::cout << "Decryption time: " << (end - start)
              << ", " << bits << "-bit throughput: " << count / (end - start) << " decryptions/second\n";

    // Validation: compare r and m
    XMP_CHECK_ERROR(xmpIntegersCmp(handle, res.data(), m, r, count));

    std::cout << "Validating results...\n";
    for (int i = 0; i < count; ++i) {
        if (res[i] != 0) {
            std::cerr << "  Error at index " << i << "\n";
            exit(EXIT_FAILURE);
        }
    }

    // Free integers and handle
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, n));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, e));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, d));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, m));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, c));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, r));
    XMP_CHECK_ERROR(xmpHandleDestroy(handle));

    std::cout << "Simple RSA executed successfully\n";
    return 0;
}
