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

int main() {
    const int count = 100000;
    const int bits = 2048;
    const int hbits = bits / 2;
    double start, end;

    uint32_t limbs = bits / 8 / sizeof(uint32_t);
    uint32_t hlimbs = (limbs + 1) / 2; // +1 to round up
    size_t bytes = limbs * sizeof(uint32_t);
    size_t hbytes = bytes / 2;

    xmpHandle_t handle;

    xmpIntegers_t n, e;       // public key
    xmpIntegers_t p, q, d;    // private key
    xmpIntegers_t m, c, r;    // messages
    xmpIntegers_t dp, dq, cp, cq;
    xmpIntegers_t cm, mp, mq, sq, sp; // temporaries

    // Allocate host data for initialization
    uint32_t *h_n = (uint32_t*)calloc(1, bytes);
    uint32_t *h_e = (uint32_t*)calloc(1, 4);
    uint32_t *h_p = (uint32_t*)calloc(1, hbytes);
    uint32_t *h_q = (uint32_t*)calloc(1, hbytes);
    uint32_t *h_m = (uint32_t*)calloc(count, bytes);
    uint32_t *h_dp = (uint32_t*)calloc(1, hbytes);
    uint32_t *h_dq = (uint32_t*)calloc(1, hbytes);
    uint32_t *h_cp = (uint32_t*)calloc(1, bytes);
    uint32_t *h_cq = (uint32_t*)calloc(1, bytes);
    int32_t *results = (int32_t*)calloc(count, sizeof(int32_t));

    XMP_CHECK_ERROR(xmpHandleCreate(&handle));

    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &n, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &e, 32, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &p, hbits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &q, hbits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &d, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &cp, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &cq, bits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &m, bits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &c, bits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &r, bits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &sp, bits + hbits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &sq, bits + hbits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &mp, hbits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &mq, hbits, count));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &dp, hbits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &dq, hbits, 1));
    XMP_CHECK_ERROR(xmpIntegersCreate(handle, &cm, hbits, count));

    // Hardcoded keys for testing
    h_n[0] = 17460671;
    h_e[0] = 65537;
    h_p[0] = 4931;
    h_q[0] = 3541;
    h_dp[0] = 1063;
    h_dq[0] = 113;
    h_cp[0] = 15212136;
    h_cq[0] = 2248536;

    for (int i = 0; i < count; i++) {
        h_m[i * limbs] = rand32() % h_n[0];
    }

    XMP_CHECK_ERROR(xmpIntegersImport(handle, n, limbs, -1, sizeof(uint32_t), 0, 0, h_n, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, e, 1, -1, sizeof(uint32_t), 0, 0, h_e, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, p, hlimbs, -1, sizeof(uint32_t), 0, 0, h_p, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, q, hlimbs, -1, sizeof(uint32_t), 0, 0, h_q, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, m, limbs, -1, sizeof(uint32_t), 0, 0, h_m, count));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, dp, hlimbs, -1, sizeof(uint32_t), 0, 0, h_dp, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, dq, hlimbs, -1, sizeof(uint32_t), 0, 0, h_dq, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, cp, limbs, -1, sizeof(uint32_t), 0, 0, h_cp, 1));
    XMP_CHECK_ERROR(xmpIntegersImport(handle, cq, limbs, -1, sizeof(uint32_t), 0, 0, h_cq, 1));

    start = wallclock();
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, c, m, e, n, count));
    end = wallclock();
    std::cout << "Encryption time: " << end - start << ", " << bits << " bit throughput: " << count / (end - start) << " encryptions/second" << std::endl;

    start = wallclock();
    XMP_CHECK_ERROR(xmpIntegersMod(handle, cm, c, p, count));
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, mp, cm, dp, p, count));
    XMP_CHECK_ERROR(xmpIntegersMod(handle, cm, c, q, count));
    XMP_CHECK_ERROR(xmpIntegersPowm(handle, mq, cm, dq, q, count));
    XMP_CHECK_ERROR(xmpIntegersMul(handle, sp, mp, cp, count));
    XMP_CHECK_ERROR(xmpIntegersMul(handle, sq, mq, cq, count));
    XMP_CHECK_ERROR(xmpIntegersAdd(handle, sp, sp, sq, count));
    XMP_CHECK_ERROR(xmpIntegersMod(handle, r, sp, n, count));
    end = wallclock();
    std::cout << "Decryption time: " << end - start << ", " << bits << " bit throughput: " << count / (end - start) << " decryptions/second" << std::endl;

    XMP_CHECK_ERROR(xmpIntegersCmp(handle, results, m, r, count));
    std::cout << "Validating results..." << std::endl;
    for (int i = 0; i < count; i++) {
        if (results[i] != 0) {
            std::cerr << "  Error at index " << i << std::endl;
            exit(1);
        }
    }

    // Free resources
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, n));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, e));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, p));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, q));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, m));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, c));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, cm));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, sp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, sq));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, dp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, dq));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, cp));
    XMP_CHECK_ERROR(xmpIntegersDestroy(handle, cq));
    XMP_CHECK_ERROR(xmpHandleDestroy(handle));

    free(h_n);
    free(h_e);
    free(h_p);
    free(h_q);
    free(h_m);
    free(h_dp);
    free(h_dq);
    free(h_cp);
    free(h_cq);
    free(results);

    std::cout << "CRT RSA executed successfully" << std::endl;
    return 0;
}
