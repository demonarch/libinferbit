#include "inferbit_internal.h"

ib_simd_level ib_detect_simd(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    /* ARM64 always has NEON */
    return IB_SIMD_NEON;
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;

    #if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    #else
    return IB_SIMD_NONE;
    #endif

    if (ebx & (1 << 16)) return IB_SIMD_AVX512;
    if (ebx & (1 << 5))  return IB_SIMD_AVX2;

    return IB_SIMD_NONE;
#else
    return IB_SIMD_NONE;
#endif
}
