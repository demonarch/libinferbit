/*
 * platform.h — Cross-platform abstractions for mmap, threading, and shared library export
 */

#ifndef IB_PLATFORM_H
#define IB_PLATFORM_H

#ifdef _WIN32

/* ── Windows ────────────────────────────────────────────────── */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>      /* SEEK_SET / SEEK_CUR for the pread shim */
#include <stdlib.h>     /* malloc — backs the aligned_alloc shim */
#include <sys/stat.h>
#include <sys/types.h>

/* MSVC has no C11 aligned_alloc. The 64-byte alignment in this codebase
 * is a SIMD-throughput hint, not a correctness requirement (and the
 * NEON kernels that care aren't built on Windows anyway), so fall back
 * to plain malloc — which keeps the matching free() calls valid. */
#define aligned_alloc(alignment, size) malloc(size)

/* ssize_t doesn't exist on Windows */
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

/* mmap emulation */
#define PROT_READ     0x1
#define MAP_PRIVATE   0x02
#define MAP_FAILED    ((void*)-1)

static inline void* ib_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    (void)addr; (void)prot; (void)flags; (void)offset;
    HANDLE fh = (HANDLE)_get_osfhandle(fd);
    if (fh == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE mapping = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) return MAP_FAILED;
    void* ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    return ptr ? ptr : MAP_FAILED;
}

static inline int ib_munmap(void* addr, size_t length) {
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

/* Use _open/_read/_close/_fstat on Windows */
#define ib_open   _open
#define ib_read   _read
#define ib_close  _close
#define ib_fstat  _fstat
#define ib_stat   _stat
#define ib_struct_stat struct _stat

/* S_ISDIR doesn't exist on MSVC */
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

/* unlink → _unlink on Windows */
#define unlink _unlink

/* O_RDONLY is in fcntl.h */

/* clock_gettime emulation */
#include <time.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int ib_clock_gettime(int clk, struct timespec* ts) {
    (void)clk;
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    ts->tv_sec = (time_t)(count.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)((count.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
}

/* ── POSIX I/O shims for the drive-mode (Path D) code ───────────
 * Drive mode (residency_mode=1) is a POSIX-only streaming optimisation
 * and is never enabled on Windows — these shims only have to compile +
 * link so the shared drive-mode code paths build. */
#include <string.h>

static inline ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
    __int64 cur = _lseeki64(fd, 0, SEEK_CUR);
    if (cur < 0) return -1;
    if (_lseeki64(fd, (__int64)offset, SEEK_SET) < 0) return -1;
    int r = _read(fd, buf, (unsigned int)count);
    _lseeki64(fd, cur, SEEK_SET);
    return (ssize_t)r;
}

#define ib_write _write

/* Only _SC_PAGESIZE is used; map sysconf() to the Win32 page size. */
#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 1
#endif
static inline long ib_win_pagesize(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwPageSize;
}
#define sysconf(name) ib_win_pagesize()

static inline int mkstemp(char* tmpl) {
    if (_mktemp_s(tmpl, strlen(tmpl) + 1) != 0) return -1;
    int fd = -1;
    if (_sopen_s(&fd, tmpl, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY,
                 _SH_DENYNO, _S_IREAD | _S_IWRITE) != 0)
        return -1;
    return fd;
}

/* Online logical CPU count, clamped to [1,64] with a safe fallback of 4. */
static inline int ib_hardware_concurrency(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int)si.dwNumberOfProcessors;
    if (n <= 0) return 4;
    if (n > 64) return 64;
    return n;
}

/* Thread-local storage */
#define _Thread_local __declspec(thread)

/* DLL export */
#ifdef INFERBIT_BUILD_DLL
#define IB_API __declspec(dllexport)
#else
#define IB_API __declspec(dllimport)
#endif

#else

/* ── POSIX (macOS, Linux) ───────────────────────────────────── */

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>   /* sysctlbyname — Apple Silicon P-core query */
#endif

#define ib_mmap    mmap
#define ib_munmap  munmap
#define ib_open    open
#define ib_read    read
#define ib_write   write
#define ib_close   close
#define ib_fstat   fstat
#define ib_stat    stat
#define ib_struct_stat struct stat
#define ib_clock_gettime clock_gettime

/* Online logical CPU count, clamped to [1,64] with a safe fallback of 4.
 * On Apple Silicon, prefers the performance-core count instead. */
static inline int ib_hardware_concurrency(void) {
    long n = -1;
#if defined(__APPLE__)
    /* Apple Silicon: prefer the performance-core count — E-cores are
     * far slower for matmul and create scheduling stragglers. Falls
     * back to the total online count on Intel Macs / older macOS. */
    {
        int pcores = 0;
        size_t sz = sizeof(pcores);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &pcores, &sz, NULL, 0) == 0
            && pcores >= 1) {
            n = pcores;
        }
    }
#endif
    if (n < 1) n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) return 4;
    if (n > 64) return 64;
    return (int)n;
}

#define IB_API

#endif /* _WIN32 */

#endif /* IB_PLATFORM_H */
