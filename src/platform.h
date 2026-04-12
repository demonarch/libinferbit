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
#include <sys/stat.h>
#include <sys/types.h>

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

#define ib_mmap    mmap
#define ib_munmap  munmap
#define ib_open    open
#define ib_read    read
#define ib_close   close
#define ib_fstat   fstat
#define ib_stat    stat
#define ib_struct_stat struct stat
#define ib_clock_gettime clock_gettime

#define IB_API

#endif /* _WIN32 */

#endif /* IB_PLATFORM_H */
