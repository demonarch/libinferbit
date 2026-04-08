#include "inferbit_internal.h"
#include <stdarg.h>
#include <stdio.h>

#define IB_ERROR_BUF_SIZE 512

static _Thread_local char ib_error_buf[IB_ERROR_BUF_SIZE];
static _Thread_local int  ib_error_set = 0;

void ib_set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(ib_error_buf, IB_ERROR_BUF_SIZE, fmt, args);
    va_end(args);
    ib_error_set = 1;
}

const char* inferbit_last_error(void) {
    if (!ib_error_set) return NULL;
    ib_error_set = 0;
    return ib_error_buf;
}
