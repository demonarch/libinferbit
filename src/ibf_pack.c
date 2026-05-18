/*
 * ibf_pack.c — Distribution-time compression for .ibf files.
 *
 * See ibf_pack.h. The runtime contract is non-negotiable: pack and
 * unpack are *out-of-band* utilities. The .ibf loader (ibf_loader.c)
 * never calls into this file; it mmaps a plain .ibf and that's it.
 *
 * Build wiring: CMake locates libzstd (find_package(zstd) or
 * pkg_check_modules(libzstd)); when found it defines IB_HAS_ZSTD on
 * this translation unit and links the resolved zstd target. When zstd
 * is absent the source still compiles — both entrypoints just return
 * -1 with a clear error.
 */

#include "ibf_pack.h"
#include "inferbit_internal.h"

#if defined(IB_HAS_ZSTD)
#include <zstd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(IB_HAS_ZSTD)

/* Long-range mode window. 27 = 128 MiB; matches the "--long=27"
 * hint in docs/v2/00_CORRECTION.md §5i.1. PQv2 indices reuse a
 * small alphabet across long spans of the file; a big window lets
 * zstd find those repeats. */
#define IB_PACK_WINDOW_LOG 27

/* Default compression level when caller passes <=0. zstd defines
 * the maximum at ZSTD_maxCLevel() == 22; level 19 is the standard
 * "release artifact" sweet-spot (fast enough on a build box, very
 * close to max ratio). */
#define IB_PACK_DEFAULT_LEVEL 19

static int pack_open_files(const char *in_path, const char *out_path,
                           FILE **fin, FILE **fout)
{
    *fin = fopen(in_path, "rb");
    if (!*fin) {
        ib_set_error("inferbit_pack: cannot open input '%s'", in_path);
        return -1;
    }
    *fout = fopen(out_path, "wb");
    if (!*fout) {
        fclose(*fin);
        *fin = NULL;
        ib_set_error("inferbit_pack: cannot open output '%s'", out_path);
        return -1;
    }
    return 0;
}

#endif /* IB_HAS_ZSTD */

int inferbit_pack(const char *ibf_in, const char *zst_out, int level)
{
#if !defined(IB_HAS_ZSTD)
    (void)ibf_in; (void)zst_out; (void)level;
    ib_set_error("inferbit_pack: zstd not available at build time");
    return -1;
#else
    if (!ibf_in || !zst_out) {
        ib_set_error("inferbit_pack: NULL path argument");
        return -1;
    }
    if (level <= 0) {
        level = IB_PACK_DEFAULT_LEVEL;
    }
    int max_level = ZSTD_maxCLevel();
    if (level > max_level) {
        level = max_level;
    }

    FILE *fin = NULL;
    FILE *fout = NULL;
    if (pack_open_files(ibf_in, zst_out, &fin, &fout) != 0) {
        return -1;
    }

    ZSTD_CCtx *cctx = ZSTD_createCCtx();
    if (!cctx) {
        fclose(fin);
        fclose(fout);
        ib_set_error("inferbit_pack: ZSTD_createCCtx failed");
        return -1;
    }

    int rc = 0;
    void *buf_in = NULL;
    void *buf_out = NULL;
    size_t r;

    r = ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    if (ZSTD_isError(r)) {
        ib_set_error("inferbit_pack: set level: %s", ZSTD_getErrorName(r));
        rc = -1;
        goto done;
    }
    /* Long-range mode: bump the window log AND enable the long-range
     * matcher. windowLog alone widens the back-reference range; the
     * long matcher is the secondary table that actually finds far
     * matches efficiently. */
    r = ZSTD_CCtx_setParameter(cctx, ZSTD_c_windowLog, IB_PACK_WINDOW_LOG);
    if (ZSTD_isError(r)) {
        ib_set_error("inferbit_pack: set windowLog: %s", ZSTD_getErrorName(r));
        rc = -1;
        goto done;
    }
    r = ZSTD_CCtx_setParameter(cctx, ZSTD_c_enableLongDistanceMatching, 1);
    if (ZSTD_isError(r)) {
        ib_set_error("inferbit_pack: enable LDM: %s", ZSTD_getErrorName(r));
        rc = -1;
        goto done;
    }

    size_t buf_in_sz = ZSTD_CStreamInSize();
    size_t buf_out_sz = ZSTD_CStreamOutSize();
    buf_in = malloc(buf_in_sz);
    buf_out = malloc(buf_out_sz);
    if (!buf_in || !buf_out) {
        ib_set_error("inferbit_pack: out of memory");
        rc = -1;
        goto done;
    }

    int finished = 0;
    while (!finished) {
        size_t read_n = fread(buf_in, 1, buf_in_sz, fin);
        if (ferror(fin)) {
            ib_set_error("inferbit_pack: read error on '%s'", ibf_in);
            rc = -1;
            goto done;
        }
        int last = feof(fin);
        ZSTD_inBuffer input = { buf_in, read_n, 0 };
        ZSTD_EndDirective mode = last ? ZSTD_e_end : ZSTD_e_continue;
        int more;
        do {
            ZSTD_outBuffer output = { buf_out, buf_out_sz, 0 };
            size_t remaining = ZSTD_compressStream2(cctx, &output, &input, mode);
            if (ZSTD_isError(remaining)) {
                ib_set_error("inferbit_pack: zstd error %s",
                             ZSTD_getErrorName(remaining));
                rc = -1;
                goto done;
            }
            if (output.pos > 0) {
                if (fwrite(buf_out, 1, output.pos, fout) != output.pos) {
                    ib_set_error("inferbit_pack: write error on '%s'", zst_out);
                    rc = -1;
                    goto done;
                }
            }
            /* When flushing the final frame, keep looping until the
             * encoder reports 0 bytes still to flush. Mid-stream,
             * loop until our input buffer has been fully consumed. */
            more = last ? (remaining > 0) : (input.pos < input.size);
        } while (more);
        finished = last;
    }

done:
    free(buf_in);
    free(buf_out);
    ZSTD_freeCCtx(cctx);
    fclose(fin);
    if (fout) {
        fclose(fout);
    }
    if (rc != 0) {
        /* Best-effort cleanup of the partial output. Ignore unlink
         * failure — the error context already names the file. */
        remove(zst_out);
    }
    return rc;
#endif
}

int inferbit_unpack(const char *zst_in, const char *ibf_out)
{
#if !defined(IB_HAS_ZSTD)
    (void)zst_in; (void)ibf_out;
    ib_set_error("inferbit_unpack: zstd not available at build time");
    return -1;
#else
    if (!zst_in || !ibf_out) {
        ib_set_error("inferbit_unpack: NULL path argument");
        return -1;
    }

    FILE *fin = fopen(zst_in, "rb");
    if (!fin) {
        ib_set_error("inferbit_unpack: cannot open input '%s'", zst_in);
        return -1;
    }
    FILE *fout = fopen(ibf_out, "wb");
    if (!fout) {
        fclose(fin);
        ib_set_error("inferbit_unpack: cannot open output '%s'", ibf_out);
        return -1;
    }

    ZSTD_DCtx *dctx = ZSTD_createDCtx();
    if (!dctx) {
        fclose(fin);
        fclose(fout);
        ib_set_error("inferbit_unpack: ZSTD_createDCtx failed");
        return -1;
    }

    int rc = 0;
    void *buf_in = NULL;
    void *buf_out = NULL;
    size_t r;

    /* Match the encoder's windowLog. ZSTD_d_windowLogMax must be at
     * least the encoder's windowLog or decompression refuses to run
     * on overly-large windows. 27 == 128 MiB; harmless on small
     * frames. */
    r = ZSTD_DCtx_setParameter(dctx, ZSTD_d_windowLogMax, IB_PACK_WINDOW_LOG);
    if (ZSTD_isError(r)) {
        ib_set_error("inferbit_unpack: set windowLogMax: %s",
                     ZSTD_getErrorName(r));
        rc = -1;
        goto done;
    }

    size_t buf_in_sz = ZSTD_DStreamInSize();
    size_t buf_out_sz = ZSTD_DStreamOutSize();
    buf_in = malloc(buf_in_sz);
    buf_out = malloc(buf_out_sz);
    if (!buf_in || !buf_out) {
        ib_set_error("inferbit_unpack: out of memory");
        rc = -1;
        goto done;
    }

    size_t last_ret = 0;
    int saw_input = 0;
    for (;;) {
        size_t read_n = fread(buf_in, 1, buf_in_sz, fin);
        if (ferror(fin)) {
            ib_set_error("inferbit_unpack: read error on '%s'", zst_in);
            rc = -1;
            goto done;
        }
        if (read_n == 0) {
            break;
        }
        saw_input = 1;
        ZSTD_inBuffer input = { buf_in, read_n, 0 };
        while (input.pos < input.size) {
            ZSTD_outBuffer output = { buf_out, buf_out_sz, 0 };
            size_t ret = ZSTD_decompressStream(dctx, &output, &input);
            if (ZSTD_isError(ret)) {
                ib_set_error("inferbit_unpack: zstd error %s",
                             ZSTD_getErrorName(ret));
                rc = -1;
                goto done;
            }
            if (output.pos > 0) {
                if (fwrite(buf_out, 1, output.pos, fout) != output.pos) {
                    ib_set_error("inferbit_unpack: write error on '%s'",
                                 ibf_out);
                    rc = -1;
                    goto done;
                }
            }
            last_ret = ret;
        }
    }

    /* When decode is complete zstd returns 0 from the last call. A
     * non-zero return at EOF means the stream was truncated. */
    if (!saw_input) {
        ib_set_error("inferbit_unpack: input '%s' is empty", zst_in);
        rc = -1;
        goto done;
    }
    if (last_ret != 0) {
        ib_set_error("inferbit_unpack: truncated zstd stream (last_ret=%zu)",
                     last_ret);
        rc = -1;
        goto done;
    }

done:
    free(buf_in);
    free(buf_out);
    ZSTD_freeDCtx(dctx);
    fclose(fin);
    fclose(fout);
    if (rc != 0) {
        remove(ibf_out);
    }
    return rc;
#endif
}
