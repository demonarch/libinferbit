/* test_pqv2_ibf — load an IBF v6 PQv2 file, run matmul on a named
 * tensor, and dump the output. The Python side compares.
 *
 * usage: test_pqv2_ibf <file.ibf> <tensor_name> <x.bin> <y_out.bin>
 */
#include "pqv2_format.h"
#include "pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <file.ibf> <tensor> <x.bin fp32> <y_out.bin fp32>\n",
                argv[0]);
        return 1;
    }
    ib_pqv2_file f;
    if (ib_pqv2_file_load(argv[1], &f) != 0) {
        fprintf(stderr, "load failed: %s\n", argv[1]);
        return 1;
    }
    printf("loaded %s: %d tensors, %.1f MB\n", argv[1], f.n_tensors,
            f._buffer_size / 1e6);

    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, argv[2]);
    if (!nt) { fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    if (nt->kind != IB_PQV2_KIND_PQV2) {
        fprintf(stderr, "tensor is not PQV2 kind\n"); return 1;
    }
    const pqv2_t *t = &nt->pq;
    printf("tensor: M=%u N=%u G=%u K=%u n_sub=%u half=%u l2_kind=%u\n",
            t->M, t->N, t->G, t->K, t->n_subchunks, t->half, t->l2_kind);

    /* Load x */
    FILE *xf = fopen(argv[3], "rb"); if (!xf) { perror(argv[3]); return 1; }
    float *x = malloc((size_t)t->N * sizeof(float));
    if (fread(x, sizeof(float), t->N, xf) != t->N) { return 1; }
    fclose(xf);

    /* Run kernel */
    float *y = calloc(t->M, sizeof(float));
    int N_ITER = 10;
    /* Warmup */
    if (t->K == 256) pqv2_matvec_tbl_int8_k256(t, x, y);
    else if (t->K == 128) pqv2_matvec_tbl_int8_k128(t, x, y);
    else if (t->K <= 64) pqv2_matvec_tbl_int8(t, x, y);
    else pqv2_matvec_lut(t, x, y);

    double t0 = now_s();
    for (int i = 0; i < N_ITER; i++) {
        if (t->K == 256) pqv2_matvec_tbl_int8_k256(t, x, y);
        else if (t->K == 128) pqv2_matvec_tbl_int8_k128(t, x, y);
        else if (t->K <= 64) pqv2_matvec_tbl_int8(t, x, y);
        else pqv2_matvec_lut(t, x, y);
    }
    double dt = (now_s() - t0) / N_ITER;
    printf("matvec: %.2f ms/iter (%.1f Gop/s)\n", dt * 1000,
            (double)t->M * t->N * 2 / dt / 1e9);

    /* Dump y */
    FILE *yf = fopen(argv[4], "wb"); if (!yf) { perror(argv[4]); return 1; }
    fwrite(y, sizeof(float), t->M, yf);
    fclose(yf);
    printf("wrote y: %u floats to %s\n", t->M, argv[4]);

    free(x); free(y);
    ib_pqv2_file_free(&f);
    return 0;
}
