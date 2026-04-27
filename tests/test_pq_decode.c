/*
 * test_pq_decode — CLI harness for IBF v5 PQ decoder.
 *
 * Modes:
 *   reconstruct <in.ibf> <out.bin>
 *       Materialize the tensor as FP16 [M*N] LE and write to out.bin.
 *       Header: 8 bytes  M (int32) N (int32).
 *
 *   matmul <in.ibf> <x.bin> <out.bin>
 *       Read x as FP32 [N] from x.bin, run fused matmul, write FP32 [M]
 *       to out.bin. Header: 4 bytes  M (int32).
 *
 *   info <in.ibf>
 *       Print metadata (no IO).
 */
#include "pq_decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int cmd_info(const char* path) {
    ib_pq_tensor t;
    if (ib_pq_load_single(path, &t) != 0) { fprintf(stderr, "load failed\n"); return 1; }
    fprintf(stdout, "format=%d M=%d N=%d G=%d K=%d n_levels=%d rotate=%d n_outlier=%d C=%d\n",
            (int)t.format, t.M, t.N, t.G, t.K, t.n_levels, t.rotate, t.n_outlier, t.C);
    ib_pq_free(&t);
    return 0;
}

static int cmd_reconstruct(const char* in_path, const char* out_path) {
    ib_pq_tensor t;
    if (ib_pq_load_single(in_path, &t) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    size_t total = (size_t)t.M * (size_t)t.N;
    uint16_t* out = (uint16_t*)malloc(total * sizeof(uint16_t));
    if (!out) { ib_pq_free(&t); return 1; }
    if (ib_pq_reconstruct_fp16(&t, out) != 0) {
        fprintf(stderr, "reconstruct failed\n"); free(out); ib_pq_free(&t); return 1;
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) { perror("fopen"); free(out); ib_pq_free(&t); return 1; }
    int32_t M = t.M, N = t.N;
    fwrite(&M, sizeof(int32_t), 1, f);
    fwrite(&N, sizeof(int32_t), 1, f);
    fwrite(out, sizeof(uint16_t), total, f);
    fclose(f);

    free(out);
    ib_pq_free(&t);
    return 0;
}

static int cmd_matmul(const char* in_path, const char* x_path, const char* out_path) {
    ib_pq_tensor t;
    if (ib_pq_load_single(in_path, &t) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    float* x = (float*)malloc((size_t)t.N * sizeof(float));
    if (!x) { ib_pq_free(&t); return 1; }
    FILE* xf = fopen(x_path, "rb");
    if (!xf) { perror("fopen x"); free(x); ib_pq_free(&t); return 1; }
    if (fread(x, sizeof(float), (size_t)t.N, xf) != (size_t)t.N) {
        fprintf(stderr, "x read short\n"); fclose(xf); free(x); ib_pq_free(&t); return 1;
    }
    fclose(xf);

    float* y = (float*)malloc((size_t)t.M * sizeof(float));
    if (!y) { free(x); ib_pq_free(&t); return 1; }
    if (ib_pq_matmul_fp32(&t, x, y) != 0) {
        fprintf(stderr, "matmul failed\n"); free(x); free(y); ib_pq_free(&t); return 1;
    }

    FILE* of = fopen(out_path, "wb");
    if (!of) { perror("fopen out"); free(x); free(y); ib_pq_free(&t); return 1; }
    int32_t M = t.M;
    fwrite(&M, sizeof(int32_t), 1, of);
    fwrite(y, sizeof(float), (size_t)t.M, of);
    fclose(of);

    free(x); free(y);
    ib_pq_free(&t);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && strcmp(argv[1], "info") == 0) {
        return cmd_info(argv[2]);
    }
    if (argc >= 4 && strcmp(argv[1], "reconstruct") == 0) {
        return cmd_reconstruct(argv[2], argv[3]);
    }
    if (argc >= 5 && strcmp(argv[1], "matmul") == 0) {
        return cmd_matmul(argv[2], argv[3], argv[4]);
    }
    fprintf(stderr,
        "usage:\n"
        "  test_pq_decode info <in.ibf>\n"
        "  test_pq_decode reconstruct <in.ibf> <out.bin>\n"
        "  test_pq_decode matmul <in.ibf> <x.bin> <out.bin>\n"
    );
    return 2;
}
