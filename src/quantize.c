/*
 * quantize.c — Weight quantization (FP16/FP32 → INT4/INT8)
 *
 * Per-channel quantization: each output row gets its own FP16 scale factor.
 * scale = max(abs(row)) / max_int_val
 * quantized[i] = round(value[i] / scale)
 */

#include "inferbit_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── FP16 helpers ───────────────────────────────────────────── */

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float f; uint32_t b = sign; memcpy(&f, &b, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t b = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &b, 4); return f;
    }
    uint32_t b = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &b, 4); return f;
}

static uint16_t f32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000);
    int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = b & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

/* ── BF16 helper ────────────────────────────────────────────── */

static float bf16_to_f32(uint16_t h) {
    uint32_t b = (uint32_t)h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

/* ── Read a row of source weights as FP32 ───────────────────── */

static void read_row_fp32(float* out, const void* data, const char* dtype, int cols, int row) {
    if (strcmp(dtype, "F32") == 0) {
        const float* src = (const float*)data + (size_t)row * cols;
        memcpy(out, src, cols * sizeof(float));
    } else if (strcmp(dtype, "F16") == 0) {
        const uint8_t* base = (const uint8_t*)data + (size_t)row * cols * 2;
        for (int i = 0; i < cols; i++) {
            uint16_t v; memcpy(&v, base + i * 2, 2);
            out[i] = fp16_to_f32(v);
        }
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint8_t* base = (const uint8_t*)data + (size_t)row * cols * 2;
        for (int i = 0; i < cols; i++) {
            uint16_t v; memcpy(&v, base + i * 2, 2);
            out[i] = bf16_to_f32(v);
        }
    }
}

/* ── Quantize a matrix to INT8 ──────────────────────────────── */

void ib_quantize_int8(
    int8_t* out_weights,       /* [rows * cols] */
    uint16_t* out_scales,      /* [rows] FP16 */
    const void* src_data,
    const char* src_dtype,
    int rows, int cols
) {
    float* row_buf = malloc(cols * sizeof(float));
    if (!row_buf) return;

    for (int r = 0; r < rows; r++) {
        read_row_fp32(row_buf, src_data, src_dtype, cols, r);

        /* Find max absolute value */
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(row_buf[c]);
            if (a > max_abs) max_abs = a;
        }

        /* Compute scale: map [-max_abs, max_abs] to [-127, 127] */
        float scale = max_abs / 127.0f;
        /* Clamp to smallest normal FP16 (2^-14 ≈ 6.1e-5) to avoid underflow */
        if (scale < 6.104e-05f) scale = 6.104e-05f;
        float inv_scale = 1.0f / scale;

        out_scales[r] = f32_to_fp16(scale);

        /* Quantize */
        int8_t* dst = out_weights + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            float v = row_buf[c] * inv_scale;
            int32_t q = (int32_t)roundf(v);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            dst[c] = (int8_t)q;
        }
    }

    free(row_buf);
}

/* ── Quantize a matrix to INT4 ──────────────────────────────── */

void ib_quantize_int4(
    uint8_t* out_weights,      /* [rows * cols / 2] packed nibbles */
    uint16_t* out_scales,      /* [rows] FP16 */
    const void* src_data,
    const char* src_dtype,
    int rows, int cols
) {
    float* row_buf = malloc(cols * sizeof(float));
    if (!row_buf) return;

    for (int r = 0; r < rows; r++) {
        read_row_fp32(row_buf, src_data, src_dtype, cols, r);

        /* Find max absolute value */
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(row_buf[c]);
            if (a > max_abs) max_abs = a;
        }

        /* Scale: map to [-7, 7] (stored as [0, 15] with bias 8) */
        float scale = max_abs / 7.0f;
        if (scale < 6.104e-05f) scale = 6.104e-05f;
        float inv_scale = 1.0f / scale;

        out_scales[r] = f32_to_fp16(scale);

        /* Quantize and pack nibbles */
        uint8_t* dst = out_weights + (size_t)r * (cols / 2);
        for (int c = 0; c < cols; c += 2) {
            float v0 = row_buf[c] * inv_scale;
            float v1 = (c + 1 < cols) ? row_buf[c + 1] * inv_scale : 0.0f;

            int32_t q0 = (int32_t)roundf(v0);
            int32_t q1 = (int32_t)roundf(v1);
            if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
            if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;

            /* Store with bias 8: [-7,7] → [1,15] (0 maps to 8) */
            uint8_t lo = (uint8_t)(q0 + 8) & 0x0F;
            uint8_t hi = (uint8_t)(q1 + 8) & 0x0F;
            dst[c / 2] = lo | (hi << 4);
        }
    }

    free(row_buf);
}

/* ── Copy FP16 norm weights (no quantization) ───────────────── */

void ib_copy_norm_fp16(
    uint16_t* out,
    const void* src_data,
    const char* src_dtype,
    int size
) {
    if (strcmp(src_dtype, "F16") == 0) {
        memcpy(out, src_data, size * sizeof(uint16_t));
    } else if (strcmp(src_dtype, "F32") == 0) {
        const float* src = (const float*)src_data;
        for (int i = 0; i < size; i++) {
            out[i] = f32_to_fp16(src[i]);
        }
    } else if (strcmp(src_dtype, "BF16") == 0) {
        const uint16_t* src = (const uint16_t*)src_data;
        for (int i = 0; i < size; i++) {
            out[i] = f32_to_fp16(bf16_to_f32(src[i]));
        }
    }
}
