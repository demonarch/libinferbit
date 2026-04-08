/*
 * avx2.c — AVX2-optimized kernels for x86_64
 *
 * Compiled only on x86_64 with -mavx2 -mfma.
 * These replace the scalar fallbacks via the dispatch table.
 */

#if defined(__x86_64__) || defined(_M_X64)

#include "../inferbit_internal.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

/* ── INT8 matmul: out[M] = weights[M,N] @ input[N] ─────────── */

static void avx2_matmul_int8(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const int8_t* w = (const int8_t*)weights;

    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int j = 0;
        /* Process 16 elements per iteration (2 x 8) */
        for (; j + 15 < N; j += 16) {
            /* Load 16 INT8 weights */
            __m128i w8_lo = _mm_loadl_epi64((const __m128i*)(row + j));
            __m128i w8_hi = _mm_loadl_epi64((const __m128i*)(row + j + 8));

            /* Sign-extend INT8 → INT32 */
            __m256i w32_lo = _mm256_cvtepi8_epi32(w8_lo);
            __m256i w32_hi = _mm256_cvtepi8_epi32(w8_hi);

            /* Convert to float */
            __m256 wf_lo = _mm256_cvtepi32_ps(w32_lo);
            __m256 wf_hi = _mm256_cvtepi32_ps(w32_hi);

            /* Load 16 input floats */
            __m256 in_lo = _mm256_loadu_ps(input + j);
            __m256 in_hi = _mm256_loadu_ps(input + j + 8);

            /* FMA: acc += weight * input */
            acc0 = _mm256_fmadd_ps(wf_lo, in_lo, acc0);
            acc1 = _mm256_fmadd_ps(wf_hi, in_hi, acc1);
        }

        /* Horizontal sum */
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi128 = _mm256_extractf128_ps(sum, 1);
        __m128 lo128 = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        /* Scalar tail */
        for (; j < N; j++) {
            result += (float)row[j] * input[j];
        }

        out[i] = result * scales[i];
    }
}

/* ── INT4 matmul: out[M] = weights[M,N] @ input[N] ─────────── */

static void avx2_matmul_int4(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const __m256i mask_lo = _mm256_set1_epi8(0x0F);
    const __m256i bias = _mm256_set1_epi8(8);

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int j = 0;
        /* Process 16 input elements per iteration (8 packed bytes → 16 INT4 values) */
        for (; j + 15 < N; j += 16) {
            /* Load 8 bytes = 16 INT4 values */
            __m128i packed = _mm_loadl_epi64((const __m128i*)(row + j / 2));
            __m256i packed256 = _mm256_cvtepu8_epi16(packed);

            /* Extract low and high nibbles */
            /* Low nibble: packed & 0x0F */
            __m256i lo_16 = _mm256_and_si256(packed256, _mm256_set1_epi16(0x0F));
            /* High nibble: (packed >> 4) & 0x0F */
            __m256i hi_16 = _mm256_and_si256(_mm256_srli_epi16(packed256, 4), _mm256_set1_epi16(0x0F));

            /* Interleave: lo0, hi0, lo1, hi1, ... → need to deinterleave */
            /* Actually, nibbles are packed as [lo|hi] per byte, so for input indices:
             * byte[k] low nibble  → input[2k]
             * byte[k] high nibble → input[2k+1]
             *
             * We have 8 bytes → 8 lo values and 8 hi values
             * We need: v[0]=lo[0], v[1]=hi[0], v[2]=lo[1], v[3]=hi[1], ...
             */

            /* Simpler approach: unpack to 16 int32s */
            /* Low nibbles → even indices, high nibbles → odd indices */
            /* Process in two groups of 8 */

            /* First 4 bytes → 8 values */
            int32_t vals[16];
            for (int k = 0; k < 8; k++) {
                uint8_t byte = row[j / 2 + k];
                vals[k * 2]     = (int32_t)(byte & 0x0F) - 8;
                vals[k * 2 + 1] = (int32_t)((byte >> 4) & 0x0F) - 8;
            }

            __m256 wf_lo = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)vals));
            __m256 wf_hi = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(vals + 8)));

            __m256 in_lo = _mm256_loadu_ps(input + j);
            __m256 in_hi = _mm256_loadu_ps(input + j + 8);

            acc0 = _mm256_fmadd_ps(wf_lo, in_lo, acc0);
            acc1 = _mm256_fmadd_ps(wf_hi, in_hi, acc1);
        }

        /* Horizontal sum */
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi128 = _mm256_extractf128_ps(sum, 1);
        __m128 lo128 = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);

        /* Scalar tail */
        for (; j < N; j += 2) {
            uint8_t byte = row[j / 2];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            result += (float)v0 * input[j];
            if (j + 1 < N) result += (float)v1 * input[j + 1];
        }

        out[i] = result * scales[i];
    }
}

/* ── RMSNorm ────────────────────────────────────────────────── */

static void avx2_rmsnorm(
    float* out, const float* input, const float* weight,
    float eps, int N
) {
    /* Compute sum of squares */
    __m256 ss_acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        ss_acc = _mm256_fmadd_ps(v, v, ss_acc);
    }
    /* Horizontal sum */
    __m128 hi = _mm256_extractf128_ps(ss_acc, 1);
    __m128 lo = _mm256_castps256_ps128(ss_acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float ss = _mm_cvtss_f32(sum128);
    /* Scalar tail */
    for (; i < N; i++) ss += input[i] * input[i];

    float scale = 1.0f / sqrtf(ss / (float)N + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    /* Apply normalization */
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 w = _mm256_loadu_ps(weight + i);
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(v, vscale), w);
        _mm256_storeu_ps(out + i, r);
    }
    for (; i < N; i++) {
        out[i] = input[i] * scale * weight[i];
    }
}

/* ── Softmax ────────────────────────────────────────────────── */

static void avx2_softmax(float* data, int N) {
    /* Find max */
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    /* Horizontal max */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m128 = _mm_max_ps(lo, hi);
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(2, 3, 0, 1)));
    m128 = _mm_max_ps(m128, _mm_shuffle_ps(m128, m128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(m128);
    for (; i < N; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    /* exp(x - max) and sum — scalar for correctness (expf is hard to vectorize well) */
    float sum = 0.0f;
    for (i = 0; i < N; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    /* Normalize */
    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vinv));
    }
    float inv = 1.0f / sum;
    for (; i < N; i++) {
        data[i] *= inv;
    }
}

/* ── SiLU multiply ──────────────────────────────────────────── */

static void avx2_silu_mul(float* out, const float* gate, const float* up, int N) {
    /* SiLU(x) = x / (1 + exp(-x)) — use scalar expf, vectorize the rest */
    int i = 0;
    for (; i + 7 < N; i += 8) {
        /* Compute SiLU scalar (expf not easily vectorized) */
        float silu_vals[8];
        for (int k = 0; k < 8; k++) {
            float x = gate[i + k];
            silu_vals[k] = x / (1.0f + expf(-x));
        }
        __m256 vs = _mm256_loadu_ps(silu_vals);
        __m256 vu = _mm256_loadu_ps(up + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vs, vu));
    }
    for (; i < N; i++) {
        float x = gate[i];
        out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}

/* ── RoPE ───────────────────────────────────────────────────── */

/* RoPE is hard to vectorize well due to sin/cos, keep scalar */
static void avx2_rope(
    float* q, float* k, int head_dim, int pos, float theta
) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_a - q1 * sin_a;
        q[i + 1] = q0 * sin_a + q1 * cos_a;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_a - k1 * sin_a;
        k[i + 1] = k0 * sin_a + k1 * cos_a;
    }
}

/* ── Registration ───────────────────────────────────────────── */

void ib_init_kernels_avx2(ib_kernels* kern) {
    kern->matmul_int4 = avx2_matmul_int4;
    kern->matmul_int8 = avx2_matmul_int8;
    kern->rmsnorm     = avx2_rmsnorm;
    kern->rope        = avx2_rope;
    kern->softmax     = avx2_softmax;
    kern->silu_mul    = avx2_silu_mul;
}

#else
/* Not x86_64 — provide stub */
#include "../inferbit_internal.h"
void ib_init_kernels_avx2(ib_kernels* kern) { (void)kern; }
#endif
