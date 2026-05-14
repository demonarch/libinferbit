/* MPSMatrixMultiplication GEMM ceiling benchmark.
 *
 * Measures TFLOPS achievable by Apple's MetalPerformanceShaders GEMM at
 * TinyLlama's prefill matmul shapes. Result is the upper bound on what
 * any kernel can reach on this hardware — comparing our hand-rolled
 * INT4/PQv2 kernels against this number tells us how much headroom is
 * left in the kernel axis (vs the rest of the pipeline).
 *
 * Usage: bench_mps_gemm [B [iters]]
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Run B×N x N×M (output is B×M) with fp16 weights / fp16 input,
 * iters times. Returns avg ms per matmul. */
static double bench_one(id<MTLDevice> dev,
                         id<MTLCommandQueue> q,
                         int B, int N, int M,
                         int iters)
{
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:B columns:N rowBytes:N*sizeof(uint16_t)
                         dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:N columns:M rowBytes:M*sizeof(uint16_t)
                         dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:B columns:M rowBytes:M*sizeof(uint16_t)
                         dataType:MPSDataTypeFloat16];

    id<MTLBuffer> bufA = [dev newBufferWithLength:(NSUInteger)B*N*sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [dev newBufferWithLength:(NSUInteger)N*M*sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [dev newBufferWithLength:(NSUInteger)B*M*sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared];
    /* Fill with non-zero data so the kernel doesn't get short-circuited. */
    uint16_t one = 0x3c00u; /* fp16 1.0 */
    uint16_t *aP = (uint16_t *)bufA.contents;
    uint16_t *bP = (uint16_t *)bufB.contents;
    for (NSUInteger i = 0; i < (NSUInteger)B*N; i++) aP[i] = one;
    for (NSUInteger i = 0; i < (NSUInteger)N*M; i++) bP[i] = one;

    MPSMatrix *A = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *Bm = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *C = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev
         transposeLeft:NO
        transposeRight:NO
            resultRows:B
         resultColumns:M
       interiorColumns:N
                 alpha:1.0
                  beta:0.0];

    /* Warmup. */
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:Bm resultMatrix:C];
        [cb commit];
        [cb waitUntilCompleted];
    }

    double t0 = now_s();
    /* Pack all iters into ONE command buffer to amortize sync. This is
     * the real ceiling: how much GEMM/s the hardware can dispatch when
     * the CPU isn't in the loop. */
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int i = 0; i < iters; i++) {
            [mm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:Bm resultMatrix:C];
        }
        [cb commit];
        [cb waitUntilCompleted];
    }
    double dt = now_s() - t0;

    double per_ms = (dt * 1000.0) / iters;
    double flops = 2.0 * (double)B * (double)N * (double)M;
    double tflops = (flops / (per_ms * 1e-3)) / 1e12;
    printf("  B=%-4d N=%-5d M=%-5d  %.3f ms/matmul  %.2f TFLOPS\n",
           B, N, M, per_ms, tflops);
    return per_ms;
}

int main(int argc, char **argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 32;
    int iters = (argc > 2) ? atoi(argv[2]) : 200;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("Device: %s\n", [[dev name] UTF8String]);
        printf("MPSMatrixMultiplication GEMM ceiling, B=%d iters=%d\n", B, iters);

        /* TinyLlama prefill shapes (B = prompt-token count). */
        printf("TinyLlama (hidden=2048, intermediate=5632, vocab=32000):\n");
        bench_one(dev, q, B, 2048, 2048, iters);   /* Q / O */
        bench_one(dev, q, B, 2048, 5632, iters);   /* gate/up */
        bench_one(dev, q, B, 5632, 2048, iters);   /* down */
        bench_one(dev, q, 1, 2048, 32000, iters);  /* lm_head (last-token only) */

        /* Llama-3.2-1B */
        printf("Llama-3.2-1B (hidden=2048, intermediate=8192):\n");
        bench_one(dev, q, B, 2048, 8192, iters);   /* gate/up */
        bench_one(dev, q, B, 8192, 2048, iters);   /* down */

        /* Llama-3.1-8B */
        printf("Llama-3.1-8B (hidden=4096, intermediate=14336):\n");
        bench_one(dev, q, B, 4096, 14336, iters);  /* gate/up */
        bench_one(dev, q, B, 14336, 4096, iters);  /* down */
    }
    return 0;
}
