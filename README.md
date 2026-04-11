# libinferbit

C shared library for quantized LLM inference. The core engine behind [InferBit](https://github.com/demonarch/inferbit-py).

Loads INT4/INT8 quantized models and runs transformer inference with SIMD-optimized kernels on CPU. Designed to be called via FFI from Python, Node.js, or any language that can load a shared library.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Produces `libinferbit.so` (Linux), `libinferbit.dylib` (macOS), or `inferbit.dll` (Windows).

### Run tests

```bash
./build/test_inferbit
./build/test_ibf_loader
./build/test_forward
./build/test_convert
```

## C API

```c
#include "inferbit.h"

// Load a quantized model
inferbit_config* config = inferbit_config_create();
inferbit_config_set_threads(config, 8);
inferbit_model* model = inferbit_load("model.ibf", config);
inferbit_config_free(config);

// Generate tokens
int32_t input[] = {1, 2, 3, 4, 5};
int32_t output[256];
inferbit_sample_params params = inferbit_default_sample_params();
params.temperature = 0.7f;
params.max_tokens = 256;

int n = inferbit_generate(model, input, 5, output, 256, params);

// Streaming
int callback(int32_t token, void* ctx) {
    printf("token: %d\n", token);
    return 1; // return 0 to stop
}
inferbit_generate_stream(model, input, 5, callback, NULL, params);

// Convert safetensors to IBF
inferbit_convert_config ccfg = inferbit_default_convert_config();
ccfg.default_bits = 4;
ccfg.sensitive_bits = 8;
inferbit_convert("model.safetensors", "model.ibf", &ccfg);

// Model info
printf("arch: %s\n", inferbit_model_architecture(model));
printf("layers: %d\n", inferbit_model_num_layers(model));
printf("memory: %zu MB\n", inferbit_model_total_memory(model) / 1024 / 1024);

// Cleanup
inferbit_free(model);
```

## Features

### Inference
- Full transformer forward pass (embedding, RMSNorm, GQA/MQA/MHA attention, RoPE, SiLU MLP)
- INT4, INT8, and INT2 weight quantization with per-channel FP16 scales
- Selective precision: INT8 for attention/embeddings, INT4 for MLP layers
- KV-cache with FP32, INT8, or INT4 storage
- Greedy and sampled generation (temperature, top-k, top-p, repeat penalty)
- Streaming token output via callback
- Speculative decoding with draft model (greedy mode)
- EOS token detection

### Kernels
- ARM NEON (Apple Silicon, ARM servers)
- x86 AVX2 (Intel/AMD)
- Scalar fallback (any platform)
- Runtime SIMD auto-detection
- Multi-threaded matmul and parallel attention heads

### Conversion
- Safetensors (single file and multi-shard)
- GGUF (F16, F32, BF16, Q4_0, Q8_0)
- HuggingFace `config.json` for exact architecture parameters
- Hadamard rotation support (for future quantization research)
- Structured sparsity with magnitude-based masks
- Progress callbacks

### Evaluation
- Perplexity computation (teacher forcing)
- Throughput benchmarking with warmup
- Quality gates (max perplexity, min tokens/sec, max memory)
- Calibration profile search (INT2 -> INT4 -> INT8, pick first passing)

### Format
- `.ibf` binary format: mmap-friendly, 64-byte aligned, JSON header + packed weights
- Supports mixed per-layer bit-widths
- Sparsity mask metadata
- Model architecture fully described in header (no hardcoded models)

## Supported Architectures

Any LLaMA-family model: LLaMA 2/3, Mistral, TinyLlama, Code Llama, and any model with GQA/MQA/MHA attention, RMSNorm, SiLU activation, and RoPE positional encoding.

Architecture is defined by the `.ibf` header, not by compiled-in code. New architectures are supported by producing a valid `.ibf` file.

## File Structure

```
include/
  inferbit.h          Public C API (single header)

src/
  forward.c           Transformer forward pass
  generate.c          Token generation and sampling
  ibf_loader.c        .ibf file parser and mmap loader
  convert.c           Safetensors to IBF converter
  convert_gguf.c      GGUF to IBF converter
  safetensors.c       Safetensors format parser
  safetensors_multi.c Multi-shard safetensors support
  tensor_source.c     Unified single/multi-shard abstraction
  gguf.c              GGUF format parser
  config_json.c       HuggingFace config.json parser
  quantize.c          FP16/BF16/FP32 to INT4/INT8/INT2 quantization
  eval.c              Perplexity, throughput, quality gates
  calibrate.c         Quantization profile search
  threading.c         Thread pool with atomic barrier sync
  simd.c              Runtime SIMD detection (CPUID/NEON)
  kv_cache.c          KV-cache management
  speculative.c       Speculative decoding API
  error.c             Thread-local error handling
  config.c            Runtime configuration
  version.c           Version info
  kernels/
    scalar.c          Scalar fallback kernels + dispatch table
    neon.c            ARM NEON optimized kernels
    avx2.c            x86 AVX2 optimized kernels
    ternary.c         INT2 ternary kernels (experimental)

tests/
  test_basic.c        API, config, null safety (11 tests)
  test_ibf_loader.c   IBF parsing, mmap, validation (8 tests)
  test_forward.c      Forward pass, generation, KV cache (7 tests)
  test_convert.c      Conversion pipeline end-to-end (6 tests)
```

## Benchmarks

Apple Silicon, INT4 + INT8 attention, 8 threads:

| Model | Size | Decode |
|-------|------|--------|
| TinyLlama 1.1B | 643 MB | 34.6 tok/s |
| Mistral 7B | 3,971 MB | 6.8 tok/s |

## Language Bindings

libinferbit is designed for FFI consumption:

| Language | Package | Binding method |
|----------|---------|---------------|
| Python | [inferbit](https://pypi.org/project/inferbit/) | ctypes |
| Node.js | @inferbit/node (coming soon) | bun:ffi / koffi |
| Any | — | dlopen + function pointers |

The C API uses opaque pointers, simple types (int32, float, char*), and callback function pointers. No C++ name mangling, no runtime dependencies beyond libc and libm.

## License

MIT
