/* Quick IBF inspector: load via public API, dump header + first layer
 * tensor metadata. Used to confirm that a real file matches what the
 * Metal upload helper expects (w4a8 weights, fp16 norms, fp16 KV). */
#include <stdio.h>
#include <stdlib.h>
#include "inferbit.h"
#include "inferbit_internal.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.ibf>\n", argv[0]); return 1; }
    inferbit_config *cfg = inferbit_config_create();
    inferbit_model *m = inferbit_load(argv[1], cfg);
    if (!m) { fprintf(stderr, "load failed\n"); return 2; }
    printf("arch:           %s\n", m->header.architecture);
    printf("layers:         %d\n", m->header.num_layers);
    printf("hidden:         %d\n", m->header.hidden_size);
    printf("intermediate:   %d\n", m->header.intermediate_size);
    printf("n_heads:        %d\n", m->header.num_heads);
    printf("n_kv_heads:     %d\n", m->header.num_kv_heads);
    printf("head_dim:       %d\n", m->header.head_dim);
    printf("vocab:          %d\n", m->header.vocab_size);
    printf("max_ctx:        %d\n", m->header.max_context_length);
    printf("rope_theta:     %g\n", m->header.rope_theta);
    printf("norm_eps:       %g\n", m->header.norm_epsilon);
    printf("kv_bits:        %d\n", m->header.kv_bits);
    printf("default_bits:   %d\n", m->header.default_bits);

    printf("\ntoken_embedding:  bits=%d  size=%zu  scale_size=%zu  off=0x%zx  pq=%s\n",
           m->token_embedding.bits, m->token_embedding.size,
           m->token_embedding.scale_size, m->token_embedding.offset,
           m->token_embedding.pq ? "yes" : "no");
    printf("output_norm:      bits=%d  size=%zu\n",
           m->output_norm.bits, m->output_norm.size);
    printf("output_head:      bits=%d  size=%zu  scale_size=%zu  off=0x%zx  pq=%s  tied=%s\n",
           m->output_head.bits, m->output_head.size,
           m->output_head.scale_size, m->output_head.offset,
           m->output_head.pq ? "yes" : "no",
           (m->output_head.offset == m->token_embedding.offset) ? "YES" : "no");

    if (m->header.num_layers > 0) {
        ib_layer_meta *L = &m->layers[0];
        printf("\nLayer[0] tensors:\n");
        #define DUMP(name) \
            printf("  %-15s bits=%d size=%zu scale_size=%zu pq=%s\n", \
                   #name, L->name.bits, L->name.size, L->name.scale_size, \
                   L->name.pq ? "yes" : "no")
        DUMP(q_proj);   DUMP(k_proj);   DUMP(v_proj);   DUMP(o_proj);
        DUMP(gate_proj); DUMP(up_proj); DUMP(down_proj);
        DUMP(input_norm); DUMP(post_attn_norm);
        #undef DUMP
    }

    inferbit_free(m);
    inferbit_config_free(cfg);
    return 0;
}
