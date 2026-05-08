#include "pqv2_format.h"
#include "pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define IB_PQV2_MAGIC "IBFV6PQ2"
#define IB_PQV2_VERSION 1u

static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

/* Parse a PQV2 single-tensor blob in-place (no copy). All pointers index
 * into 'buf'. Returns 0 on success, fills out. */
static int parse_pqv2_blob(const uint8_t *buf, size_t size, pqv2_t *out) {
    if (size < 4 + 32) return -1;
    if (memcmp(buf, "PQV2", 4) != 0) return -1;
    const uint32_t *hdr = (const uint32_t *)(buf + 4);
    out->M = hdr[0]; out->N = hdr[1]; out->G = hdr[2]; out->K = hdr[3];
    out->n_subchunks = hdr[4]; out->half = hdr[5];
    out->l2_kind = hdr[6]; out->l2_K = hdr[7];

    size_t cursor = 4 + 32;
    size_t n_chunks = out->N / out->G;
    size_t row_bytes = (size_t)out->M * 2;
    size_t cb_q_bytes = (size_t)out->n_subchunks * out->K * out->half;
    size_t cb_s_bytes = (size_t)out->n_subchunks * out->K * 2;
    size_t idx_bytes = (size_t)out->M * n_chunks * out->n_subchunks;

    if (cursor + row_bytes + cb_q_bytes + cb_s_bytes + idx_bytes > size) return -1;
    out->row_scale = (const uint16_t *)(buf + cursor);  cursor += row_bytes;
    out->cb_q = (const int8_t *)(buf + cursor);          cursor += cb_q_bytes;
    out->cb_scale = (const uint16_t *)(buf + cursor);    cursor += cb_s_bytes;
    out->indices = (const uint8_t *)(buf + cursor);      cursor += idx_bytes;
    out->l2_cb_q = NULL; out->l2_cb_scale = NULL; out->l2_indices = NULL;

    if (out->l2_kind == 2) {
        size_t l2q_bytes = (size_t)out->n_subchunks * out->l2_K * out->half;
        size_t l2s_bytes = (size_t)out->n_subchunks * out->l2_K * 2;
        if (cursor + l2q_bytes + l2s_bytes + idx_bytes > size) return -1;
        out->l2_cb_q = (const int8_t *)(buf + cursor);    cursor += l2q_bytes;
        out->l2_cb_scale = (const uint16_t *)(buf + cursor); cursor += l2s_bytes;
        out->l2_indices = (const uint8_t *)(buf + cursor);  cursor += idx_bytes;
    }
    return 0;
}

int ib_pqv2_file_load(const char *path, ib_pqv2_file *out) {
    memset(out, 0, sizeof(*out));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return -1; }
    size_t fsz = (size_t)st.st_size;
    void *buf = mmap(NULL, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buf == MAP_FAILED) { close(fd); return -1; }
    out->_buffer = buf; out->_buffer_size = fsz; out->_is_mmap = 1; out->_fd = fd;

    const uint8_t *p = (const uint8_t *)buf;
    if (fsz < 24) goto fail;
    if (memcmp(p, IB_PQV2_MAGIC, 8) != 0) goto fail;
    uint32_t version = *(const uint32_t *)(p + 8);
    if (version != IB_PQV2_VERSION) goto fail;
    int n = (int)*(const uint32_t *)(p + 12);
    uint32_t manifest_size = *(const uint32_t *)(p + 16);
    if (24 + manifest_size > fsz) goto fail;

    out->n_tensors = n;
    out->tensors = calloc((size_t)n, sizeof(*out->tensors));
    if (!out->tensors) goto fail;

    const uint8_t *m = p + 24;
    const uint8_t *m_end = m + manifest_size;
    for (int i = 0; i < n; i++) {
        if (m + 2 > m_end) goto fail;
        uint16_t nl = *(const uint16_t *)m; m += 2;
        if (m + nl > m_end) goto fail;
        ib_pqv2_named_tensor *t = &out->tensors[i];
        t->name = malloc(nl + 1);
        memcpy(t->name, m, nl); t->name[nl] = '\0';
        m += nl;

        if (m + 1 + 1 + 2 + 16 + 8 + 8 > m_end) goto fail;
        t->kind = m[0]; t->ndim = m[1]; m += 4;
        memcpy(t->shape, m, 16); m += 16;
        uint64_t blob_off = *(const uint64_t *)m; m += 8;
        uint64_t blob_size = *(const uint64_t *)m; m += 8;
        if (blob_off + blob_size > fsz) goto fail;

        if (t->kind == IB_PQV2_KIND_PQV2) {
            if (parse_pqv2_blob(p + blob_off, (size_t)blob_size, &t->pq) != 0) goto fail;
        } else {
            t->raw_data = p + blob_off;
            t->raw_size = (size_t)blob_size;
        }
    }
    return 0;

fail:
    ib_pqv2_file_free(out);
    return -1;
}

void ib_pqv2_file_free(ib_pqv2_file *f) {
    if (f->tensors) {
        for (int i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
        free(f->tensors);
    }
    if (f->_buffer) {
        if (f->_is_mmap) munmap(f->_buffer, f->_buffer_size);
        else free(f->_buffer);
    }
    if (f->_fd >= 0) close(f->_fd);
    memset(f, 0, sizeof(*f));
}

const ib_pqv2_named_tensor *ib_pqv2_find(const ib_pqv2_file *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}
