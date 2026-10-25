/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spmv_plan.h"
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *spmv_kernel_name(enum spmv_kernel k)
{
    switch (k) {
    case spmv_original: return "original";
    case spmv_csr_scalar: return "csr-scalar";
    case spmv_csr_pack: return "csr-pack";
    case spmv_sell: return "sell";
    }
    return "invalid";
}

void spmv_plan_free(struct spmv_plan *p)
{
    free(p->slice_ptr); free(p->columns); free(p->values);
    p->slice_ptr = NULL; p->columns = NULL; p->values = NULL;
    p->extra_bytes = 0;
}

static int fallback(struct spmv_plan *p, const char *why)
{
    spmv_plan_free(p);
    p->kernel = spmv_original;
    p->reason = why;
    return 0;
}

/* Linux MemAvailable includes reclaimable pages. Failure leaves the explicit
 * budget in force. Half of current availability is a conservative policy,
 * NOT a theorem about SELL. No allocation probes or swap experiments.
 */
static size_t memory_budget(size_t budget)
{
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    unsigned long long kb;
    if (!f) return budget;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            if (kb <= SIZE_MAX / 1024 && (size_t)kb * 512 < budget)
                budget = (size_t)kb * 512;
            break;
        }
    }
    fclose(f);
    return budget;
}

int spmv_plan_prepare(struct spmv_plan *p,
                      const struct csr_matrix_int32 *a,
                      const struct vector *x, const struct vector *y,
                      enum spmv_kernel requested, int chunk, size_t budget)
{
    memset(p, 0, sizeof(*p));
    p->matrix = a; p->kernel = requested; p->chunk = chunk;
    p->reason = "selected";
    if (requested < spmv_original || requested > spmv_sell ||
        (chunk != 8 && chunk != 16)) return EINVAL;
    if (a->num_rows < 0 || a->num_columns < 0 || a->num_nonzeros < 0 ||
        x->num_values != a->num_columns || y->num_values != a->num_rows)
        return EINVAL;
    if (requested == spmv_original) return 0;
    if (a->value_format != csr_value_f32 ||
        x->value_format != vector_value_f32 || y->value_format != vector_value_f32)
        return fallback(p, "unsupported-types");
    /* Overflow of counters is rejected even for a direct API caller. */
    if (a->num_nonzeros > INT64_MAX / 2) return EOVERFLOW;
    if (!a->row_ptr || a->row_ptr[0] != 0) return EINVAL;
    for (int64_t i = 0; i < a->num_rows; i++)
        if (a->row_ptr[i] < 0 || a->row_ptr[i+1] < a->row_ptr[i] ||
            a->row_ptr[i+1] > a->num_nonzeros) return EINVAL;
    if (a->row_ptr[a->num_rows] != a->num_nonzeros) return EINVAL;
    if (a->num_nonzeros && (!a->values || !a->column_indices)) return EINVAL;
    for (int64_t k = 0; k < a->num_nonzeros; k++)
        if (a->column_indices[k] < 0 || a->column_indices[k] >= a->num_columns)
            return EINVAL;
    if (requested == spmv_csr_scalar) return 0;
#if defined(__x86_64__) || defined(__i386__)
    if (!__builtin_cpu_supports("avx2")) return fallback(p, "no-avx2");
    if (requested == spmv_sell && chunk == 16 &&
        !__builtin_cpu_supports("avx512f")) return fallback(p, "no-avx512f");
#else
    return fallback(p, "unsupported-isa");
#endif
    if (requested == spmv_csr_pack) return 0;
    p->slices = ((int64_t)a->num_rows + chunk - 1) / chunk;
    p->full_slices = 1;
    /* Inspect without allocating: 64-bit offsets and checked slot/byte sums. */
    for (int64_t s = 0; s < p->slices; s++) {
        int64_t width = 0, count = 0;
        for (int lane = 0; lane < chunk; lane++) {
            int64_t r = s * chunk + lane;
            if (r < a->num_rows) {
                int64_t len = a->row_ptr[r+1] - a->row_ptr[r];
                if (len > width) width = len;
                count += len;
            }
        }
        if (width > (INT64_MAX - p->slots) / chunk)
            return fallback(p, "slot-overflow");
        p->slots += width * chunk;
        if (count != width * chunk || (s+1)*chunk > a->num_rows)
            p->full_slices = 0;
    }
    /* Explicit v1 guard: at most 2x padding, regardless of free memory. */
    if (a->num_nonzeros && p->slots - a->num_nonzeros > a->num_nonzeros)
        return fallback(p, "padding-over-2x");
    if ((uint64_t)(p->slices+1) > SIZE_MAX / sizeof(int64_t))
        return fallback(p, "byte-overflow");
    size_t meta = (size_t)(p->slices+1) * sizeof(int64_t);
    if ((uint64_t)p->slots > (SIZE_MAX-meta) / (sizeof(float)+sizeof(int32_t)))
        return fallback(p, "byte-overflow");
    size_t bytes = meta + (size_t)p->slots * (sizeof(float)+sizeof(int32_t));
    if (bytes > memory_budget(budget)) return fallback(p, "memory-budget");
    p->slice_ptr = malloc(meta);
    /* malloc(0) is not required to return a non-null pointer. */
    p->columns = p->slots ? malloc((size_t)p->slots*sizeof(int32_t)) : NULL;
    p->values = p->slots ? malloc((size_t)p->slots*sizeof(float)) : NULL;
    if (!p->slice_ptr || (p->slots && (!p->columns || !p->values)))
        return fallback(p, "allocation-failed");
    p->extra_bytes = bytes;
    p->slice_ptr[0] = 0;
    for (int64_t s = 0; s < p->slices; s++) {
        int64_t width = 0;
        for (int lane = 0; lane < chunk; lane++) {
            int64_t r = s*chunk+lane;
            if (r < a->num_rows && a->row_ptr[r+1]-a->row_ptr[r] > width)
                width = a->row_ptr[r+1]-a->row_ptr[r];
        }
        int64_t base = p->slice_ptr[s];
        p->slice_ptr[s+1] = base + width*chunk;
        for (int64_t k = 0; k < width; k++) {
            for (int lane = 0; lane < chunk; lane++) {
                int64_t r = s*chunk+lane, dst = base+k*chunk+lane;
                if (r < a->num_rows && k < a->row_ptr[r+1]-a->row_ptr[r]) {
                    int64_t src = a->row_ptr[r]+k;
                    p->columns[dst] = a->column_indices[src];
                    p->values[dst] = ((const float *)a->values)[src];
                } else {
                    p->columns[dst] = -1; /* masked gather: never reads x[-1/0] */
                    p->values[dst] = 0;
                }
            }
        }
    }
    return 0;
}

int spmv_plan_execute(const struct spmv_plan *p, const struct vector *x,
                      struct vector *y, int64_t *flops)
{
    if (p->matrix->num_rows != y->num_values ||
        p->matrix->num_columns != x->num_values) return EINVAL;
    if (p->kernel == spmv_original)
        return csr_matrix_int32_spmv(p->matrix, x, y, flops);
    if (x->value_format != vector_value_f32 || y->value_format != vector_value_f32)
        return EINVAL;
    if (*flops > INT64_MAX - 2*p->matrix->num_nonzeros) return EOVERFLOW;
    switch (p->kernel) {
    case spmv_csr_scalar: spmv_scalar_f32(p->matrix, x->values, y->values); break;
    case spmv_csr_pack: spmv_pack_f32(p->matrix, x->values, y->values); break;
    case spmv_sell:
        if (p->chunk == 8) spmv_sell8_f32(p, x->values, y->values);
        else spmv_sell16_f32(p, x->values, y->values);
        break;
    default: return EINVAL;
    }
    *flops += 2*p->matrix->num_nonzeros;
    return 0;
}
