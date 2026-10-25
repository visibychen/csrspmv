/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_PLAN_H
#define SPMV_PLAN_H

#include "csr.h"
#include "vector.h"
#include <stddef.h>

enum spmv_kernel { spmv_original, spmv_csr_scalar, spmv_csr_pack, spmv_sell };

/* A plan borrows CSR; SELL also owns a snapshot of its values/structure.
 * Rebuild after ANY matrix mutation. x and y may change between calls.
 * prepare/free are serial; execute is called by EVERY thread of one team,
 * just like csr_matrix_int32_spmv. Concurrent teams may not share y.
 */
struct spmv_plan {
    const struct csr_matrix_int32 *matrix;
    enum spmv_kernel kernel;
    const char *reason;
    int chunk, full_slices;
    int64_t slices, slots;
    size_t extra_bytes;
    int64_t *slice_ptr;
    int32_t *columns;
    float *values;
};

const char *spmv_kernel_name(enum spmv_kernel kernel);
int spmv_plan_prepare(struct spmv_plan *plan,
                      const struct csr_matrix_int32 *matrix,
                      const struct vector *x, const struct vector *y,
                      enum spmv_kernel requested, int chunk, size_t budget);
int spmv_plan_execute(const struct spmv_plan *plan, const struct vector *x,
                      struct vector *y, int64_t *flops);
void spmv_plan_free(struct spmv_plan *plan);

void spmv_scalar_f32(const struct csr_matrix_int32 *, const float *, float *);
void spmv_pack_f32(const struct csr_matrix_int32 *, const float *, float *);
void spmv_sell8_f32(const struct spmv_plan *, const float *, float *);
void spmv_sell16_f32(const struct spmv_plan *, const float *, float *);

#endif
