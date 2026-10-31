/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_V3_H
#define SPMV_V3_H
#include "csr.h"
#include "vector.h"
#include <stdint.h>
enum spmv_v3_backend { v3_auto, v3_tile8, v3_pipeline4, v3_csr };
struct spmv_v3_options { enum spmv_v3_backend backend; int work; int64_t serial_threshold; int allow_simd; };
struct spmv_v3_plan {
    const struct csr_matrix_int32 *matrix;
    struct spmv_v3_options options;
    int supported, serial, threads, simd;
};
void spmv_v3_defaults(struct spmv_v3_options *);
const char *spmv_v3_backend_name(enum spmv_v3_backend);
int spmv_v3_prepare(struct spmv_v3_plan *,const struct csr_matrix_int32 *,const struct vector *,const struct vector *,const struct spmv_v3_options *);
/* Collective within a team; each call completes all y updates. Rebuild after
 * structural/type mutation. Values/x may change; y must not overlap x/CSR. */
int spmv_v3_execute(const struct spmv_v3_plan *,const struct vector *,struct vector *,int64_t *);
int64_t spmv_v3_cut(const struct csr_matrix_int32 *,int,int,int);
int spmv_v3_is_tile8(const struct csr_matrix_int32 *,int64_t,int64_t);
#endif
