/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_V222_H
#define SPMV_V222_H
#include "spmv_v22.h"
#define V222_MAX_RUNS 8
struct spmv_v222_options { int policy, runs; }; /* policy guarded/v22/v2; runs auto/off/on */
struct v222_run { int64_t begin, end; };
struct spmv_v222_state {
    struct v222_run runs[V222_MAX_RUNS];
    int count, active;
    int64_t total_runs, empty_rows;
    size_t discarded_index_bytes;
    const char *route;
};
struct spmv_plan;
int spmv_v222_prepare(struct spmv_plan *, const struct csr_matrix_int32 *,
    const struct vector *, const struct vector *, const struct spmv_v2_options *,
    const struct spmv_v22_options *, const struct spmv_v222_options *, int64_t, size_t);
int spmv_v222_execute(const struct spmv_plan *, const struct vector *, struct vector *, int64_t *);
void spmv_v222_report(const struct spmv_plan *, FILE *);
#endif
