/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_V221_H
#define SPMV_V221_H
#include "spmv_v2.h"
#include <stdio.h>
struct spmv_v221_options {
    int guarded, empty; /* empty: auto=0, off=1, on=2 */
    int parallel_validation, balance_work;
};
struct spmv_v221_state {
    int threads, validation_threads, statistics_known, index_retry;
    int64_t empty_rows, nonempty_rows, index_capacity;
    double row_s, column_s, retry_s;
    const char *route;
};
struct spmv_plan;
int spmv_v221_prepare(struct spmv_plan *,const struct csr_matrix_int32 *,
    const struct vector *,const struct vector *,const struct spmv_v2_options *,
    const struct spmv_v221_options *,int64_t,size_t);
void spmv_v221_report(const struct spmv_plan *,FILE *);
#endif
