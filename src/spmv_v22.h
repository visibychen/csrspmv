/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_V22_H
#define SPMV_V22_H
#include "spmv_v2.h"
#include <stdio.h>
struct spmv_v22_options { int guarded, empty; }; /* empty: auto=0, off=1, on=2 */
struct v22_bounds { int64_t begin,end; int32_t first,last; };
struct spmv_v22_state {
    int threads, empty_runs;
    int64_t calls, empty_rows;
    int32_t count;
    int32_t *rows;
    struct v22_bounds *bounds;
    const char *route;
    double validation_s;
};
struct spmv_plan;
int spmv_v22_prepare(struct spmv_plan *,const struct csr_matrix_int32 *,
    const struct vector *,const struct vector *,const struct spmv_v2_options *,
    const struct spmv_v22_options *,int64_t,size_t);
int spmv_v22_execute(const struct spmv_plan *,const struct vector *,struct vector *,int64_t *);
void spmv_v22_report(const struct spmv_plan *,FILE *);
#endif
