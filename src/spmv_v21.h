/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef SPMV_V21_H
#define SPMV_V21_H
#include "spmv_v2.h"
#include <stdio.h>
#define V21_MAX_MODELS 256
enum v21_kernel { v21_auto, v21_original, v21_csr, v21_short8, v21_merge, v21_long };
struct spmv_v21_options {
    enum v21_kernel kernel;
    const char *profile;
    int threads, prepare_mode, cache, owners, diagnostics;
};
struct v21_slot {
    int64_t row, nz, end_row, end_nz;
    struct v2_fragment fragment[2];
    int count;
    double compute_s, publish_s, fixup_s, complete_s;
};
struct v21_group { int64_t row; int first, last; };
struct spmv_v21_state {
    struct spmv_v21_options options;
    struct spmv_v2_options common;
    enum v21_kernel selected;
    int threads, prepare_threads, all8, work, avx2, groups;
    int64_t expected_calls, longest, empty_rows, long_threshold;
    double validation_s, policy_s, predicted_s, baseline_predicted_s;
    unsigned long long fingerprint;
    int profile_match;
    struct v21_slot *slots;
    struct v21_group *group;
};
struct spmv_plan;
void spmv_v21_defaults(struct spmv_v21_options *);
const char *spmv_v21_name(enum v21_kernel);
unsigned long long spmv_v21_fingerprint(void);
int spmv_v21_prepare(struct spmv_plan *, const struct csr_matrix_int32 *,
    const struct vector *, const struct vector *, const struct spmv_v2_options *,
    const struct spmv_v21_options *, int64_t, size_t);
int spmv_v21_execute(const struct spmv_plan *, const struct vector *, struct vector *, int64_t *);
void spmv_v21_report(const struct spmv_plan *, FILE *);
#endif
