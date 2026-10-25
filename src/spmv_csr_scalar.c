/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spmv_plan.h"

/* Local control, including when caller overrides CFLAGS. GCC is the supported
 * compiler for this comparison. No project-wide fast-math or novec changes.
 */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize,no-tree-slp-vectorize")))
#endif
void spmv_scalar_f32(const struct csr_matrix_int32 *a,
                     const float *restrict x, float *restrict y)
{
    const float *restrict v = a->values;
#pragma omp for schedule(static)
    for (int32_t r = 0; r < a->num_rows; r++) {
        float sum = 0;
        for (int64_t k = a->row_ptr[r]; k < a->row_ptr[r+1]; k++)
            sum += v[k]*x[a->column_indices[k]];
        y[r] += sum;
    }
}
