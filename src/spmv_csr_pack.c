/* SPDX-License-Identifier: GPL-3.0-or-later
 * AOCL-style scalar loads packed into SIMD, independently implemented from
 * the published algorithm. Sources and numerical differences: docs/v1.md.
 */
#include "spmv_plan.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
__attribute__((target("avx2")))
void spmv_pack_f32(const struct csr_matrix_int32 *a,
                   const float *restrict x, float *restrict y)
{
    const float *restrict v = a->values;
    const int32_t *restrict j = a->column_indices;
#pragma omp for schedule(static)
    for (int32_t r = 0; r < a->num_rows; r++) {
        int64_t k = a->row_ptr[r], end = a->row_ptr[r+1];
        __m256 sums = _mm256_setzero_ps();
        for (; end-k >= 8; k += 8) {
            __m256 vx = _mm256_set_ps(x[j[k+7]],x[j[k+6]],x[j[k+5]],x[j[k+4]],
                                     x[j[k+3]],x[j[k+2]],x[j[k+1]],x[j[k]]);
            sums = _mm256_add_ps(sums, _mm256_mul_ps(_mm256_loadu_ps(v+k),vx));
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, sums);
        float sum = 0;
        for (int l = 0; l < 8; l++) sum += lanes[l];
        for (; k < end; k++) sum += v[k]*x[j[k]];
        y[r] += sum;
    }
}
#else
void spmv_pack_f32(const struct csr_matrix_int32 *a, const float *x, float *y)
{ spmv_scalar_f32(a,x,y); }
#endif
