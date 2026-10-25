/* SPDX-License-Identifier: GPL-3.0-or-later
 * SELL-C-sigma, sigma=1 (no row permutation). Independent implementation.
 * Kreutzer et al., DOI 10.1137/130930352; GHOST algorithm reference.
 */
#include "spmv_plan.h"
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

__attribute__((target("avx2")))
void spmv_sell8_f32(const struct spmv_plan *p,
                    const float *restrict x, float *restrict y)
{
#pragma omp for schedule(static)
    for (int64_t s = 0; s < p->slices; s++) {
        __m256 sum = _mm256_setzero_ps();
        for (int64_t k = p->slice_ptr[s]; k < p->slice_ptr[s+1]; k += 8) {
            __m256i j = _mm256_loadu_si256((const __m256i *)(p->columns+k));
            __m256 vx;
            if (p->full_slices) vx = _mm256_i32gather_ps(x,j,4);
            else {
                __m256 mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(j,_mm256_set1_epi32(-1)));
                vx = _mm256_mask_i32gather_ps(_mm256_setzero_ps(),x,j,mask,4);
            }
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(p->values+k),vx));
        }
        int64_t r = s*8;
        if (r+8 <= p->matrix->num_rows)
            _mm256_storeu_ps(y+r,_mm256_add_ps(_mm256_loadu_ps(y+r),sum));
        else {
            __m256i mask = _mm256_setr_epi32(r<p->matrix->num_rows?-1:0,
                r+1<p->matrix->num_rows?-1:0,r+2<p->matrix->num_rows?-1:0,
                r+3<p->matrix->num_rows?-1:0,r+4<p->matrix->num_rows?-1:0,
                r+5<p->matrix->num_rows?-1:0,r+6<p->matrix->num_rows?-1:0,
                r+7<p->matrix->num_rows?-1:0);
            _mm256_maskstore_ps(y+r,mask,_mm256_add_ps(_mm256_maskload_ps(y+r,mask),sum));
        }
    }
}

__attribute__((target("avx512f")))
void spmv_sell16_f32(const struct spmv_plan *p,
                     const float *restrict x, float *restrict y)
{
#pragma omp for schedule(static)
    for (int64_t s = 0; s < p->slices; s++) {
        __m512 sum = _mm512_setzero_ps();
        for (int64_t k = p->slice_ptr[s]; k < p->slice_ptr[s+1]; k += 16) {
            __m512i j = _mm512_loadu_si512(p->columns+k);
            __mmask16 mask = p->full_slices ? 0xffff :
                _mm512_cmp_epi32_mask(j,_mm512_set1_epi32(-1),_MM_CMPINT_GT);
            __m512 vx = _mm512_mask_i32gather_ps(_mm512_setzero_ps(),mask,j,x,4);
            sum = _mm512_add_ps(sum,_mm512_mul_ps(_mm512_loadu_ps(p->values+k),vx));
        }
        int64_t r = s*16;
        int64_t left = p->matrix->num_rows-r;
        __mmask16 mask = left >= 16 ? 0xffff : (1u << left)-1;
        _mm512_mask_storeu_ps(y+r,mask,_mm512_add_ps(_mm512_maskz_loadu_ps(mask,y+r),sum));
    }
}
#else
void spmv_sell8_f32(const struct spmv_plan *p, const float *x, float *y)
{ spmv_scalar_f32(p->matrix,x,y); }
void spmv_sell16_f32(const struct spmv_plan *p, const float *x, float *y)
{ spmv_scalar_f32(p->matrix,x,y); }
#endif
