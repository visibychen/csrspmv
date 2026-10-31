/* SPDX-License-Identifier: GPL-3.0-or-later
 * Direct-CSR local register transpose. No converted matrix or cached Ax.
 */
#include "spmv_v3.h"
#include <errno.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#endif
#include <string.h>
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <immintrin.h>
#define HAVE_TARGET_AVX2 1
#endif
void spmv_v3_defaults(struct spmv_v3_options *o)
{*o=(struct spmv_v3_options){v3_tile8,1,32768,1};}
const char *spmv_v3_backend_name(enum spmv_v3_backend b)
{switch(b){case v3_auto:return "auto";case v3_tile8:return "tile8";case v3_pipeline4:return "pipeline4";case v3_csr:return "csr";}return "invalid";}
static int avx2_available(void)
{
#ifdef HAVE_TARGET_AVX2
    return !!__builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}
int spmv_v3_prepare(struct spmv_v3_plan *p,const struct csr_matrix_int32 *a,
 const struct vector *x,const struct vector *y,const struct spmv_v3_options *o)
{
    if(!p)return EINVAL;
    memset(p,0,sizeof(*p));
    if(!a||!x||!y||!o)return EINVAL;
    p->matrix=a;p->options=*o;
    if(o->backend<v3_auto||o->backend>v3_csr||(o->work!=0&&o->work!=1)||
       o->serial_threshold<0||(o->allow_simd!=0&&o->allow_simd!=1)||
       a->num_rows<0||a->num_columns<0||a->num_nonzeros<0||
       x->num_values!=a->num_columns||y->num_values!=a->num_rows)return EINVAL;
    if(a->num_nonzeros>INT64_MAX/2||a->num_nonzeros>INT64_MAX-a->num_rows)return EOVERFLOW;
    p->threads=omp_get_max_threads();
    p->supported=a->value_format==csr_value_f32&&x->value_format==vector_value_f32&&y->value_format==vector_value_f32;
    if(!p->supported)return 0;
    if(!a->row_ptr||a->row_ptr[0]||a->row_ptr[a->num_rows]!=a->num_nonzeros||
       (a->num_nonzeros&&(!a->column_indices||!a->values))||
       (x->num_values&&!x->values)||(y->num_values&&!y->values))return EINVAL;
    int bad=0;
    int par=p->threads>1&&a->num_rows+a->num_nonzeros>=1048576;
    if(par){
#pragma omp parallel reduction(|:bad)
        {
#pragma omp for schedule(static) nowait
            for(int64_t r=0;r<a->num_rows;r++)
                if(a->row_ptr[r]<0||a->row_ptr[r+1]<a->row_ptr[r]||a->row_ptr[r+1]>a->num_nonzeros)bad=1;
#pragma omp for schedule(static)
            for(int64_t k=0;k<a->num_nonzeros;k++)
                if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns)bad=1;
        }
    }else{
        for(int64_t r=0;r<a->num_rows;r++)
            if(a->row_ptr[r]<0||a->row_ptr[r+1]<a->row_ptr[r]||a->row_ptr[r+1]>a->num_nonzeros)bad=1;
        for(int64_t k=0;k<a->num_nonzeros;k++)
            if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns)bad=1;
    }
    if(bad)return EINVAL;
    p->serial=p->threads==1||a->num_rows+a->num_nonzeros<=o->serial_threshold;
    if(p->serial)p->threads=1;
    p->simd=o->allow_simd&&avx2_available();return 0;
}
static int64_t target(int64_t n,int t,int nt)
{return n/nt*t+(t<n%nt?t:n%nt);}
int64_t spmv_v3_cut(const struct csr_matrix_int32 *a,int t,int nt,int work)
{
    if(!work)return target(a->num_rows,t,nt);
    int64_t goal=target(a->num_nonzeros+a->num_rows,t,nt),lo=0,hi=a->num_rows;
    while(lo<hi){int64_t mid=lo+(hi-lo)/2;if(a->row_ptr[mid]+mid<goal)lo=mid+1;else hi=mid;}
    return lo;
}
int spmv_v3_is_tile8(const struct csr_matrix_int32 *a,int64_t row,int64_t end)
{
    if(end-row<8)return 0;
    int64_t b=a->row_ptr[row];
    for(int k=1;k<=8;k++)if(a->row_ptr[row+k]!=b+8*k)return 0;
    return 1;
}
static void generic(const struct csr_matrix_int32 *a,const float *restrict x,float *restrict y,int64_t b,int64_t e)
{
    const float *restrict v=a->values;const int32_t *restrict j=a->column_indices;
    for(int64_t r=b;r<e;r++){
        float sum=0;
        for(int64_t k=a->row_ptr[r];k<a->row_ptr[r+1];k++)sum+=v[k]*x[j[k]];
        y[r]+=sum;
    }
}
static int short4(const struct csr_matrix_int32 *a,int64_t r)
{
    int64_t lo=INT64_MAX,hi=0;
    for(int i=0;i<4;i++){int64_t n=a->row_ptr[r+i+1]-a->row_ptr[r+i];if(n<lo)lo=n;if(n>hi)hi=n;}
    return lo>=1&&hi<=16&&hi<=2*lo;
}
static void pipeline4(const struct csr_matrix_int32 *a,const float *restrict x,float *restrict y,int64_t r)
{
    const float *restrict v=a->values;const int32_t *restrict j=a->column_indices;
    int64_t k0=a->row_ptr[r],e0=a->row_ptr[r+1],k1=e0,e1=a->row_ptr[r+2],k2=e1,e2=a->row_ptr[r+3],k3=e2,e3=a->row_ptr[r+4];
    float s0=0,s1=0,s2=0,s3=0;
    while(k0<e0||k1<e1||k2<e2||k3<e3){
        if(k0<e0){s0+=v[k0]*x[j[k0]];k0++;}
        if(k1<e1){s1+=v[k1]*x[j[k1]];k1++;}
        if(k2<e2){s2+=v[k2]*x[j[k2]];k2++;}
        if(k3<e3){s3+=v[k3]*x[j[k3]];k3++;}
    }
    y[r]+=s0;y[r+1]+=s1;y[r+2]+=s2;y[r+3]+=s3;
}
#ifdef HAVE_TARGET_AVX2
__attribute__((target("avx2"),always_inline))
static inline __m256 pair_values(const float *lo,const float *hi)
{return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(lo)),_mm_loadu_ps(hi),1);}
__attribute__((target("avx2"),always_inline))
static inline __m256 pair_indices(const int32_t *lo,const int32_t *hi)
{return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si128((const __m128i *)lo))),_mm_castsi128_ps(_mm_loadu_si128((const __m128i *)hi)),1);}
__attribute__((target("avx2"),always_inline))
static inline void transpose4(__m256 *a,__m256 *b,__m256 *c,__m256 *d)
{
    __m256 t0=_mm256_unpacklo_ps(*a,*b),t1=_mm256_unpackhi_ps(*a,*b),t2=_mm256_unpacklo_ps(*c,*d),t3=_mm256_unpackhi_ps(*c,*d);
    *a=_mm256_shuffle_ps(t0,t2,0x44);*b=_mm256_shuffle_ps(t0,t2,0xee);
    *c=_mm256_shuffle_ps(t1,t3,0x44);*d=_mm256_shuffle_ps(t1,t3,0xee);
}
__attribute__((target("avx2")))
static void tile8(const struct csr_matrix_int32 *a,const float *x,float *y,int64_t r)
{
    const float *v=(const float *)a->values+a->row_ptr[r];const int32_t *j=a->column_indices+a->row_ptr[r];
    __m256 sum=_mm256_setzero_ps();
    for(int off=0;off<8;off+=4){
        __m256 a0=pair_values(v+off,v+32+off),a1=pair_values(v+8+off,v+40+off),a2=pair_values(v+16+off,v+48+off),a3=pair_values(v+24+off,v+56+off);
        __m256 j0=pair_indices(j+off,j+32+off),j1=pair_indices(j+8+off,j+40+off),j2=pair_indices(j+16+off,j+48+off),j3=pair_indices(j+24+off,j+56+off);
        transpose4(&a0,&a1,&a2,&a3);transpose4(&j0,&j1,&j2,&j3);
        sum=_mm256_add_ps(sum,_mm256_mul_ps(a0,_mm256_i32gather_ps(x,_mm256_castps_si256(j0),4)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(a1,_mm256_i32gather_ps(x,_mm256_castps_si256(j1),4)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(a2,_mm256_i32gather_ps(x,_mm256_castps_si256(j2),4)));
        sum=_mm256_add_ps(sum,_mm256_mul_ps(a3,_mm256_i32gather_ps(x,_mm256_castps_si256(j3),4)));
    }
    _mm256_storeu_ps(y+r,_mm256_add_ps(_mm256_loadu_ps(y+r),sum));
}
#endif
static void rows(const struct spmv_v3_plan *p,const float *x,float *y,int64_t b,int64_t e)
{
    const struct csr_matrix_int32 *a=p->matrix;
    if(p->options.backend==v3_csr){generic(a,x,y,b,e);return;}
    for(int64_t r=b;r<e;){
#ifdef HAVE_TARGET_AVX2
        if(p->simd&&(p->options.backend==v3_auto||p->options.backend==v3_tile8)&&spmv_v3_is_tile8(a,r,e)){
            tile8(a,x,y,r);r+=8;continue;
        }
#endif
        if(e-r>=4&&(p->options.backend==v3_pipeline4||(p->options.backend==v3_auto&&short4(a,r)))){
            pipeline4(a,x,y,r);r+=4;
        }else{
            int64_t end=e-r>=4?r+4:e;generic(a,x,y,r,end);r=end;
        }
    }
}
int spmv_v3_execute(const struct spmv_v3_plan *p,const struct vector *x,struct vector *y,int64_t *flops)
{
    if(!p||!p->matrix||!x||!y||!flops||x->num_values!=p->matrix->num_columns||y->num_values!=p->matrix->num_rows)return EINVAL;
    if(!p->supported)return csr_matrix_int32_spmv(p->matrix,x,y,flops);
    if(x->value_format!=vector_value_f32||y->value_format!=vector_value_f32)return EINVAL;
    if(*flops>INT64_MAX-2*p->matrix->num_nonzeros)return EOVERFLOW;
    int tid=omp_get_thread_num(),nt=omp_get_num_threads();
    int64_t b=spmv_v3_cut(p->matrix,tid,nt,p->options.work),e=spmv_v3_cut(p->matrix,tid+1,nt,p->options.work);
    rows(p,x->values,y->values,b,e);
    if(omp_in_parallel()){
#pragma omp barrier
    }
    *flops+=2*p->matrix->num_nonzeros;return 0;
}
