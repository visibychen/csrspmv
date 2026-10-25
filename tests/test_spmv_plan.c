/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spmv_plan.h"
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Link with -Wl,--wrap=malloc: inject failures into each plan allocation. */
static int fail_after = -1;
void *__real_malloc(size_t);
void *__wrap_malloc(size_t n)
{
    if (fail_after == 0) { fail_after = -1; return NULL; }
    if (fail_after > 0) fail_after--;
    return __real_malloc(n);
}
static unsigned rng = 20261025;
static unsigned next(void) { rng = rng*1664525u+1013904223u; return rng; }
static int checks = 0;

static void run_case(int m, int n, int len, int ragged, int special)
{
    struct csr_matrix_int32 a = {0};
    a.num_rows=m; a.num_columns=n; a.value_format=csr_value_f32;
    a.row_ptr=calloc((size_t)m+1,sizeof(int64_t));
    for (int r=0;r<m;r++) a.row_ptr[r+1]=a.row_ptr[r]+(n ?
        (special==4 && r==0 ? 0 : (ragged ? next()%(len+1) : len)) : 0);
    a.num_nonzeros=a.row_ptr[m];
    a.column_indices=malloc((size_t)a.num_nonzeros*sizeof(int32_t));
    a.values=malloc((size_t)a.num_nonzeros*sizeof(float));
    for (int64_t k=0;k<a.num_nonzeros;k++) {
        a.column_indices[k]=next()%n; /* unsorted, repeated, including col zero */
        ((float *)a.values)[k]=special==4 ? 1.0f : ((int)(next()%31)-15)*0.125f;
    }
    struct vector x={n,vector_value_f32,calloc((size_t)n+1,sizeof(float))};
    struct vector y={m,vector_value_f32,calloc((size_t)m+1,sizeof(float))};
    if (special==2) { /* strong cancellation plus subnormal and signed zero */
        for (int64_t k=0;k<a.num_nonzeros;k++)
            ((float *)a.values)[k]=(float[]){1e20f,1.0f,-1e20f,0x1p-140f,-0.0f}[k%5];
    }
    for (int kernel=0;kernel<4;kernel++) for (int c=8;c<=16;c+=8) {
        for (int r=0;r<m;r++) ((float *)y.values)[r]=(r%7-3)*0.25f;
        long double *ref=malloc(((size_t)m+1)*sizeof(long double));
        long double *bound=calloc((size_t)m+1,sizeof(long double));
        for (int r=0;r<m;r++) ref[r]=((float *)y.values)[r];
        struct spmv_plan p;
        assert(!spmv_plan_prepare(&p,&a,&x,&y,kernel,c,SIZE_MAX));
        for (int rep=0;rep<3;rep++) {
            for (int j=0;j<n;j++) ((float *)x.values)[j]=(j%11-5+rep)*0.25f;
            if ((special==1 || special==4) && n) ((float *)x.values)[0]=INFINITY;
            if (special==3 && n) ((float *)x.values)[0]=NAN;
            for (int r=0;r<m;r++) {
                long double sum=0,ab=0;
                for (int64_t k=a.row_ptr[r];k<a.row_ptr[r+1];k++) {
                    long double prod=(long double)((float *)a.values)[k]*((float *)x.values)[a.column_indices[k]];
                    sum+=prod; ab+=fabsl(prod);
                }
                ref[r]+=sum;
                /* Forward error allowance using row length and sum |a*x|,
                 * not relative error against a cancellation-small result. */
                bound[r]+=(a.row_ptr[r+1]-a.row_ptr[r]+4)*FLT_EPSILON*ab*4;
            }
            int err=0; int64_t flops=0;
#pragma omp parallel reduction(|:err) reduction(max:flops)
            { flops=0; err=spmv_plan_execute(&p,&x,&y,&flops); }
            assert(!err && flops==2*a.num_nonzeros);
            for (int r=0;r<m;r++) {
                long double got=((float *)y.values)[r];
                if (isnan(ref[r])) assert(isnan(got));
                else if (isinf(ref[r])) assert(isinf(got) && !!signbit(got)==!!signbit(ref[r]));
                else assert(isfinite(got) && fabsl(got-ref[r])<=bound[r]+1e-35L);
            }
            checks++;
        }
        spmv_plan_free(&p); free(ref); free(bound);
    }
    free(a.row_ptr); free(a.column_indices); free(a.values); free(x.values); free(y.values);
}

static void guards(void)
{
    int64_t rows[]={0,8,16,24,32,40,48,56,64};
    int32_t cols[64]={0}; float vals[64]={0}, xv[1]={1}, yv[8]={0};
    struct csr_matrix_int32 a={8,1,rows,64,cols,csr_value_f32,vals};
    struct vector x={1,vector_value_f32,xv},y={8,vector_value_f32,yv};
    struct spmv_plan p;
    assert(!spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,0));
    assert(p.kernel==spmv_original && !strcmp(p.reason,"memory-budget"));
    spmv_plan_free(&p); checks++;
    for (int i=0;i<3;i++) {
        fail_after=i;
        assert(!spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX));
        assert(p.kernel==spmv_original && !strcmp(p.reason,"allocation-failed"));
        spmv_plan_free(&p); checks++;
    }
    cols[0]=-1;
    assert(spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX)==EINVAL); checks++;
    cols[0]=0; rows[2]=7;
    assert(spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX)==EINVAL); checks++;
    rows[2]=16;
    a.num_nonzeros=INT64_MAX;
    assert(spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX)==EOVERFLOW); checks++;
    a.num_nonzeros=64; y.num_values=7;
    assert(spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX)==EINVAL); checks++;
    y.num_values=8; a.value_format=csr_value_f64;
    assert(!spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX));
    assert(p.kernel==spmv_original); spmv_plan_free(&p); checks++;
    a.value_format=csr_value_f32;
    rows[1]=64; for(int i=2;i<9;i++)rows[i]=64;
    assert(!spmv_plan_prepare(&p,&a,&x,&y,spmv_sell,8,SIZE_MAX));
    assert(p.kernel==spmv_original && !strcmp(p.reason,"padding-over-2x"));
    spmv_plan_free(&p); checks++;
}

int main(void)
{
    guards();
    int lens[]={0,1,2,4,7,8,9,15,16,17,32,64,256};
    for (int i=0;i<13;i++) {
        run_case(19,37,lens[i],0,0);
        run_case(33,7,lens[i],1,0);
    }
    run_case(0,7,0,0,0); run_case(7,0,0,0,0);
    run_case(3,9,1,1,1); run_case(17,13,9,1,1);
    run_case(17,13,9,1,3); run_case(19,13,17,1,2);
    run_case(31,1,8,0,4); run_case(7,9,8,0,0); run_case(9,9,8,0,0);
    printf("PASS: %d checks, max_threads=%d\n",checks,omp_get_max_threads());
    return 0;
}
