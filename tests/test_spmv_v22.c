/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spmv_plan.h"
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int checks,fail_after=-1;
void *__real_malloc(size_t);
void *__wrap_malloc(size_t n)
{if(fail_after==0){fail_after=-1;return NULL;}if(fail_after>0)fail_after--;return __real_malloc(n);}
static void test_case(int m,int profile,int special)
{
    int n=37;int64_t *rp=calloc((size_t)m+1,sizeof(*rp));
    for(int i=0;i<m;i++) {
        int len=profile==0?8:profile==1?i%18:profile==2?(i%100?0:8):
            profile==3?(i<m/10?9:0):profile==4?(i==0?200003:i==m/2?100003:1):0;
        rp[i+1]=rp[i]+len;
    }
    int64_t nz=rp[m];int32_t *j=malloc(((size_t)nz+1)*sizeof(*j));float *v=malloc(((size_t)nz+1)*sizeof(*v));
    for(int64_t k=0;k<nz;k++){j[k]=(k*17+3)%n;v[k]=((int)(k%17)-8)*0.125f;}
    if(special==1)for(int64_t k=0;k<nz;k++)v[k]=(float[]){1e20f,1,-1e20f,0x1p-140f,-0.0f}[k%5];
    struct csr_matrix_int32 a={m,n,rp,nz,j,csr_value_f32,v};
    float xv[37],*yv=malloc(((size_t)m+1)*sizeof(*yv));struct vector x={n,vector_value_f32,xv},y={m,vector_value_f32,yv};
    long double *ref=malloc(((size_t)m+1)*sizeof(*ref)),*bound=malloc(((size_t)m+1)*sizeof(*bound));
    for(int guarded=0;guarded<2;guarded++)for(int empty=0;empty<3;empty++) {
        struct spmv_v22_options o={guarded,empty};struct spmv_v2_options c;spmv_v2_defaults(&c);struct spmv_plan p;
        assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,3,SIZE_MAX));
        for(int i=0;i<m;i++){ref[i]=yv[i]=(i%5-2)*0.25f;bound[i]=0;}
        int prepared=omp_get_max_threads();
        for(int rep=0;rep<3;rep++) {
            for(int k=0;k<n;k++)xv[k]=(k%9+rep-4)*0.25f;
            if(special==2)xv[0]=INFINITY;
            if(special==3)xv[0]=NAN;
            if(rep==2)for(int64_t k=0;k<nz;k++)v[k]*=.5f;
            for(int i=0;i<m;i++) {
                long double sum=0,ab=0;
                for(int64_t k=rp[i];k<rp[i+1];k++){long double z=(long double)v[k]*xv[j[k]];sum+=z;ab+=fabsl(z);}
                ref[i]+=sum;bound[i]+=(rp[i+1]-rp[i]+32)*FLT_EPSILON*ab*16;
            }
            int nt=rep==0?p.v22.threads:rep==1?1:prepared+1,err=0;int64_t flops=0;
#pragma omp parallel num_threads(nt) reduction(|:err) reduction(max:flops)
            {flops=0;err=spmv_plan_execute(&p,&x,&y,&flops);}
            assert(!err&&flops==2*nz);
            for(int i=0;i<m;i++) {
                if(isnan(ref[i]))assert(isnan(yv[i]));
                else if(isinf(ref[i]))assert(isinf(yv[i])&&!!signbit(yv[i])==!!signbit(ref[i]));
                else assert(isfinite(yv[i])&&fabsl(yv[i]-ref[i])<=bound[i]+1e-35L);
            }
            checks++;
        }
        spmv_plan_free(&p);for(int64_t k=0;k<nz;k++)v[k]*=2;
    }
    free(rp);free(j);free(v);free(yv);free(ref);free(bound);
}
static void auto_empty(void)
{
    int m=100009,n=12000007;int64_t *rp=calloc((size_t)m+1,sizeof(*rp));
    for(int r=0;r<m;r++)rp[r+1]=rp[r]+(r%100==0?8:0);
    int64_t nz=rp[m];int32_t *j=calloc((size_t)nz,sizeof(*j));float *v=malloc((size_t)nz*sizeof(*v));
    float *xx=calloc((size_t)n,sizeof(*xx)),*yy=calloc((size_t)m,sizeof(*yy));assert(rp&&j&&v&&xx&&yy);
    for(int64_t k=0;k<nz;k++)v[k]=1;xx[0]=1;
    struct csr_matrix_int32 a={m,n,rp,nz,j,csr_value_f32,v};struct vector x={n,vector_value_f32,xx},y={m,vector_value_f32,yy};
    struct spmv_v22_options o={1,0};struct spmv_v2_options c;spmv_v2_defaults(&c);struct spmv_plan p;
    assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,3,SIZE_MAX));assert(p.v22.empty_runs&&p.v22.rows);
    for(int rep=0;rep<3;rep++) {
        xx[0]=rep+1;int err=0;int64_t flops=0;
#pragma omp parallel num_threads(p.v22.threads) reduction(|:err) reduction(max:flops)
        {flops=0;err=spmv_plan_execute(&p,&x,&y,&flops);}
        assert(!err&&flops==2*nz);
        for(int r=0;r<m;r++)assert(yy[r]==(r%100==0?8*(rep+1)*(rep+2)/2:0));checks++;
    }
    spmv_plan_free(&p);free(rp);free(j);free(v);free(xx);free(yy);
}
static void guards(void)
{
    int64_t rp[]={0,8,16};int32_t j[16]={0};float vv[16]={0},xx[]={1},yy[]={-0.0f,1};
    struct csr_matrix_int32 a={2,1,rp,16,j,csr_value_f32,vv};struct vector x={1,vector_value_f32,xx},y={2,vector_value_f32,yy};
    struct spmv_v22_options o={1,2};struct spmv_v2_options c;spmv_v2_defaults(&c);struct spmv_plan p;
    j[0]=1;assert(spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);j[0]=0;checks++;
    rp[1]=17;assert(spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);rp[1]=8;checks++;
    for(int f=0;f<2;f++) {
        fail_after=f;assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,3,SIZE_MAX));
        assert(!p.v22.empty_runs&&!p.v22.rows&&!p.v22.bounds);spmv_plan_free(&p);fail_after=-1;checks++;
    }
    assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,0));assert(!p.extra_bytes&&!p.v22.empty_runs);spmv_plan_free(&p);checks++;
    assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,0,0));int64_t flops=0;assert(spmv_plan_execute(&p,&x,&y,&flops)==EINVAL);spmv_plan_free(&p);checks++;
    rp[1]=rp[2]=a.num_nonzeros=0;o.empty=0;
    assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));assert(p.v22.empty_runs);
    assert(!spmv_plan_execute(&p,&x,&y,&flops));assert(!signbit(yy[0]));spmv_plan_free(&p);checks++;
    a.value_format=csr_value_f64;assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));assert(p.kernel==spmv_original);spmv_plan_free(&p);checks++;
    /* A large empty shape triggers exact delegation without a giant nnz allocation. */
    int m=4194304;int64_t *rows=calloc((size_t)m+1,sizeof(*rows));float *out=calloc((size_t)m,sizeof(float));assert(rows&&out);
    a=(struct csr_matrix_int32){m,1,rows,0,NULL,csr_value_f32,NULL};y=(struct vector){m,vector_value_f32,out};
    assert(!spmv_v22_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));assert(!strcmp(p.v22.route,"v2-large"));spmv_plan_free(&p);checks++;
    free(rows);free(out);
    int64_t huge[]={0,2147483647LL,2147483648LL,5000000000LL};a=(struct csr_matrix_int32){3,1,huge,5000000000LL,NULL,csr_value_f32,NULL};
    for(int t=1;t<=257;t++){int64_t last=0;for(int k=0;k<=t;k++){int64_t z=spmv_v2_row_cut(&a,k,t,1);assert(z>=last&&z<=3);last=z;}assert(last==3);checks++;}
}
int main(void)
{
    guards();auto_empty();
    for(int m=0;m<=35;m+=7)for(int p=0;p<6;p++)test_case(m,p,0);
    test_case(100009,2,0);test_case(201,3,0);test_case(19,1,1);test_case(19,1,2);test_case(19,1,3);
    printf("PASS v2.2: %d checks, max_threads=%d\n",checks,omp_get_max_threads());return 0;
}
