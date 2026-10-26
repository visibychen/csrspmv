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
#include <unistd.h>
static int checks,fail_after=-1;
void *__real_malloc(size_t);
void *__wrap_malloc(size_t n)
{
    if(fail_after==0){fail_after=-1;return NULL;}
    if(fail_after>0)fail_after--;
    return __real_malloc(n);
}
static void test_case(int m,int profile,int special)
{
    int n=37;int64_t *rp=calloc((size_t)m+1,sizeof(*rp));
    for(int i=0;i<m;i++) {
        int len=profile==0?8:profile==1?i%18:profile==2?(i%17?0:3):
            profile==3?(i==1||i==m/2?10007:1):0;
        rp[i+1]=rp[i]+len;
    }
    int64_t nnz=rp[m];int32_t *col=malloc(((size_t)nnz+1)*sizeof(*col));
    float *val=malloc(((size_t)nnz+1)*sizeof(*val));
    for(int64_t i=0;i<nnz;i++) {col[i]=(i*17+3)%n;val[i]=((int)(i%17)-8)*0.125f;}
    if(special==1)for(int64_t i=0;i<nnz;i++)val[i]=(float[]){1e20f,1,-1e20f,0x1p-140f,-0.0f}[i%5];
    struct csr_matrix_int32 a={m,n,rp,nnz,col,csr_value_f32,val};
    float xv[37],*yv=malloc(((size_t)m+1)*sizeof(*yv));
    struct vector x={n,vector_value_f32,xv},y={m,vector_value_f32,yv};
    long double *ref=malloc(((size_t)m+1)*sizeof(*ref)),*bound=calloc((size_t)m+1,sizeof(*bound));
    for(int kernel=0;kernel<=5;kernel++)for(int cache=0;cache<=1;cache++)for(int owners=0;owners<=1;owners++) {
        struct spmv_v21_options o;spmv_v21_defaults(&o);o.kernel=kernel;o.cache=cache;o.owners=owners;o.diagnostics=1;
        struct spmv_v2_options common;spmv_v2_defaults(&common);
        struct spmv_plan p;assert(!spmv_v21_prepare(&p,&a,&x,&y,&common,&o,3,SIZE_MAX));
        for(int i=0;i<m;i++){yv[i]=(i%5-2)*0.25f;ref[i]=yv[i];bound[i]=0;}
        int prepared=omp_get_max_threads();
        for(int rep=0;rep<3;rep++) {
            for(int j=0;j<n;j++)xv[j]=(j%9+rep-4)*0.25f;
            if(special==2)xv[0]=INFINITY;
            if(special==3)xv[0]=NAN;
            if(rep==2)for(int64_t j=0;j<nnz;j++)val[j]*=0.5f;
            for(int i=0;i<m;i++) {
                long double sum=0,absum=0;
                for(int64_t j=rp[i];j<rp[i+1];j++) {
                    long double v=(long double)val[j]*xv[col[j]];sum+=v;absum+=fabsl(v);
                }
                ref[i]+=sum;bound[i]+=(rp[i+1]-rp[i]+32)*FLT_EPSILON*absum*16;
            }
            /* Both smaller and larger actual teams exercise uniform fallback. */
            int nt=rep==0?prepared:rep==1?1:prepared+1,err=0;int64_t flops=0;
#pragma omp parallel num_threads(nt) reduction(|:err) reduction(max:flops)
            {flops=0;err=spmv_plan_execute(&p,&x,&y,&flops);}
            assert(!err&&flops==2*nnz);
            for(int i=0;i<m;i++) {
                if(isnan(ref[i]))assert(isnan(yv[i]));
                else if(isinf(ref[i]))assert(isinf(yv[i])&&!!signbit(yv[i])==!!signbit(ref[i]));
                else assert(isfinite(yv[i])&&fabsl(yv[i]-ref[i])<=bound[i]+1e-35L);
            }
            checks++;
        }
        spmv_plan_free(&p);
        for(int64_t j=0;j<nnz;j++)val[j]*=2;
    }
    free(rp);free(col);free(val);free(yv);free(ref);free(bound);
}
static void guards(void)
{
    int64_t rp[]={0,8,16};int32_t col[16]={0};float v[16]={0},xx[]={1},yy[]={-0.0f,1};
    struct csr_matrix_int32 a={2,1,rp,16,col,csr_value_f32,v};
    struct vector x={1,vector_value_f32,xx},y={2,vector_value_f32,yy};
    struct spmv_v21_options o;spmv_v21_defaults(&o);
    struct spmv_v2_options c;spmv_v2_defaults(&c);struct spmv_plan p;
    col[0]=1;assert(spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);col[0]=0;checks++;
    rp[1]=17;assert(spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);rp[1]=8;checks++;
    assert(spmv_v21_prepare(&p,&a,&x,&y,&c,&o,-1,SIZE_MAX)==EINVAL);checks++;
    o.kernel=v21_merge;c.strict_order=1;assert(spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);c.strict_order=0;checks++;
    for(int fail=0;fail<2;fail++) {
        fail_after=fail;assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));
        assert(p.v21.selected==v21_csr&&!p.v21.slots&&!p.v21.group);spmv_plan_free(&p);fail_after=-1;checks++;
    }
    assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,0));assert(p.v21.selected==v21_csr&&!p.extra_bytes);spmv_plan_free(&p);checks++;
    o.kernel=v21_csr;assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,0,0));assert(p.extra_bytes==0&&p.v21.threads==1);
    int64_t zero=0;assert(spmv_plan_execute(&p,&x,&y,&zero)==EINVAL);spmv_plan_free(&p);checks++;
    rp[1]=rp[2]=a.num_nonzeros=0;c.execution=v2_serial;
    assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));int64_t flops=0;
    assert(!spmv_plan_execute(&p,&x,&y,&flops));assert(!signbit(yy[0]));spmv_plan_free(&p);checks++;
    char file[]="/tmp/spmv-v21-test-XXXXXX";int fd=mkstemp(file);assert(fd>=0);close(fd);o.profile=file;o.kernel=v21_auto;
    FILE *f=fopen(file,"w");assert(f);fputs("SPMV21 1 0000000000000000 8388608 64 0.1\n",f);fclose(f);
    assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));assert(!p.v21.profile_match&&p.v21.selected==v21_original);spmv_plan_free(&p);checks++;
    const char *bad[]={"bad\n","SPMV21 1 0 8 64 nan\n","SPMV21 1 0 8 64 0.1\nM 0 1 1 nan 0 0 0\n", "SPMV21 1 0 8 64 0.1\nP 0 0 0 0\n"};
    for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) {
        f=fopen(file,"w");fputs(bad[i],f);fclose(f);
        assert(spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX)==EINVAL);checks++;
    }
    /* Synthetic coefficients test the parser and call-count decision, not speed. */
    c.execution=v2_exec_auto;
    f=fopen(file,"w");assert(f);
    fprintf(f,"SPMV21 1 %016llx 8388608 64 0.1\nM 0 1 %d 0 0 2 0\nM 0 2 1 0 0 1 20\nP 1 0 0 0\n",
        spmv_v21_fingerprint(),omp_get_max_threads());fclose(f);
    assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));
    assert(p.v21.profile_match&&p.v21.selected==v21_original);spmv_plan_free(&p);checks++;
    assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,100,SIZE_MAX));
    assert(p.v21.selected==v21_csr&&p.v21.threads==1);spmv_plan_free(&p);checks++;
    unlink(file);o.profile=NULL;c.execution=v2_exec_auto;
    a.value_format=csr_value_f64;assert(!spmv_v21_prepare(&p,&a,&x,&y,&c,&o,1,SIZE_MAX));assert(p.kernel==spmv_original);spmv_plan_free(&p);checks++;
}
int main(void)
{
    guards();
    for(int m=0;m<=35;m+=7)for(int profile=0;profile<5;profile++)test_case(m,profile,0);
    test_case(19,1,1);test_case(3,3,0);test_case(19,1,2);test_case(19,1,3);
    printf("PASS v2.1: %d checks, max_threads=%d\n",checks,omp_get_max_threads());return 0;
}
