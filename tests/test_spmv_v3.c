/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "spmv_v3.h"
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#define omp_in_parallel() 0
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int checks;
static unsigned state=2026103103;
static unsigned next(void){state=1664525u*state+1013904223u;return state;}
static void testcase(int m,int len,int profile,int special)
{
    int n=37;int64_t *rp=calloc((size_t)m+1,sizeof(*rp));assert(rp);
    for(int r=0;r<m;r++){
        int L=profile==1?7+r%3:profile==2?(r==0?10001:r==18?3007:1):profile==3?(r%100?0:len):profile==4?(r==m/2?7:len):len;
        rp[r+1]=rp[r]+L;
    }
    int64_t nz=rp[m];int32_t *j=malloc(((size_t)nz+1)*sizeof(*j));float *v=malloc(((size_t)nz+1)*sizeof(*v));
    float xx[37],*yy=malloc(((size_t)m+1)*sizeof(*yy)),*ordered=malloc(((size_t)m+1)*sizeof(float));
    long double *ref=malloc(((size_t)m+1)*sizeof(*ref)),*bound=malloc(((size_t)m+1)*sizeof(*bound));assert(j&&v&&yy&&ordered&&ref&&bound);
    for(int64_t k=0;k<nz;k++){j[k]=next()%n;v[k]=((int)(next()%31)-15)*.125f;if(special==3)v[k]=(float[]){1e20f,1,-1e20f,0x1p-140f,-0.0f}[k%5];}
    struct csr_matrix_int32 a={m,n,rp,nz,j,csr_value_f32,v};struct vector x={n,vector_value_f32,xx},y={m,vector_value_f32,yy};
    for(int nt=1;nt<=25;nt+=nt==1?10:7){
        for(int work=0;work<2;work++){
            int64_t last=0;
            for(int t=0;t<=nt;t++){int64_t b=spmv_v3_cut(&a,t,nt,work);assert(b>=last&&b<=m);last=b;}assert(last==m);checks++;
        }
    }
    for(int backend=0;backend<4;backend++)for(int work=0;work<2;work++){
        struct spmv_v3_options o;spmv_v3_defaults(&o);o.backend=backend;o.work=work;o.serial_threshold=0;
        /* Each backend is also tested with SIMD disabled in a separate case. */
        o.allow_simd=special!=4;
        struct spmv_v3_plan p;assert(!spmv_v3_prepare(&p,&a,&x,&y,&o));
        for(int r=0;r<m;r++){ref[r]=ordered[r]=yy[r]=(r%7-3)*.25f;bound[r]=0;if(special==5)ref[r]=ordered[r]=yy[r]=-0.0f;}
        int maximum=omp_get_max_threads();
        for(int rep=0;rep<3;rep++){
            for(int k=0;k<n;k++)xx[k]=(k%11-5+rep)*.25f;
            if(special==1)xx[0]=INFINITY;
            if(special==2)xx[0]=NAN;
            if(rep==2)for(int64_t k=0;k<nz;k++)v[k]*=.5f;
            for(int r=0;r<m;r++){
                long double sum=0,ab=0;float exact=0;
                for(int64_t k=rp[r];k<rp[r+1];k++){long double z=(long double)v[k]*xx[j[k]];sum+=z;ab+=fabsl(z);exact+=v[k]*xx[j[k]];}
                ref[r]+=sum;ordered[r]+=exact;bound[r]+=(rp[r+1]-rp[r]+32)*FLT_EPSILON*ab*16;
            }
            int nt=rep==0?maximum:rep==1?1:maximum+1,err=0;int64_t flops=0;
#pragma omp parallel num_threads(nt) reduction(|:err) reduction(max:flops)
            {flops=0;err=spmv_v3_execute(&p,&x,&y,&flops);}
            assert(!err&&flops==2*nz);
            for(int r=0;r<m;r++){
                if(isnan(ordered[r]))assert(isnan(yy[r]));
                else if(isinf(ordered[r]))assert(isinf(yy[r])&&!!signbit(yy[r])==!!signbit(ordered[r]));
                else{assert(yy[r]==ordered[r]);if(yy[r]==0)assert(!!signbit(yy[r])==!!signbit(ordered[r]));assert(fabsl(yy[r]-ref[r])<=bound[r]+1e-35L);}
            }
            checks++;
        }
        for(int64_t k=0;k<nz;k++)v[k]*=2;
    }
    free(rp);free(j);free(v);free(yy);free(ordered);free(ref);free(bound);
}
static void guards(void)
{
    int64_t rp[]={0,8,16};int32_t j[16]={0};float v[16]={0},xx[]={1},yy[2]={-0.0f,2};
    struct csr_matrix_int32 a={2,1,rp,16,j,csr_value_f32,v};struct vector x={1,vector_value_f32,xx},y={2,vector_value_f32,yy};
    struct spmv_v3_plan p;struct spmv_v3_options o;spmv_v3_defaults(&o);
    j[0]=-1;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);j[0]=1;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);j[0]=0;checks+=2;
    rp[0]=1;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);rp[0]=0;rp[1]=17;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);rp[1]=8;checks+=2;
    o.backend=4;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);o.backend=v3_csr;o.serial_threshold=-1;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EINVAL);o.serial_threshold=0;checks+=2;
    assert(!spmv_v3_prepare(&p,&a,&x,&y,&o));int64_t flops=INT64_MAX;assert(spmv_v3_execute(&p,&x,&y,&flops)==EOVERFLOW);checks++;
    assert(spmv_v3_prepare(NULL,&a,&x,&y,&o)==EINVAL);assert(spmv_v3_prepare(&p,NULL,&x,&y,&o)==EINVAL);checks+=2;
    a.num_nonzeros=INT64_MAX;assert(spmv_v3_prepare(&p,&a,&x,&y,&o)==EOVERFLOW);a.num_nonzeros=16;checks++;
    rp[1]=rp[2]=a.num_nonzeros=0;assert(!spmv_v3_prepare(&p,&a,&x,&y,&o));flops=0;assert(!spmv_v3_execute(&p,&x,&y,&flops));assert(!signbit(yy[0])&&yy[1]==2);checks++;
    int64_t zero[]={0};a=(struct csr_matrix_int32){0,0,zero,0,NULL,csr_value_f32,NULL};x=(struct vector){0,vector_value_f32,NULL};y=x;assert(!spmv_v3_prepare(&p,&a,&x,&y,&o));assert(!spmv_v3_execute(&p,&x,&y,&flops));checks++;
    int64_t big[]={0,2147483647LL,2147483648LL,5000000000LL};a=(struct csr_matrix_int32){3,1,big,5000000000LL,NULL,csr_value_f32,NULL};
    for(int nt=1;nt<258;nt+=17){int64_t last=0;for(int t=0;t<=nt;t++){int64_t cut=spmv_v3_cut(&a,t,nt,1);assert(cut>=last&&cut<=3);last=cut;}assert(last==3);checks++;}
}
int main(void)
{
    int sizes[]={0,1,7,8,9,15,16,17,31,33,128};int lengths[]={0,1,2,4,7,8,9,16,64};
    for(unsigned i=0;i<sizeof(sizes)/sizeof(*sizes);i++)for(unsigned j=0;j<sizeof(lengths)/sizeof(*lengths);j++)testcase(sizes[i],lengths[j],0,0);
    for(int special=0;special<=5;special++)for(int profile=0;profile<=4;profile++)testcase(129,8,profile,special);
    guards();printf("PASS v3: %d checks, max_threads=%d\n",checks,omp_get_max_threads());return 0;
}
