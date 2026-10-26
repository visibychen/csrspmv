/*
Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/* SPDX-License-Identifier: GPL-3.0-or-later
 * v2.1 inspector/executor and empirical cost policy. No matrix conversion.
 * Merge coordinates use the NVIDIA-derived helper in spmv_v2.c, whose
 * BSD-3-Clause notice is retained there and in docs/merge-spmv-LICENSE.txt.
 * Delayed fragments/owner fix-up adapt that method to y+=Ax and f32.
 * Indexed AVX2 packing is an independent implementation of the method
 * documented in docs/v2.1.md (AOCL kernel templates and existing v1 pack).
 */
#define _GNU_SOURCE
#include "spmv_plan.h"
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static double clock_s(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
void spmv_v21_defaults(struct spmv_v21_options *o)
{ *o=(struct spmv_v21_options){v21_auto,NULL,0,0,1,1,0}; }
const char *spmv_v21_name(enum v21_kernel k)
{
    switch(k) {
    case v21_auto:return "auto"; case v21_original:return "original";
    case v21_csr:return "csr"; case v21_short8:return "short8";
    case v21_merge:return "merge"; case v21_long:return "long";
    } return "invalid";
}
static void hash_bytes(unsigned long long *h,const void *p,size_t n)
{ const unsigned char *b=p; for(size_t i=0;i<n;i++) {*h^=b[i];*h*=1099511628211ULL;} }
unsigned long long spmv_v21_fingerprint(void)
{
    unsigned long long h=14695981039346656037ULL;
    const char *build=__VERSION__
#ifdef __OPTIMIZE__
        ":optimized"
#endif
#ifdef __AVX2__
        ":avx2"
#endif
#ifdef __AVX512F__
        ":avx512"
#endif
#ifdef __FAST_MATH__
        ":fastmath"
#endif
        ":v21-abi1-contract-off";
    hash_bytes(&h,build,strlen(build));
    FILE *f=fopen("/proc/cpuinfo","r"); char line[512];
    if(f) {while(fgets(line,sizeof(line),f)) if(!strncmp(line,"model name",10)) {
        hash_bytes(&h,line,strlen(line));break;} fclose(f);}
    const char *names[]={"OMP_PLACES","OMP_PROC_BIND","OMP_WAIT_POLICY","OMP_DYNAMIC"};
    for(size_t i=0;i<sizeof(names)/sizeof(*names);i++) {
        const char *v=getenv(names[i]); if(!v) v="<unset>";
        hash_bytes(&h,v,strlen(v)+1);
    }
    int np=omp_get_num_places(),procs=omp_get_num_procs();
    hash_bytes(&h,&np,sizeof(np));hash_bytes(&h,&procs,sizeof(procs));
    for(int i=0;i<np;i++) {
        int n=omp_get_place_num_procs(i);
        hash_bytes(&h,&n,sizeof(n));
        if(n>0 && n<=4096) {
            int ids[4096];omp_get_place_proc_ids(i,ids);
            hash_bytes(&h,ids,(size_t)n*sizeof(*ids));
        }
    }
    return h;
}
struct cost { int regime,kernel,threads; double nz,rows,barrier,launch; };
struct profile {
    unsigned long long fingerprint;
    int64_t cache_bytes,long_threshold;
    double margin;
    int count,preps;
    struct cost models[V21_MAX_MODELS],prep[64];
};
static int finite_cost(const struct cost *m)
{ return isfinite(m->nz)&&isfinite(m->rows)&&isfinite(m->barrier)&&isfinite(m->launch)
    &&m->nz>=0&&m->rows>=0&&m->barrier>=0&&m->launch>=0&&m->threads>0&&m->threads<=4096; }
static int load_profile(const char *path,struct profile *q)
{
    memset(q,0,sizeof(*q));FILE *f=fopen(path,"r");if(!f) return errno;
    char line[512],extra;int version;
    if(!fgets(line,sizeof(line),f)||sscanf(line,"SPMV21 %d %llx %ld %ld %lf %c",
        &version,&q->fingerprint,&q->cache_bytes,&q->long_threshold,&q->margin,&extra)!=5
        ||version!=1||q->cache_bytes<0||q->long_threshold<8||!isfinite(q->margin)
        ||q->margin<0||q->margin>1) {fclose(f);return EINVAL;}
    while(fgets(line,sizeof(line),f)) {
        if(line[0]=='#'||line[0]=='\n') continue;
        struct cost m={0};int ok=0;
        if(line[0]=='M') {
            ok=sscanf(line,"M %d %d %d %lf %lf %lf %lf %c",&m.regime,&m.kernel,&m.threads,
                &m.nz,&m.rows,&m.barrier,&m.launch,&extra)==7;
            if(!ok||!finite_cost(&m)||m.regime<0||m.regime>1||m.kernel<1||m.kernel>5
                ||q->count>=V21_MAX_MODELS) {fclose(f);return EINVAL;}
            for(int i=0;i<q->count;i++) if(q->models[i].regime==m.regime&&
                q->models[i].kernel==m.kernel&&q->models[i].threads==m.threads) {
                fclose(f);return EINVAL;
            }
            q->models[q->count++]=m;
        } else if(line[0]=='P') {
            ok=sscanf(line,"P %d %lf %lf %lf %c",&m.threads,&m.nz,&m.rows,&m.launch,&extra)==4;
            if(!ok||!finite_cost(&m)||q->preps>=64) {fclose(f);return EINVAL;}
            for(int i=0;i<q->preps;i++) if(q->prep[i].threads==m.threads) {fclose(f);return EINVAL;}
            q->prep[q->preps++]=m;
        } else {fclose(f);return EINVAL;}
    }
    int err=ferror(f)?EIO:0;fclose(f);return err;
}
static int64_t cut_total(int64_t n,int t,int nt)
{ return n/nt*t+(t<n%nt?t:n%nt); }
static void validate_row(const struct csr_matrix_int32 *a,int64_t r,
    int *bad,int *all8,int64_t *longest,int64_t *empty)
{
    int64_t b=a->row_ptr[r],e=a->row_ptr[r+1];
    if(b<0||e<b||e>a->num_nonzeros) {*bad=1;return;}
    int64_t len=e-b;*all8&=len==8;if(len>*longest) *longest=len;*empty+=len==0;
}
static void loads(const struct csr_matrix_int32 *a,int nt,int work,int64_t *nz,int64_t *rows)
{
    *nz=*rows=0;
    for(int t=0;t<nt;t++) {
        int64_t b=spmv_v2_row_cut(a,t,nt,work),e=spmv_v2_row_cut(a,t+1,nt,work);
        int64_t n=a->row_ptr[e]-a->row_ptr[b];
        if(n>*nz) *nz=n;
        if(e-b>*rows) *rows=e-b;
    }
}
static double estimate(const struct cost *c,int64_t nz,int64_t rows,int64_t calls)
{ return c->launch+calls*(c->nz*nz+c->rows*rows+c->barrier); }
static void bind_slots(struct spmv_plan *p)
{
    struct spmv_v21_state *s=&p->v21;const struct csr_matrix_int32 *a=p->matrix;
    int nt=s->threads;
    for(int t=0;t<nt;t++) {
        struct v21_slot *z=&s->slots[t];memset(z,0,sizeof(*z));
        if(s->selected==v21_merge) {
            spmv_v2_merge_cut(a,cut_total(a->num_rows+a->num_nonzeros,t,nt),&z->row,&z->nz);
            spmv_v2_merge_cut(a,cut_total(a->num_rows+a->num_nonzeros,t+1,nt),&z->end_row,&z->end_nz);
            int64_t k=z->nz;
            if(z->row<z->end_row) {
                if(k>a->row_ptr[z->row]) z->fragment[z->count++].row=z->row;
                k=a->row_ptr[z->end_row];
            }
            if(k<z->end_nz) z->fragment[z->count++].row=z->end_row;
        } else {
            z->row=spmv_v2_row_cut(a,t,nt,s->work);
            z->end_row=spmv_v2_row_cut(a,t+1,nt,s->work);
        }
    }
    s->groups=0;
    if(s->selected==v21_merge) for(int t=0;t<nt;t++) for(int f=0;f<s->slots[t].count;f++) {
        int64_t r=s->slots[t].fragment[f].row;int pos=2*t+f;
        if(!s->groups||s->group[s->groups-1].row!=r)
            s->group[s->groups++]=(struct v21_group){r,pos,pos};
        else s->group[s->groups-1].last=pos;
    }
}
int spmv_v21_prepare(struct spmv_plan *p,const struct csr_matrix_int32 *a,
    const struct vector *x,const struct vector *y,const struct spmv_v2_options *common,
    const struct spmv_v21_options *o,int64_t calls,size_t budget)
{
    memset(p,0,sizeof(*p));p->matrix=a;p->kernel=spmv_v21;p->reason="forced";
    struct spmv_v21_state *s=&p->v21;s->options=*o;s->common=*common;s->expected_calls=calls;
    int max_threads=omp_get_max_threads();s->threads=max_threads;s->prepare_threads=1;
    s->long_threshold=64;
    if(o->kernel<0||o->kernel>v21_long||o->threads<0||o->threads>max_threads||
       o->prepare_mode<0||o->prepare_mode>2||(o->cache!=0&&o->cache!=1)||
       (o->owners!=0&&o->owners!=1)||(o->diagnostics!=0&&o->diagnostics!=1)||calls<0||
       common->partition<0||common->partition>v2_work||common->execution<0||
       common->execution>v2_parallel||(common->prefetch!=0&&common->prefetch!=32&&common->prefetch!=128)||
       (common->strict_order&&(o->kernel==v21_merge||o->kernel==v21_long||o->kernel==v21_original))||
       (o->kernel==v21_merge&&common->partition!=v2_part_auto)) return EINVAL;
    if(a->num_rows<0||a->num_columns<0||a->num_nonzeros<0||x->num_values!=a->num_columns||
       y->num_values!=a->num_rows) return EINVAL;
    if(a->num_nonzeros>INT64_MAX/2||a->num_nonzeros>INT64_MAX-a->num_rows) return EOVERFLOW;
    if(a->value_format!=csr_value_f32||x->value_format!=vector_value_f32||y->value_format!=vector_value_f32) {
        p->kernel=spmv_original;p->reason="unsupported-types";return 0;
    }
    if(!a->row_ptr||(a->num_nonzeros&&(!a->column_indices||!a->values))||
       (x->num_values&&!x->values)||(y->num_values&&!y->values)) return EINVAL;
    if(!calls) {s->threads=1;s->selected=v21_csr;p->reason="zero-calls";return 0;}
    if(a->row_ptr[0]||a->row_ptr[a->num_rows]!=a->num_nonzeros) return EINVAL;
    s->fingerprint=spmv_v21_fingerprint();struct profile q;memset(&q,0,sizeof(q));
    if(o->profile) {
        int err=load_profile(o->profile,&q);if(err)return err;
        s->profile_match=q.fingerprint==s->fingerprint;
        if(s->profile_match) s->long_threshold=q.long_threshold;
    }
#if defined(__x86_64__) || defined(__i386__)
    s->avx2=!!__builtin_cpu_supports("avx2");
#endif
    if(s->profile_match) {
        double best=DBL_MAX;
        for(int i=0;i<q.preps;i++) if(q.prep[i].threads<=max_threads) {
            double e=estimate(&q.prep[i],a->num_nonzeros,a->num_rows,1);
            if(e<best){best=e;s->prepare_threads=q.prep[i].threads;}
        }
    }
    if(o->prepare_mode) s->prepare_threads=o->prepare_mode==1?1:max_threads;
    int bad=0,all8=1;int64_t longest=0,empty=0;double start=clock_s();
    if(s->prepare_threads==1) {
        for(int64_t r=0;r<a->num_rows;r++) validate_row(a,r,&bad,&all8,&longest,&empty);
        for(int64_t k=0;k<a->num_nonzeros;k++)
            if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns) bad=1;
    } else {
        int nt=s->prepare_threads;
#pragma omp parallel num_threads(nt) reduction(|:bad) reduction(&:all8) reduction(max:longest) reduction(+:empty)
        {
#pragma omp for schedule(static)
            for(int64_t r=0;r<a->num_rows;r++) validate_row(a,r,&bad,&all8,&longest,&empty);
#pragma omp for schedule(static)
            for(int64_t k=0;k<a->num_nonzeros;k++)
                if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns) bad=1;
        }
    }
    s->validation_s=clock_s()-start;if(bad)return EINVAL;
    s->all8=all8;s->longest=longest;s->empty_rows=empty;start=clock_s();
    s->selected=common->strict_order?v21_csr:v21_original;
    s->work=common->partition==v2_work;
    if(o->threads) s->threads=o->threads;
    if(common->execution==v2_serial) s->threads=1;
    if(o->kernel==v21_auto) {
        p->reason=s->profile_match?"profile-conservative":o->profile?"profile-mismatch":"no-profile";
        if(s->profile_match) {
            long double bytes=8.0L*a->num_nonzeros+8.0L*(a->num_rows+1LL)+4.0L*(a->num_rows+(int64_t)a->num_columns);
            int regime=bytes>q.cache_bytes;double baseline=DBL_MAX,best=DBL_MAX;
            enum v21_kernel chosen=s->selected;int chosen_t=s->threads,chosen_w=s->work;
            for(int i=0;i<q.count;i++) {
                const struct cost *c=&q.models[i];int nt=c->threads;enum v21_kernel k=c->kernel;
                if(c->regime!=regime||nt>max_threads||(o->threads&&nt!=o->threads)||
                   (common->execution==v2_serial&&nt!=1)||
                   (common->execution==v2_parallel&&nt!=s->threads)||
                   (common->strict_order&&(k==v21_merge||k==v21_long||k==v21_original))||
                   (k==v21_short8&&!all8)||(k==v21_long&&(!s->avx2||longest<s->long_threshold))||
                   (k==v21_merge&&(nt==1||common->partition!=v2_part_auto))) continue;
                for(int work=0;work<=1;work++) {
                    if(k==v21_original&&work)continue;
                    if(common->partition==v2_rows&&work)continue;
                    if(common->partition==v2_work&&!work&&k!=v21_original)continue;
                    if(k==v21_merge&&work)continue;
                    int64_t nz,rows;
                    if(k==v21_merge) {nz=(a->num_nonzeros+nt-1)/nt;rows=(a->num_rows+nt-1LL)/nt;}
                    else loads(a,nt,work,&nz,&rows);
                    double e=estimate(c,nz,rows,calls);
                    if(!isfinite(e))continue;
                    if(nt==s->threads&&k==s->selected&&work==s->work)baseline=e;
                    if(e<best){best=e;chosen=k;chosen_t=nt;chosen_w=work;}
                }
            }
            s->baseline_predicted_s=baseline==DBL_MAX?0:baseline;
            s->predicted_s=best==DBL_MAX?0:best;
            if(baseline!=DBL_MAX&&best<baseline*(1-q.margin)) {
                s->selected=chosen;s->threads=chosen_t;s->work=chosen_w;p->reason="profile-cost";
            }
        }
    } else s->selected=o->kernel;
    if(s->selected==v21_short8&&!all8){s->selected=v21_csr;p->reason="not-all-eight";}
    if(s->selected==v21_long&&!s->avx2){s->selected=v21_csr;p->reason="no-avx2";}
    if(s->selected!=v21_merge&&common->partition==v2_part_auto&&o->kernel!=v21_auto) {
        int64_t n,r;loads(a,s->threads,0,&n,&r);
        s->work=(long double)(n+r)*s->threads>1.5L*(a->num_rows+a->num_nonzeros);
    }
    if(s->selected==v21_original)s->work=0;
    if(s->selected!=v21_original&&(o->cache||s->selected==v21_merge||o->diagnostics)) {
        size_t nt=(size_t)s->threads;
        size_t extra=nt*(sizeof(*s->slots)+2*sizeof(*s->group));
        if(nt>SIZE_MAX/(sizeof(*s->slots)+2*sizeof(*s->group))||extra>budget) {
            s->selected=v21_csr;p->reason="plan-budget";
        } else {
            s->slots=malloc(nt*sizeof(*s->slots));s->group=malloc(2*nt*sizeof(*s->group));
            if(!s->slots||!s->group) {
                free(s->slots);free(s->group);s->slots=NULL;s->group=NULL;
                s->selected=v21_csr;p->reason="plan-allocation";
            } else {p->extra_bytes=extra;bind_slots(p);}
        }
    }
    s->policy_s=clock_s()-start;return 0;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize,no-tree-slp-vectorize")))
#endif
static float ordered_sum(const float *v,const int32_t *j,const float *x,int64_t b,int64_t e,int d,int64_t stop)
{
    float sum=0;
    for(int64_t k=b;k<e;k++) {
        if(d&&d<stop-k)__builtin_prefetch(x+j[k+d],0,1);
        sum+=v[k]*x[j[k]];
    } return sum;
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2")))
static float packed_sum(const float *v,const int32_t *j,const float *x,int64_t k,int64_t e)
{
    __m256 acc=_mm256_setzero_ps();
    for(;e-k>=8;k+=8) {
        __m256 vx=_mm256_set_ps(x[j[k+7]],x[j[k+6]],x[j[k+5]],x[j[k+4]],
            x[j[k+3]],x[j[k+2]],x[j[k+1]],x[j[k]]);
        acc=_mm256_add_ps(acc,_mm256_mul_ps(_mm256_loadu_ps(v+k),vx));
    }
    float lanes[8];_mm256_storeu_ps(lanes,acc);float sum=0;
    for(int l=0;l<8;l++)sum+=lanes[l];
    for(;k<e;k++)sum+=v[k]*x[j[k]];
    return sum;
}
#endif
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize,no-tree-slp-vectorize")))
#endif
static void row_kernel(const struct spmv_plan *p,const float *x,float *y,int tid,int nt)
{
    const struct spmv_v21_state *s=&p->v21;const struct csr_matrix_int32 *a=p->matrix;
    int cached=s->options.cache&&s->slots&&nt==s->threads;
    int64_t b=cached?s->slots[tid].row:spmv_v2_row_cut(a,tid,nt,s->work);
    int64_t e=cached?s->slots[tid].end_row:spmv_v2_row_cut(a,tid+1,nt,s->work);
    const float *v=a->values;const int32_t *j=a->column_indices;
    for(int64_t r=b;r<e;r++) {
        int64_t k=s->selected==v21_short8?8*r:a->row_ptr[r],end=a->row_ptr[r+1];float sum=0;
        if(s->selected==v21_short8) {
            if(s->common.prefetch&&s->common.prefetch<a->row_ptr[e]-k)
                __builtin_prefetch(x+j[k+s->common.prefetch],0,1);
            sum+=v[k]*x[j[k]];sum+=v[k+1]*x[j[k+1]];
            sum+=v[k+2]*x[j[k+2]];sum+=v[k+3]*x[j[k+3]];
            sum+=v[k+4]*x[j[k+4]];sum+=v[k+5]*x[j[k+5]];
            sum+=v[k+6]*x[j[k+6]];sum+=v[k+7]*x[j[k+7]];
            y[r]+=sum;continue;
        }
#if defined(__x86_64__) || defined(__i386__)
        if(s->selected==v21_long&&s->avx2&&end-k>=s->long_threshold) sum=packed_sum(v,j,x,k,end);
        else
#endif
        sum=ordered_sum(v,j,x,k,end,s->common.prefetch,a->row_ptr[e]);
        y[r]+=sum;
    }
}
static void fix_group(const struct spmv_v21_state *s,int g,float *y)
{
    const struct v21_group *z=&s->group[g];float sum=0;
    for(int i=z->first;i<=z->last;i++) {
        const struct v21_slot *t=&s->slots[i/2];int f=i%2;
        if(f<t->count&&t->fragment[f].row==z->row)sum+=t->fragment[f].sum;
    } y[z->row]+=sum;
}
static void merge_kernel(const struct spmv_plan *p,const float *x,float *y,int tid,int nt)
{
    const struct spmv_v21_state *s=&p->v21;const struct csr_matrix_int32 *a=p->matrix;
    struct v21_slot *z=&s->slots[tid];int64_t r=z->row,k=z->nz,er=z->end_row,ek=z->end_nz;
    if(!s->options.cache) {
        spmv_v2_merge_cut(a,cut_total(a->num_rows+a->num_nonzeros,tid,nt),&r,&k);
        spmv_v2_merge_cut(a,cut_total(a->num_rows+a->num_nonzeros,tid+1,nt),&er,&ek);
    }
    const float *v=a->values;const int32_t *j=a->column_indices;int f=0;
    double c0=s->options.diagnostics?clock_s():0;
    for(;r<er;r++) {
        int partial=k>a->row_ptr[r];float sum=ordered_sum(v,j,x,k,a->row_ptr[r+1],0,0);k=a->row_ptr[r+1];
        if(partial)z->fragment[f++].sum=sum;else y[r]+=sum;
    }
    if(k<ek)z->fragment[f++].sum=ordered_sum(v,j,x,k,ek,0,0);
    double c1=s->options.diagnostics?clock_s():0;
#pragma omp barrier
    double c2=s->options.diagnostics?clock_s():0;
    if(s->options.owners) {
        for(int g=tid;g<s->groups;g+=nt)fix_group(s,g,y);
    } else if(tid==0) for(int g=0;g<s->groups;g++)fix_group(s,g,y);
    double c3=s->options.diagnostics?clock_s():0;
#pragma omp barrier
    if(s->options.diagnostics) {
        z->compute_s+=c1-c0;z->publish_s+=c2-c1;z->fixup_s+=c3-c2;z->complete_s+=clock_s()-c3;
    }
}
int spmv_v21_execute(const struct spmv_plan *p,const struct vector *x,struct vector *y,int64_t *flops)
{
    const struct spmv_v21_state *s=&p->v21;const struct csr_matrix_int32 *a=p->matrix;
    if(!s->expected_calls)return EINVAL;
    if(x->value_format!=vector_value_f32||y->value_format!=vector_value_f32)return EINVAL;
    if(*flops>INT64_MAX-2*a->num_nonzeros)return EOVERFLOW;
    if(s->selected==v21_original)return csr_matrix_int32_spmv(a,x,y,flops);
    int tid=omp_get_thread_num(),nt=omp_get_num_threads();
    if(s->selected==v21_merge&&nt==s->threads&&nt>1&&s->slots) merge_kernel(p,x->values,y->values,tid,nt);
    else {
        double c0=s->options.diagnostics?clock_s():0;
        row_kernel(p,x->values,y->values,tid,nt);
        double c1=s->options.diagnostics?clock_s():0;
        if(omp_in_parallel()) {
#pragma omp barrier
        }
        if(s->options.diagnostics&&s->slots&&nt==s->threads) {
            s->slots[tid].compute_s+=c1-c0;s->slots[tid].complete_s+=clock_s()-c1;
        }
    }
    *flops+=2*a->num_nonzeros;return 0;
}
void spmv_v21_report(const struct spmv_plan *p,FILE *f)
{
    const struct spmv_v21_state *s=&p->v21;
    fprintf(f,"spmv_v21: kernel=%s partition=%s threads=%d prepare_threads=%d calls=%ld profile_match=%d fingerprint=%016llx empty_rows=%ld longest=%ld validation_s=%.9f policy_s=%.9f predicted_s=%.9f baseline_predicted_s=%.9f groups=%d\n",
        spmv_v21_name(s->selected),s->work?"work":"rows",s->threads,s->prepare_threads,
        s->expected_calls,s->profile_match,s->fingerprint,s->empty_rows,s->longest,
        s->validation_s,s->policy_s,s->predicted_s,s->baseline_predicted_s,s->groups);
    if(s->options.diagnostics&&s->slots)for(int t=0;t<s->threads;t++) {
        const struct v21_slot *z=&s->slots[t];
        fprintf(f,"spmv_v21_thread: thread=%d compute_s=%.9f publish_s=%.9f fixup_s=%.9f complete_s=%.9f\n",
            t,z->compute_s,z->publish_s,z->fixup_s,z->complete_s);
    }
}
