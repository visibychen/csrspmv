/* SPDX-License-Identifier: GPL-3.0-or-later
 * Guarded extensions of v2; original CSR and v2 kernels remain unchanged.
 * Nonempty-row indexing follows the method used by PETSc compressed rows,
 * independently adapted to preserve empty-row y+=0 and per-call completion.
 * No external cost profile, numerical matrix copy, or cached Ax.
 */
#include "spmv_plan.h"
#include <errno.h>
#include <limits.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define V22_LARGE_EVENTS INT64_C(4194304)
#define V22_RESIDENT_BYTES INT64_C(2097152)
#define V22_RESIDENT_EVENTS INT64_C(262144)
#define V22_MERGE_EVENTS INT64_C(1048576)
static double clock_s(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int64_t cut(int64_t total,int t,int nt)
{return total/nt*t+(t<total%nt?t:total%nt);}
static int32_t lower_bound(const int32_t *a,int32_t n,int64_t row)
{
    int32_t lo=0,hi=n;
    while(lo<hi){int32_t mid=lo+(hi-lo)/2;if(a[mid]<row)lo=mid+1;else hi=mid;}
    return lo;
}
static int delegate(struct spmv_plan *p,const struct csr_matrix_int32 *a,
    const struct vector *x,const struct vector *y,const struct spmv_v2_options *c,
    int64_t calls,size_t budget,const char *route)
{
    int err=spmv_v2_prepare(p,a,x,y,c,budget);
    if(!err&&p->kernel==spmv_v2) {
        p->kernel=spmv_v22;p->v22.calls=calls;p->v22.route=route;
        p->v22.threads=p->v2.serial?1:omp_get_max_threads();
    }
    return err;
}
int spmv_v22_prepare(struct spmv_plan *p,const struct csr_matrix_int32 *a,
    const struct vector *x,const struct vector *y,const struct spmv_v2_options *c,
    const struct spmv_v22_options *o,int64_t calls,size_t budget)
{
    memset(p,0,sizeof(*p));p->matrix=a;p->kernel=spmv_v22;p->reason="guarded";
    if(calls<0||o->guarded<0||o->guarded>1||o->empty<0||o->empty>2||
       a->num_rows<0||a->num_columns<0||a->num_nonzeros<0||
       x->num_values!=a->num_columns||y->num_values!=a->num_rows)return EINVAL;
    if(a->num_nonzeros>INT64_MAX/2||a->num_nonzeros>INT64_MAX-a->num_rows)return EOVERFLOW;
    int64_t events=a->num_nonzeros+a->num_rows;
    if(!o->guarded)return delegate(p,a,x,y,c,calls,budget,"v2-disabled");
    if(c->kernel<0||c->kernel>v2_merge||c->partition<0||c->partition>v2_work||
       c->execution<0||c->execution>v2_parallel||c->serial_threshold<0||
       (c->prefetch!=0&&c->prefetch!=32&&c->prefetch!=128))return EINVAL;
    if(o->empty==2&&(c->kernel!=v2_auto||c->partition!=v2_part_auto||c->execution!=v2_exec_auto))return EINVAL;
    if(c->kernel!=v2_auto||c->partition!=v2_part_auto||c->execution!=v2_exec_auto||c->serial_threshold!=32768||c->prefetch)
        return delegate(p,a,x,y,c,calls,budget,"v2-explicit");
    if(a->value_format!=csr_value_f32||x->value_format!=vector_value_f32||y->value_format!=vector_value_f32)
        return delegate(p,a,x,y,c,calls,budget,"v2-types");
    struct spmv_v22_state *s=&p->v22;struct spmv_v2_state *v=&p->v2;
    s->calls=calls;s->threads=omp_get_max_threads();s->route="v2-rules-fast-validation";v->options=*c;
    if(!calls){s->threads=1;s->route="zero-calls";return 0;}
    if(events>=V22_LARGE_EVENTS)return delegate(p,a,x,y,c,calls,budget,"v2-large");
    if(!a->row_ptr||a->row_ptr[0]||a->row_ptr[a->num_rows]!=a->num_nonzeros||
       (a->num_nonzeros&&(!a->column_indices||!a->values))||
       (x->num_values&&!x->values)||(y->num_values&&!y->values))return EINVAL;
    long double bytes=8.0L*a->num_nonzeros+8.0L*(a->num_rows+1LL)+4.0L*((int64_t)a->num_rows+a->num_columns);
    int small=events<=32768||(events<=V22_RESIDENT_EVENTS&&bytes<=V22_RESIDENT_BYTES);
    int64_t capacity=a->num_nonzeros<a->num_rows?a->num_nonzeros:a->num_rows;
    int sampled=0,ns=a->num_rows<64?a->num_rows:64;
    for(int i=0;i<ns;i++) {
        int64_t row=ns>1?(int64_t)i*(a->num_rows-1)/(ns-1):0;
        sampled+=a->row_ptr[row]==a->row_ptr[row+1];
    }
    int want=o->empty==2||(o->empty==0&&!small&&calls>=2&&ns&&sampled*10>=ns*9);
    if(want&&capacity>0&&(size_t)capacity<=SIZE_MAX/sizeof(int32_t)&&
       (size_t)capacity*sizeof(int32_t)<=budget&&
       (size_t)s->threads<=(budget-(size_t)capacity*sizeof(int32_t))/sizeof(*s->bounds))
        s->rows=malloc((size_t)capacity*sizeof(int32_t));
    int bad=0,all8=1;int64_t longest=0,empty=0,count=0;double t=clock_s();
    for(int64_t r=0;r<a->num_rows;r++) {
        int64_t b=a->row_ptr[r],e=a->row_ptr[r+1];
        if(b<0||e<b||e>a->num_nonzeros){bad=1;continue;}
        int64_t len=e-b;all8&=len==8;empty+=len==0;if(len>longest)longest=len;
        if(len&&s->rows) {
            if(count>=capacity)bad=1;
            else s->rows[count++]=(int32_t)r;
        }
    }
    for(int64_t k=0;k<a->num_nonzeros;k++)if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns)bad=1;
    s->validation_s=clock_s()-t;
    if(bad){free(s->rows);s->rows=NULL;return EINVAL;}
    s->empty_rows=empty;v->all8=all8;v->longest=longest;
    v->kernel=all8?v2_short8:v2_csr;
    if(small){s->threads=1;s->route="small-serial";}
    if(!a->num_nonzeros) {
        s->empty_runs=1;s->route="zero-runs";
        if((int64_t)a->num_rows*4<=V22_RESIDENT_BYTES)s->threads=1;
    } else if(s->rows&&(o->empty==2||empty*10>=(int64_t)a->num_rows*9)&&longest<=64) {
        s->bounds=malloc((size_t)s->threads*sizeof(*s->bounds));
        if(s->bounds) {
            s->count=(int32_t)count;s->empty_runs=1;s->route="empty-runs";
            for(int i=0;i<s->threads;i++) {
                struct v22_bounds *z=&s->bounds[i];z->begin=cut(a->num_rows,i,s->threads);z->end=cut(a->num_rows,i+1,s->threads);
                z->first=lower_bound(s->rows,s->count,z->begin);z->last=lower_bound(s->rows,s->count,z->end);
            }
            p->extra_bytes=(size_t)capacity*sizeof(int32_t)+(size_t)s->threads*sizeof(*s->bounds);
        }
    }
    if(!s->empty_runs){free(s->rows);s->rows=NULL;free(s->bounds);s->bounds=NULL;}
    v->serial=s->threads==1;v->partition=v2_rows;
    int64_t max_work=0;
    for(int i=0;i<s->threads;i++) {
        int64_t b=cut(a->num_rows,i,s->threads),e=cut(a->num_rows,i+1,s->threads);
        int64_t w=a->row_ptr[e]-a->row_ptr[b]+e-b;if(w>max_work)max_work=w;
    }
    double ideal=(double)events/s->threads;
    if(max_work>1.5*ideal)v->partition=v2_work;
    if(!s->empty_runs&&!c->strict_order&&s->threads>1&&longest>2*ideal) {
        v->kernel=v2_merge;
        if(calls>=2&&events<=V22_MERGE_EVENTS&&longest>=a->num_nonzeros/4) {
            s->threads=s->threads<2?s->threads:2;s->route="few-giants";
        }
        v->capacity=s->threads;
        if((size_t)s->threads<=budget/sizeof(*v->scratch))v->scratch=malloc((size_t)s->threads*sizeof(*v->scratch));
        if(!v->scratch){v->kernel=v2_csr;p->reason="scratch-fallback";}
        else p->extra_bytes=(size_t)s->threads*sizeof(*v->scratch);
    }
    v->serial=s->threads==1;return 0;
}
static void zeros(float *y,int64_t b,int64_t e)
{for(int64_t r=b;r<e;r++)y[r]+=0.0f;}
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize,no-tree-slp-vectorize")))
#endif
static float sum_row(const float *v,const int32_t *j,const float *x,int64_t b,int64_t e)
{float sum=0;for(int64_t k=b;k<e;k++)sum+=v[k]*x[j[k]];return sum;}
int spmv_v22_execute(const struct spmv_plan *p,const struct vector *x,struct vector *y,int64_t *flops)
{
    const struct csr_matrix_int32 *a=p->matrix;const struct spmv_v22_state *s=&p->v22;
    if(!s->calls||x->value_format!=vector_value_f32||y->value_format!=vector_value_f32)return EINVAL;
    if(*flops>INT64_MAX-2*a->num_nonzeros)return EOVERFLOW;
    int tid=omp_get_thread_num(),nt=omp_get_num_threads();
    if(!s->empty_runs||nt!=s->threads)spmv_v2_execute(p,x->values,y->values);
    else {
        int64_t b=cut(a->num_rows,tid,nt),e=cut(a->num_rows,tid+1,nt);int first=0,last=0;
        if(s->bounds){b=s->bounds[tid].begin;e=s->bounds[tid].end;first=s->bounds[tid].first;last=s->bounds[tid].last;}
        const float *v=a->values;const int32_t *j=a->column_indices;const float *xx=x->values;float *yy=y->values;
        for(int i=first;i<last;i++) {
            int64_t row=s->rows[i];zeros(yy,b,row);
            yy[row]+=sum_row(v,j,xx,a->row_ptr[row],a->row_ptr[row+1]);b=row+1;
        }
        zeros(yy,b,e);
        if(omp_in_parallel()) {
#pragma omp barrier
        }
    }
    *flops+=2*a->num_nonzeros;return 0;
}
void spmv_v22_report(const struct spmv_plan *p,FILE *f)
{
    const struct spmv_v22_state *s=&p->v22;
    fprintf(f,"spmv_v22: kernel=%s partition=%s threads=%d calls=%ld route=%s empty_rows=%ld nonempty_rows=%d validation_s=%.9f\n",
        s->empty_runs?"empty-runs":spmv_v2_name(p->v2.kernel),p->v2.partition==v2_work?"work":"rows",s->threads,s->calls,s->route,s->empty_rows,s->count,s->validation_s);
}
