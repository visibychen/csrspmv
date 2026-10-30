/* SPDX-License-Identifier: GPL-3.0-or-later
 * Prepared on top of v2.2. Original, v2 and v2.2 numerical kernels unchanged.
 * Sampling is an allocation hint, never the final eligibility decision.
 */
#include "spmv_plan.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define LARGE_EVENTS INT64_C(4194304)
#define RESIDENT_BYTES INT64_C(2097152)
#define RESIDENT_EVENTS INT64_C(262144)
#define MERGE_EVENTS INT64_C(1048576)
#define PARALLEL_COLUMNS INT64_C(1048576)
static double seconds(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static int32_t lower(const int32_t *rows,int32_t count,int64_t row)
{
    int32_t lo=0,hi=count;
    while(lo<hi){int32_t mid=lo+(hi-lo)/2;if(rows[mid]<row)lo=mid+1;else hi=mid;}
    return lo;
}
static int fits(int64_t count,int nt,size_t budget)
{
    if(count<0||nt<1||(uint64_t)count>SIZE_MAX/sizeof(int32_t))return 0;
    size_t bytes=(size_t)count*sizeof(int32_t);
    return bytes<=budget&&(size_t)nt<=(budget-bytes)/sizeof(struct v22_bounds);
}
static int delegate(struct spmv_plan *p,const struct csr_matrix_int32 *a,
    const struct vector *x,const struct vector *y,const struct spmv_v2_options *c,
    size_t budget,const char *route)
{
    int err=spmv_v2_prepare(p,a,x,y,c,budget);
    if(!err){p->v221.threads=p->kernel==spmv_v2&&p->v2.serial?1:omp_get_max_threads();
        p->v221.validation_threads=-1;p->v221.route=route;}
    return err;
}
int spmv_v221_prepare(struct spmv_plan *p,const struct csr_matrix_int32 *a,
    const struct vector *x,const struct vector *y,const struct spmv_v2_options *c,
    const struct spmv_v221_options *o,int64_t calls,size_t budget)
{
    if(!p)return EINVAL;
    memset(p,0,sizeof(*p));p->matrix=a;p->kernel=spmv_v2;p->reason="guarded-v221";
    if(!a||!x||!y||!c||!o)return EINVAL;
    if(calls<0||o->guarded<0||o->guarded>1||o->empty<0||o->empty>2||
       o->parallel_validation<0||o->parallel_validation>1||o->balance_work<0||o->balance_work>1||
       a->num_rows<0||a->num_columns<0||a->num_nonzeros<0||
       x->num_values!=a->num_columns||y->num_values!=a->num_rows)return EINVAL;
    if(a->num_nonzeros>INT64_MAX/2||a->num_nonzeros>INT64_MAX-a->num_rows)return EOVERFLOW;
    if(!o->guarded)return delegate(p,a,x,y,c,budget,"v2-disabled-direct");
    if(c->kernel<0||c->kernel>v2_merge||c->partition<0||c->partition>v2_work||
       c->execution<0||c->execution>v2_parallel||c->serial_threshold<0||
       (c->prefetch!=0&&c->prefetch!=32&&c->prefetch!=128))return EINVAL;
    if(o->empty==2&&(c->kernel!=v2_auto||c->partition!=v2_part_auto||c->execution!=v2_exec_auto))return EINVAL;
    if(c->kernel!=v2_auto||c->partition!=v2_part_auto||c->execution!=v2_exec_auto||c->serial_threshold!=32768||c->prefetch)
        return delegate(p,a,x,y,c,budget,"v2-explicit-direct");
    if(a->value_format!=csr_value_f32||x->value_format!=vector_value_f32||y->value_format!=vector_value_f32)
        return delegate(p,a,x,y,c,budget,"v2-types-direct");
    int64_t events=a->num_rows+a->num_nonzeros;
    struct spmv_v221_state *q=&p->v221;struct spmv_v22_state *s=&p->v22;struct spmv_v2_state *v=&p->v2;
    q->threads=s->threads=omp_get_max_threads();q->validation_threads=1;q->route="v2-local-direct";
    s->calls=calls;v->options=*c;
    if(!calls){p->kernel=spmv_v22;q->threads=s->threads=1;q->route="zero-calls";return 0;}
    if(events>=LARGE_EVENTS)return delegate(p,a,x,y,c,budget,"v2-large-direct");
    if(!a->row_ptr||a->row_ptr[0]||a->row_ptr[a->num_rows]!=a->num_nonzeros||
       (a->num_nonzeros&&(!a->column_indices||!a->values))||
       (x->num_values&&!x->values)||(y->num_values&&!y->values))return EINVAL;
    long double bytes=8.0L*a->num_nonzeros+8.0L*(a->num_rows+1LL)+4.0L*((int64_t)a->num_rows+a->num_columns);
    int small=events<=32768||(events<=RESIDENT_EVENTS&&bytes<=RESIDENT_BYTES);
    int64_t capacity=a->num_nonzeros<a->num_rows?a->num_nonzeros:a->num_rows;
    int sampled=0,ns=a->num_rows<64?a->num_rows:64;
    for(int i=0;i<ns;i++){
        int64_t row=ns>1?(int64_t)i*(a->num_rows-1)/(ns-1):0;
        sampled+=a->row_ptr[row]==a->row_ptr[row+1];
    }
    int hint=o->empty==2||(o->empty==0&&!small&&calls>=2&&ns&&sampled*10>=ns*9);
    if(hint&&capacity>0&&fits(capacity,s->threads,budget))s->rows=malloc((size_t)capacity*sizeof(*s->rows));
    int bad=0,all8=1;int64_t longest=0,empty=0,count=0;double t=seconds();
    for(int64_t row=0;row<a->num_rows;row++){
        int64_t begin=a->row_ptr[row],end=a->row_ptr[row+1];
        if(begin<0||end<begin||end>a->num_nonzeros){bad=1;continue;}
        int64_t len=end-begin;all8&=len==8;empty+=len==0;if(len>longest)longest=len;
        if(len){
            if(s->rows){if(count>=capacity)bad=1;else s->rows[count]=(int32_t)row;}
            count++;
        }
    }
    q->row_s=seconds()-t;
    if(bad){free(s->rows);s->rows=NULL;return EINVAL;}
    t=seconds();
    if(!small&&o->parallel_validation&&a->num_nonzeros>=PARALLEL_COLUMNS&&q->threads>1){
#pragma omp parallel reduction(|:bad)
        {
            if(!omp_get_thread_num())q->validation_threads=omp_get_num_threads();
#pragma omp for schedule(static)
            for(int64_t k=0;k<a->num_nonzeros;k++)
                if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns)bad=1;
        }
    } else {
        for(int64_t k=0;k<a->num_nonzeros;k++)
            if(a->column_indices[k]<0||a->column_indices[k]>=a->num_columns)bad=1;
    }
    q->column_s=seconds()-t;s->validation_s=q->row_s+q->column_s;
    if(bad){free(s->rows);s->rows=NULL;return EINVAL;}
    q->statistics_known=1;q->empty_rows=s->empty_rows=empty;q->nonempty_rows=count;
    v->all8=all8;v->longest=longest;v->kernel=all8?v2_short8:v2_csr;
    if(small){s->threads=1;q->route="resident-direct";}
    if(!a->num_nonzeros){
        s->empty_runs=1;q->route="zero-runs";
        if((int64_t)a->num_rows*4<=RESIDENT_BYTES)s->threads=1;
    }
    v->partition=v2_rows;int64_t max_work=0;
    for(int i=0;i<s->threads;i++){
        int64_t begin=spmv_v2_row_cut(a,i,s->threads,0),end=spmv_v2_row_cut(a,i+1,s->threads,0);
        int64_t work=a->row_ptr[end]-a->row_ptr[begin]+end-begin;if(work>max_work)max_work=work;
    }
    double ideal=(double)events/s->threads;
    if(max_work>1.5*ideal)v->partition=v2_work;
    int eligible=a->num_nonzeros&&longest<=64&&(o->empty==2||
        (o->empty==0&&!small&&calls>=2&&empty*10>=(int64_t)a->num_rows*9));
    if(eligible&&!s->rows&&fits(count,s->threads,budget)){
        q->index_retry=1;t=seconds();capacity=count;
        s->rows=malloc((size_t)capacity*sizeof(*s->rows));
        if(s->rows){
            int64_t j=0;
            for(int64_t row=0;row<a->num_rows;row++)if(a->row_ptr[row]!=a->row_ptr[row+1])s->rows[j++]=(int32_t)row;
        }
        q->retry_s=seconds()-t;
    }
    if(eligible&&s->rows){
        s->bounds=malloc((size_t)s->threads*sizeof(*s->bounds));
        if(s->bounds){
            s->count=(int32_t)count;s->empty_runs=1;q->route="empty-runs";q->index_capacity=capacity;
            int work=o->balance_work&&v->partition==v2_work;
            for(int i=0;i<s->threads;i++){
                struct v22_bounds *z=&s->bounds[i];
                z->begin=spmv_v2_row_cut(a,i,s->threads,work);z->end=spmv_v2_row_cut(a,i+1,s->threads,work);
                z->first=lower(s->rows,s->count,z->begin);z->last=lower(s->rows,s->count,z->end);
            }
            p->extra_bytes=(size_t)capacity*sizeof(*s->rows)+(size_t)s->threads*sizeof(*s->bounds);
        }
    }
    if(!s->empty_runs){free(s->rows);s->rows=NULL;free(s->bounds);s->bounds=NULL;}
    if(!s->empty_runs&&!c->strict_order&&s->threads>1&&longest>2*ideal){
        v->kernel=v2_merge;
        if(calls>=2&&events<=MERGE_EVENTS&&longest>=a->num_nonzeros/4){
            s->threads=s->threads<2?s->threads:2;q->route="few-giants-direct";
        }
        v->capacity=s->threads;
        if((size_t)s->threads<=budget/sizeof(*v->scratch))v->scratch=malloc((size_t)s->threads*sizeof(*v->scratch));
        if(!v->scratch){v->kernel=v2_csr;p->reason="scratch-fallback";}
        else p->extra_bytes=(size_t)s->threads*sizeof(*v->scratch);
    }
    v->serial=s->threads==1;q->threads=s->threads;
    p->kernel=s->empty_runs?spmv_v22:spmv_v2;
    return 0;
}
void spmv_v221_report(const struct spmv_plan *p,FILE *f)
{
    const struct spmv_v221_state *q=&p->v221;
    fprintf(f,"spmv_v221: kernel=%s executor=%s partition=%s threads=%d route=%s statistics_known=%d empty_rows=%"PRId64" nonempty_rows=%"PRId64" index_capacity=%"PRId64" index_retry=%d validation_threads=%d row_scan_s=%.9f column_validation_s=%.9f retry_s=%.9f\n",
        p->kernel==spmv_original?"original":p->v22.empty_runs?"empty-runs":spmv_v2_name(p->v2.kernel),spmv_kernel_name(p->kernel),
        p->v2.partition==v2_work?"work":"rows",q->threads,q->route,q->statistics_known,
        q->statistics_known?q->empty_rows:INT64_C(-1),q->statistics_known?q->nonempty_rows:INT64_C(-1),
        q->index_capacity,q->index_retry,q->validation_threads,q->row_s,q->column_s,q->retry_s);
}
