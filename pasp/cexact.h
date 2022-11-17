#ifndef _PASP_CEXACT
#define _PASP_CEXACT

#include "cprogram.h"
#include "cinf.h"

#include "../thpool/thpool.h"

bool dispatch_job(total_choice_t *theta, pthread_mutex_t *wakeup,
    bool *busy_procs, storage_t *S, size_t num_procs, threadpool pool, pthread_cond_t *avail,
    void (*compute_func)(void*), bool has_ad, size_t j, size_t c);

void compute_total_choice(void *data);
void compute_total_choice_maxent(void *data);

bool exact_enum(program_t *P, double (*R)[2], bool lstable_sat, psemantics_t psem);

#endif
