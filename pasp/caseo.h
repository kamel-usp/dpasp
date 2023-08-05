#ifndef _PASP_CASEO
#define _PASP_CASEO

#include <stdbool.h>
#include "cprogram.h"
#include "cinf.h"
#include "cdata.h"
#include "cmodels.h"

/**
 * Answer Set Enumeration by Optimality (ASEO).
 */

models_t* aseo(program_t *P, size_t k, psemantics_t psem, observations_t *O, int scale,
    bool (*f)(const clingo_model_t*, program_t*, models_t*, size_t, observations_t*,
      clingo_control_t*));

#endif
