#ifndef GRADIENT_FLOW_H
#define GRADIENT_FLOW_H

#include "./struct_c_def.h"


typedef struct gradflow_workspace_t {
    su3_soa   *W1;       // [8]
    su3_soa   *W2;       // [8]
    su3_soa   *staples;  // [8]
    su3_soa   *exp_aux;  // [8]  (temporary SU(3) exponentials)

    tamat_soa *Z0;       // [8]
    tamat_soa *Z1;       // [8]
    tamat_soa *Z2;       // [8]
    tamat_soa *Zcomb;    // [8]  (temporary combination)
} gradflow_workspace;


void gradflow_wilson_RKstep(__restrict su3_soa *V,
                                                        __restrict gradflow_workspace *ws,
                                                        double dt);

#endif