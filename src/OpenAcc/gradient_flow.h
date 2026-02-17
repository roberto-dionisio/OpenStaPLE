#ifndef GRADIENT_FLOW_H
#define GRADIENT_FLOW_H

#include "./struct_c_def.h"


typedef struct gradflow_workspace_t {
    su3_soa   *W1;       // [8]
    su3_soa   *W2;       // [8]
    su3_soa   *W2prime;  // [8] 2nd order estimate for error
    su3_soa   *staples;  // [8]
    su3_soa   *exp_aux;  // [8]  (temporary SU(3) exponentials)

    tamat_soa *Z0;       // [8]
    tamat_soa *Z1;       // [8]
    tamat_soa *Z2;       // [8]
    tamat_soa *Zcomb;    // [8]  (temporary combination)
} gradflow_workspace;

typedef void (*gradflow_measure_cb)(int meas_idx, double t, void *user_data);

typedef struct gradflow_adaptive_meas_params_t {
    double dt0;
    double delta;
    double dt_max;
    int    num_meas;
    double meas_each;
    double time_bin;
} gradflow_adaptive_meas_params;

void gradflow_wilson_RKstep(__restrict su3_soa *V,
                                                        __restrict gradflow_workspace *ws,
                                                        double dt);
// Adaptive wilson flow step (a la Fritzsch-Ramos)
double gradflow_wilson_RKstep_adaptive(__restrict su3_soa *V,
                                       __restrict gradflow_workspace *ws,
                                       double *t,
                                       double *dt,
                                       double delta,
                                       double dt_max,
                                       int *accepted);

void gradflow_perform_measures_localobs_adaptive(__restrict su3_soa *V,
                                                 __restrict gradflow_workspace *ws,
                                                 const gradflow_adaptive_meas_params *p,
                                                 gradflow_measure_cb cb,
                                                 void *user_data,
                                                 __restrict su3_soa *V_backup_or_null);

#endif