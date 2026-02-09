#include "../Include/common_defines.h"
#include "./geometry.h"
#include "./plaquettes.h"
#include "./stouting.h"
#include "./su3_utilities.h"
#include "./gradient_flow.h"

#ifdef MULTIDEVICE
#include "../Mpi/communications.h"
#endif

static void tamat_scale_inplace(__restrict tamat_soa *A, double alpha)
{
    int d0, d1, d2, d3;
    #pragma acc kernels present(A)
    #pragma acc loop independent gang
    for (d3 = D3_HALO; d3 < nd3 - D3_HALO; d3++) {
        for (d2 = 0; d2 < nd2; d2++) {
            for (d1 = 0; d1 < nd1; d1++) {
                for (d0 = 0; d0 < nd0; d0++) {
                    const int idxh = snum_acc(d0, d1, d2, d3);
                    const int parity = (d0 + d1 + d2 + d3) & 1;
                    #pragma acc loop seq
                    for (int mu = 0; mu < 4; mu++) {
                        const int dir = 2 * mu + parity;
                        A[dir].ic00[idxh] *= alpha;
                        A[dir].ic11[idxh] *= alpha;
                        A[dir].c01[idxh]  *= alpha;
                        A[dir].c02[idxh]  *= alpha;
                        A[dir].c12[idxh]  *= alpha;
                    }
                }
            }
        }
    }
}

static void tamat_lincomb2(__restrict tamat_soa *OUT,
                           const __restrict tamat_soa *A, double a,
                           const __restrict tamat_soa *B, double b)
{
    int d0, d1, d2, d3;
    #pragma acc parallel loop independent collapse(4) present(OUT, A, B)
    for (d3 = D3_HALO; d3 < nd3 - D3_HALO; d3++) {
        for (d2 = 0; d2 < nd2; d2++) {
            for (d1 = 0; d1 < nd1; d1++) {
                for (d0 = 0; d0 < nd0; d0++) {
                    const int idxh = snum_acc(d0, d1, d2, d3);
                    const int parity = (d0 + d1 + d2 + d3) & 1; 
                    
                    #pragma acc loop seq
                    for (int mu = 0; mu < 4; mu++) {
                        const int dir = 2 * mu + parity; // check
                        
                        // Loop unrolling i
                        OUT[dir].ic00[idxh] = a * A[dir].ic00[idxh] + b * B[dir].ic00[idxh];
                        OUT[dir].ic11[idxh] = a * A[dir].ic11[idxh] + b * B[dir].ic11[idxh];
                        OUT[dir].c01[idxh]  = a * A[dir].c01[idxh]  + b * B[dir].c01[idxh];
                        OUT[dir].c02[idxh]  = a * A[dir].c02[idxh]  + b * B[dir].c02[idxh];
                        OUT[dir].c12[idxh]  = a * A[dir].c12[idxh]  + b * B[dir].c12[idxh];
                    }
                }
            }
        }
    }
}

// OUT = (-dt) * force_Wilson(V), where force is the TA-projected derivative
static void gradflow_compute_Z_wilson(__restrict const su3_soa *V,
                                                                         __restrict su3_soa *staples,
                                                                         __restrict tamat_soa *Zout,
                                                                         double dt)
{
#if NRANKS_D3 > 1
    communicate_su3_borders((su3_soa*)V, GAUGE_HALO);
#endif

    set_su3_soa_to_zero(staples);
    calc_loc_staples_nnptrick_all(V, staples);
    conf_times_staples_ta_part(V, staples, Zout);

    tamat_scale_inplace(Zout, +dt); // check sign 
}

void gradflow_wilson_RKstep(__restrict su3_soa *V,
                                                        __restrict gradflow_workspace *ws,
                                                        double dt)
{
    //Z0 = -dt * grad S(W0), W1 = exp(1/4 Z0) W0
    gradflow_compute_Z_wilson(V, ws->staples, ws->Z0, dt);
    tamat_lincomb2(ws->Zcomb, ws->Z0, 0.25, ws->Z0, 0.0);
    exp_minus_QA_times_conf(V, ws->Zcomb, ws->W1, ws->exp_aux);

    //Z1 = -dt * grad S(W1), W2 = exp( 8/9 Z1 - 17/36 Z0 ) W1
    gradflow_compute_Z_wilson(ws->W1, ws->staples, ws->Z1, dt);
    tamat_lincomb2(ws->Zcomb, ws->Z1, (8.0 / 9.0), ws->Z0, (-17.0 / 36.0));
    exp_minus_QA_times_conf(ws->W1, ws->Zcomb, ws->W2, ws->exp_aux);

    //Z2 = -dt * grad S(W2), Vnew = exp( 3/4 Z2 - (8/9 Z1 - 17/36 Z0) ) W2
    gradflow_compute_Z_wilson(ws->W2, ws->staples, ws->Z2, dt);
    tamat_lincomb2(ws->Zcomb, ws->Z2, (3.0 / 4.0), ws->Zcomb, -1.0);
    exp_minus_QA_times_conf(ws->W2, ws->Zcomb, V, ws->exp_aux);

    // safety
    // unitarize_conf(V);
}