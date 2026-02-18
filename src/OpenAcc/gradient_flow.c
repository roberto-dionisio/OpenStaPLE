#include "../Include/common_defines.h"
#include "./geometry.h"
#include "./plaquettes.h"
#include "./stouting.h"
#include "./su3_utilities.h"
#include "./gradient_flow.h"

#ifdef MULTIDEVICE
#include "../Mpi/communications.h"
#endif

static void su3_soa_copy(__restrict su3_soa *dst, __restrict const su3_soa *src)
{
    int d0, d1, d2, d3;
    #pragma acc parallel loop collapse(4) present(dst, src)
    for (d3 = D3_HALO; d3 < nd3 - D3_HALO; d3++) {
        for (d2 = 0; d2 < nd2; d2++) {
            for (d1 = 0; d1 < nd1; d1++) {
                for (d0 = 0; d0 < nd0; d0++) {
                    const int idxh = snum_acc(d0, d1, d2, d3);
                    const int parity = (d0 + d1 + d2 + d3) & 1;
                    #pragma acc loop seq
                    for (int mu = 0; mu < 4; mu++) {
                        const int dir = 2 * mu + parity;
                        dst[dir].r0.c0[idxh] = src[dir].r0.c0[idxh];
                        dst[dir].r0.c1[idxh] = src[dir].r0.c1[idxh];
                        dst[dir].r0.c2[idxh] = src[dir].r0.c2[idxh];

                        dst[dir].r1.c0[idxh] = src[dir].r1.c0[idxh];
                        dst[dir].r1.c1[idxh] = src[dir].r1.c1[idxh];
                        dst[dir].r1.c2[idxh] = src[dir].r1.c2[idxh];
                    }
                }
            }
        }
    }
}

static double su3_soa_max_dist(__restrict const su3_soa *A,
                               __restrict const su3_soa *B)
{
    double maxd = 0.0;
    int d0, d1, d2, d3;

    // distance per link computed on the stored rc00, rc11, c01, c02, c12
    #pragma acc parallel loop collapse(4) reduction(max:maxd) present(A, B)
    for (d3 = D3_HALO; d3 < nd3 - D3_HALO; d3++) {
        for (d2 = 0; d2 < nd2; d2++) {
            for (d1 = 0; d1 < nd1; d1++) {
                for (d0 = 0; d0 < nd0; d0++) {
                    const int idxh = snum_acc(d0, d1, d2, d3);
                    const int parity = (d0 + d1 + d2 + d3) & 1;

                    #pragma acc loop seq
                    for (int mu = 0; mu < 4; mu++) {
                        const int dir = 2 * mu + parity;

                        const d_complex d00 = A[dir].r0.c0[idxh] - B[dir].r0.c0[idxh];
                        const d_complex d01 = A[dir].r0.c1[idxh] - B[dir].r0.c1[idxh];
                        const d_complex d02 = A[dir].r0.c2[idxh] - B[dir].r0.c2[idxh];

                        const d_complex d10 = A[dir].r1.c0[idxh] - B[dir].r1.c0[idxh];
                        const d_complex d11 = A[dir].r1.c1[idxh] - B[dir].r1.c1[idxh];
                        const d_complex d12 = A[dir].r1.c2[idxh] - B[dir].r1.c2[idxh];

                        double dist2 = 0.0;
                        dist2 += creal(d00) * creal(d00) + cimag(d00) * cimag(d00);
                        dist2 += creal(d01) * creal(d01) + cimag(d01) * cimag(d01);
                        dist2 += creal(d02) * creal(d02) + cimag(d02) * cimag(d02);
                        dist2 += creal(d10) * creal(d10) + cimag(d10) * cimag(d10);
                        dist2 += creal(d11) * creal(d11) + cimag(d11) * cimag(d11);
                        dist2 += creal(d12) * creal(d12) + cimag(d12) * cimag(d12);

                        const double dist = sqrt(dist2);
                        if (dist > maxd) maxd = dist;
                    }
                }
            }
        }
    }

    // norm by Nc**3
    return maxd / 9.0;
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

    tamat_scale_inplace(Zout, +dt);  
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

static double gradflow_wilson_RKstep_adaptive_aux(__restrict const su3_soa *W0,
                                                  __restrict gradflow_workspace *ws,
                                                  double dt)
{
    // Z0, W1 = exp(-1/4 Z0) W0
    gradflow_compute_Z_wilson(W0, ws->staples, ws->Z0, dt);
    tamat_lincomb2(ws->Zcomb, ws->Z0, 0.25, ws->Z0, 0.0);
    exp_minus_QA_times_conf(W0, ws->Zcomb, ws->W1, ws->exp_aux);

    // Z1
    gradflow_compute_Z_wilson(ws->W1, ws->staples, ws->Z1, dt);

    // W2' = exp(-(2 Z1 - Z0)) W0   (second order estimate)
    tamat_lincomb2(ws->Zcomb, ws->Z1, 2.0, ws->Z0, -1.0);
    exp_minus_QA_times_conf(W0, ws->Zcomb, ws->W2prime, ws->exp_aux);

    // W2 = exp(-(8/9 Z1 - 17/36 Z0)) W1
    tamat_lincomb2(ws->Zcomb, ws->Z1, (8.0 / 9.0), ws->Z0, (-17.0 / 36.0));
    exp_minus_QA_times_conf(ws->W1, ws->Zcomb, ws->W2, ws->exp_aux);

    // Z2 and W3 (third order result) into ws->W1 reusing the buffer 
    gradflow_compute_Z_wilson(ws->W2, ws->staples, ws->Z2, dt);
    tamat_lincomb2(ws->Zcomb, ws->Z2, (3.0 / 4.0), ws->Zcomb, -1.0);
    exp_minus_QA_times_conf(ws->W2, ws->Zcomb, ws->W1, ws->exp_aux);

    // error estimate
    const double max_dist = su3_soa_max_dist(ws->W1, ws->W2prime);

    // unitarizationn of the accepted sol
    unitarize_conf(ws->W1);

    return max_dist;
}

double gradflow_wilson_RKstep_adaptive(__restrict su3_soa *V,
                                       __restrict gradflow_workspace *ws,
                                       double *t,
                                       double *dt,
                                       double delta,
                                       double dt_max,
                                       int *accepted)
{
    //debug
    static long dbg_attempt = 0;

    const double t_before = *t;
    const double dt_try   = *dt;

    const double max_dist = gradflow_wilson_RKstep_adaptive_aux(V, ws, dt_try);
    //edebug
    if (max_dist < delta) {
        *accepted = 1;
        *t += *dt;
        su3_soa_copy(V, ws->W1);
    } else {
        *accepted = 0;
    }

    // adapt dt (error ~ dt**3)
    const double eps = 1.0e-30;
    const double denom = (max_dist > eps) ? max_dist : eps;
    double new_dt = (*dt) * 0.95 * pow(delta / denom, 1.0 / 3.0);

    if (new_dt > dt_max) new_dt = dt_max;
    if (new_dt < eps) new_dt = eps;

    dbg_attempt++;
    printf("AGF dbg #%ld: acc=%d  t:%.18lf -> %.18lf  dt_try=%.6e  err=%.4e  dt_new=%.6e\n",
           dbg_attempt, *accepted, t_before, *t, dt_try, max_dist, new_dt);
    *dt = new_dt;

    return max_dist;
}

void gradflow_perform_measures_localobs_adaptive(__restrict su3_soa *V,
                                                 __restrict gradflow_workspace *ws,
                                                 const gradflow_adaptive_meas_params *p,
                                                 gradflow_measure_cb cb,
                                                 void *user_data,
                                                 __restrict su3_soa *V_backup_or_null)
{
    double t = 0.0;
    double dt = p->dt0;
    int meas_count = 0;

    if (V_backup_or_null) su3_soa_copy(V_backup_or_null, V);

    while (meas_count < p->num_meas) {
        int accepted = 0;
        gradflow_wilson_RKstep_adaptive(V, ws, &t, &dt, p->delta, p->dt_max, &accepted);

        if (accepted == 1) {
            const double target = p->meas_each * (double)(meas_count + 1);
            if (fabs(t - target) <= p->time_bin) {
                if (cb) cb(meas_count, t, user_data);
                meas_count++;
            }
        }

        // adapt dt to not skip next measure time and having homegenous measurement bins
        {
            const double next_target = p->meas_each * (double)(meas_count + 1);
            if ((t + dt - next_target) > p->time_bin) {
                const double forced = next_target - t;
                if (forced > 0.0) dt = forced;
            }
        }
    }

    if (V_backup_or_null) su3_soa_copy(V, V_backup_or_null);
}
