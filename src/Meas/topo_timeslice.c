#ifndef TOPO_TIMESLICE_C
#define TOPO_TIMESLICE_C

#include "topo_timeslice.h"
#include "../Mpi/multidev.h"
#include "../Mpi/geometry_multidev.h"

#ifdef __GNUC__
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

#ifdef MULTIDEVICE
#include <mpi.h>
#endif

void compute_topo_timeslice_momenta(
    const double_soa *loc_q,
    int smap,
    int Nt, int Ns,
    double *Q_re,
    double *Q_im)
{
    const int tmap  = geom_par.tmap;
    const int k_max = Ns / 2;
    const int nk    = k_max + 1;
    const int tot   = Nt * nk;

    for(int i = 0; i < tot; i++) { Q_re[i] = 0.0; Q_im[i] = 0.0; }

    /*general parallel loop that matches other routines */
    vec4int origin = gl_loc_origin_from_rank(devinfo.myrank);

    for(int d3 = D3_HALO; d3 < LNH_N3 - D3_HALO; d3++) {
        for(int d2 = D2_HALO; d2 < LNH_N2 - D2_HALO; d2++) {
            for(int d1 = D1_HALO; d1 < LNH_N1 - D1_HALO; d1++) {
                for(int d0 = D0_HALO; d0 < LNH_N0 - D0_HALO; d0++) {
                    const int parity = (d0 + d1 + d2 + d3) % 2;
                    const int idxh   = snum_acc(d0, d1, d2, d3);
                    const double q_val = loc_q[parity].d[idxh];

                    /* Recover global 4-coordinates */
                    int gl_d[4];
                    gl_d[0] = origin.d0 + (d0 - D0_HALO);
                    gl_d[1] = origin.d1 + (d1 - D1_HALO);
                    gl_d[2] = origin.d2 + (d2 - D2_HALO);
                    gl_d[3] = origin.d3 + (d3 - D3_HALO);

                    const int t_coord = gl_d[tmap];
                    const int s_coord = gl_d[smap];

                    const double two_pi_over_Ns = 2.0 * M_PI / (double)Ns;

                    for(int k = 0; k <= k_max; k++) {
                        const double phase = two_pi_over_Ns * k * s_coord;
                        Q_re[t_coord * nk + k] += q_val * cos(phase);
                        Q_im[t_coord * nk + k] += q_val * sin(phase);
                    }
                }
            }
        }
    }

#ifdef MULTIDEVICE
    MPI_Allreduce(MPI_IN_PLACE, Q_re, tot, MPI_DOUBLE, MPI_SUM, devinfo.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, Q_im, tot, MPI_DOUBLE, MPI_SUM, devinfo.mpi_comm);
#endif
}

#endif
