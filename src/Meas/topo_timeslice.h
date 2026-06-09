#ifndef TOPO_TIMESLICE_H
#define TOPO_TIMESLICE_H

#include "../OpenAcc/struct_c_def.h"
#include "../OpenAcc/geometry.h"

// Compute Q(t, k), Q_re , Q_im for k = 0, 1, ... Ns/2

void compute_topo_timeslice_momenta(
    const double_soa *loc_q,
    int smap,
    int Nt, int Ns,
    double *Q_re,
    double *Q_im);

#endif
