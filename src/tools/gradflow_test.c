#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../Include/setting_file_parser.h"
#include "../OpenAcc/alloc_vars.h"
#include "../OpenAcc/geometry.h"
#include "../OpenAcc/io.h"
#include "../OpenAcc/deviceinit.h"
#include "../OpenAcc/gradient_flow.h"
#include "../OpenAcc/su3_measurements.h"

#ifndef __GNUC__
#include "openacc.h"
#endif

#ifdef MULTIDEVICE
#include <mpi.h>
#include "../Mpi/multidev.h"
#endif

int conf_id_iter;
int verbosity_lv;

#define ALIGN 128

static void alloc_ws(gradflow_workspace *ws)
{
    if (posix_memalign((void **)&ws->W1, ALIGN, 8 * sizeof(su3_soa))) exit(1);
    if (posix_memalign((void **)&ws->W2, ALIGN, 8 * sizeof(su3_soa))) exit(1);
    if (posix_memalign((void **)&ws->staples, ALIGN, 8 * sizeof(su3_soa))) exit(1);
    if (posix_memalign((void **)&ws->exp_aux, ALIGN, 8 * sizeof(su3_soa))) exit(1);
    if (posix_memalign((void **)&ws->W2prime, ALIGN, 8 * sizeof(su3_soa))) exit(1);

    if (posix_memalign((void **)&ws->Z0, ALIGN, 8 * sizeof(tamat_soa))) exit(1);
    if (posix_memalign((void **)&ws->Z1, ALIGN, 8 * sizeof(tamat_soa))) exit(1);
    if (posix_memalign((void **)&ws->Z2, ALIGN, 8 * sizeof(tamat_soa))) exit(1);
    if (posix_memalign((void **)&ws->Zcomb, ALIGN, 8 * sizeof(tamat_soa))) exit(1);

    
    #pragma acc enter data create(ws->W1[0:8], ws->W2[0:8], ws->W2prime[0:8], ws->staples[0:8], ws->exp_aux[0:8])
    #pragma acc enter data create(ws->Z0[0:8], ws->Z1[0:8], ws->Z2[0:8], ws->Zcomb[0:8])
}

static void free_ws(gradflow_workspace *ws)
{
    #pragma acc exit data delete(ws->W1[0:8], ws->W2[0:8], ws->W2prime[0:8], ws->staples[0:8], ws->exp_aux[0:8])
    #pragma acc exit data delete(ws->Z0[0:8], ws->Z1[0:8], ws->Z2[0:8], ws->Zcomb[0:8])

    free(ws->W1);
    free(ws->W2);
    free(ws->staples);
    free(ws->exp_aux);
    free(ws->Z0);
    free(ws->Z1);
    free(ws->Z2);
    free(ws->Zcomb);
    free(ws->W2prime);
}

static void meas_cb_plaq(int meas_idx, double t, void *user_data)
{
    (void)user_data;

    // configuration is conf_acc on device and computes plaquette on device
    double plaq = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);

    if (devinfo.myrank == 0) {
        printf("meas %d  t=%.18lf  plaq(norm)=%.18lf\n",
               meas_idx, t, plaq / GL_SIZE / 6.0 / 3.0);
    }
}

int main(int argc, char **argv)
{
#ifdef MULTIDEVICE
    MPI_Init(&argc, &argv);
    init_multidev1D(&devinfo);
#else
    devinfo.myrank = 0;
    devinfo.nranks = 1;
    devinfo.myrank_world = 0;
#endif

    if (argc < 7) {
        if (devinfo.myrank == 0) {
            fprintf(stderr,
                    "USAGE:\n"
                    "  %s <input.set> <conf_in> <use_ildg:0|1> <nsteps_or_num_meas> <dt_or_dt0> <out_name>\n"
                    "\n"
                    "Optional (enables adaptive flow):\n"
                    "  %s <input.set> <conf_in> <use_ildg:0|1> <num_meas> <dt0> <out_name> <delta> <dt_max> <meas_each> <time_bin>\n",
                    argv[0], argv[0]);
        }
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 1;
    }

    const char *setfile = argv[1];
    const char *conf_in = argv[2];
    const int use_ildg = atoi(argv[3]);
    const int nsteps_or_num_meas = atoi(argv[4]);
    const double dt_or_dt0 = atof(argv[5]);
    const char *out_name = argv[6];
    const int use_adaptive = (argc >= 11);

    // This initializes globals used throughout the code (mc_params/debug_settings/etc.).
    set_global_vars_and_fermions_from_input_file((char *)setfile);
    //debug reading geom from input
    if (geom_par.gnx == 0 || geom_par.gny == 0 || geom_par.gnz == 0 || geom_par.gnt == 0) {
    if (devinfo.myrank_world == 0) {
        fprintf(stderr,
                "attentionnnn: Geometry not read from input file  "
                "Falling back to hardcoded geometry  and default mapping.\n");
    }
    geom_par.gnx = GL_N0;
    geom_par.gny = GL_N1;
    geom_par.gnz = GL_N2;
    geom_par.gnt = GL_N3;

    // Safe default unique mapping
    geom_par.xmap = 0;
    geom_par.ymap = 1;
    geom_par.zmap = 2;
    geom_par.tmap = 3;
}

#ifndef __GNUC__
    // Match main.c device init 
    acc_device_t my_device_type = acc_device_nvidia;
#ifdef MULTIDEVICE
    select_init_acc_device(my_device_type,
        (devinfo.single_dev_choice + devinfo.myrank_world) % devinfo.proc_per_node);
#else
    select_init_acc_device(my_device_type, devinfo.single_dev_choice);
#endif
#endif

    set_geom_glv(&geom_par);
    // Allocate the standard OpenStaPLE buffers (including conf_acc).
    mem_alloc_core();
    mem_alloc_extended();
    mem_alloc_core_f();
    mem_alloc_extended_f();
    
    compute_nnp_and_nnm_openacc();
    #pragma acc enter data copyin(nnp_openacc)
    #pragma acc enter data copyin(nnm_openacc)
    // Read configuration into conf_acc
    int file_conf_id = 0;
    if (read_conf_wrapper(conf_acc, conf_in, &file_conf_id, use_ildg)) {
        if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to read conf %s\n", conf_in);
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 2;
    }
    #pragma acc update device(conf_acc[0:8])

    //plaquette before flow
    double plaq0 = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);
    if (devinfo.myrank == 0) {
        printf("Plaquette (raw sum)  before flow: %.18lf\n", plaq0);
        printf("Plaquette (normalized) before flow: %.18lf\n", plaq0 / GL_SIZE / 6.0 / 3.0);
    }
    gradflow_workspace ws;
    memset(&ws, 0, sizeof(ws));
    alloc_ws(&ws);
    // debuggino
    {
        const double dt_force = 1.0;
        gradflow_compute_Z0_wilson(conf_acc, &ws, dt_force);
        const double z0_norm = calc_force_norm(ws.Z0);

        if (devinfo.myrank == 0) {
            printf("Z0 norm (dt=1) = %.18e\n", z0_norm);
        }
    }

    if (!use_adaptive) {
        for (int i = 0; i < nsteps_or_num_meas; i++) {
            gradflow_wilson_RKstep(conf_acc, &ws, dt_or_dt0);

            if ((i % 10) == 0) {
                double plaqi = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);
                if (devinfo.myrank == 0)
                    printf("step %d  plaq(norm)=%.18lf\n", i + 1, plaqi / GL_SIZE / 6.0 / 3.0);
            }
        }
    } else {
        const double delta    = atof(argv[7]);
        const double dt_max   = atof(argv[8]);
        const double meas_each = atof(argv[9]);
        const double time_bin  = atof(argv[10]);

        gradflow_adaptive_meas_params p;
        p.dt0 = dt_or_dt0;
        p.delta = delta;
        p.dt_max = dt_max;
        p.num_meas = nsteps_or_num_meas;
        p.meas_each = meas_each;
        p.time_bin = time_bin;

        if (devinfo.myrank == 0) {
            printf("AGF: num_meas=%d dt0=%.18lf delta=%.3e dt_max=%.18lf meas_each=%.18lf time_bin=%.3e\n",
                   p.num_meas, p.dt0, p.delta, p.dt_max, p.meas_each, p.time_bin);
        }

        gradflow_perform_measures_localobs_adaptive(conf_acc, &ws, &p, meas_cb_plaq, NULL, NULL);
    }
    
    // plaquette after flow
    double plaq1 = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);
    if (devinfo.myrank == 0) {
        printf("Plaquette (raw sum)  after  flow: %.18lf\n", plaq1);
        printf("Plaquette (normalized) after  flow: %.18lf\n", plaq1 / GL_SIZE / 6.0 / 3.0);
    }

    // Bring back and write.
    #pragma acc update host(conf_acc[0:8])
    save_conf_wrapper(conf_acc, out_name, file_conf_id, use_ildg);

    free_ws(&ws);

#ifdef MULTIDEVICE
    MPI_Finalize();
#endif
    return 0;
}