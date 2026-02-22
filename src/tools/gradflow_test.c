#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

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

typedef struct conf_entry_t{
    long long id;
    char *name;
}conf_entry;

typedef struct meas_store_t {
    int num_meas;
    double meas_each;
    double time_bin;
    double *plaq_norm;   // size num_meas+1, plaq_norm[0] is t=0
    double *t_meas;      // size num_meas+1, t_meas[0] = 0
} meas_store;

static int file_exists(const char *path)
{
    struct stat st;
    return (stat(path, &st) == 0);
}

static long long parse_conf_id_from_name(const char *fname)
{
    // expected something.{digits}  (having in minnd stored_conf.00010)
    const char *dot = strrchr(fname, '.');
    if (!dot || !dot[1]) return -1;

    char *endp = NULL;
    errno = 0;
    long long v = strtoll(dot + 1, &endp, 10);
    if (errno != 0 || endp == (dot + 1) || (endp && *endp != '\0')) return -1;
    return v;
}

static int starts_with(const char *s, const char *prefix)
{
    if (!prefix || !*prefix) return 1;
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int cmp_conf_entry_by_id(const void *a, const void *b)
{
    const conf_entry *A = (const conf_entry *)a;
    const conf_entry *B = (const conf_entry *)b;
    if (A->id < B->id) return -1;
    if (A->id > B->id) return 1;
    return strcmp(A->name, B->name);
}

static void free_conf_list(conf_entry *list, int n)
{
    if (!list) return;
    for (int i = 0; i < n; i++) free(list[i].name);
    free(list);
}

// prepare list of confs in current dir matching pattern and sorted by id
static int list_confs_in_cwd(const char *prefix, conf_entry **out_list, int *out_n)
{
    *out_list = NULL;
    *out_n = 0;

    DIR *dir = opendir(".");
    if (!dir) return 1;

    int cap = 128;
    int n = 0;
    conf_entry *list = (conf_entry *)calloc((size_t)cap, sizeof(*list));
    if (!list) {
        closedir(dir);
        return 1;
    }

    struct dirent *de = NULL;
    while ((de = readdir(dir)) != NULL) {
        const char *name = de->d_name;

        if (name[0] == '.') continue;
        if (!starts_with(name, prefix)) continue;
        if (de->d_type != DT_REG && de->d_type != DT_UNKNOWN) continue;

        if (!file_exists(name)) continue;

        long long id = parse_conf_id_from_name(name);
        if (id < 0) continue;

        if (n == cap) {
            cap *= 2;
            conf_entry *tmp = (conf_entry *)realloc(list, (size_t)cap * sizeof(*list));
            if (!tmp) {
                free_conf_list(list, n);
                closedir(dir);
                return 1;
            }
            list = tmp;
        }

        list[n].id = id;
        list[n].name = strdup(name);
        if (!list[n].name) {
            free_conf_list(list, n);
            closedir(dir);
            return 1;
        }
        n++;
    }
    closedir(dir);

    if (n == 0) {
        free(list);
        list = NULL;
    } else {
        qsort(list, (size_t)n, sizeof(*list), cmp_conf_entry_by_id);
    }

    *out_list = list;
    *out_n = n;
    return 0;
}



static void log_measure_config(const char *setfile,
                         const char *prefix_or_file,
                         int use_ildg,
                         const gradflow_adaptive_meas_params *p,
                         int conf_stride,
                         const char *out_file,
                         int nlist)
{
    if (devinfo.myrank != 0) return;
    printf( "=== OpenStaPLE offline adaptive gradient-flow measurement ===\n");
    printf( "setfile: %s\n", setfile);
    printf( "input:   %s\n", prefix_or_file);
    printf( "use_ildg: %d\n", use_ildg);
    printf("out_file: %s\n", out_file);
    printf("Found confs: %d\n", nlist);
    printf("conf_stride: %d\n", conf_stride);
    printf("Geometry: gnx=%d gny=%d gnz=%d gnt=%d (GL_SIZE=%d)\n",
           geom_par.gnx, geom_par.gny, geom_par.gnz, geom_par.gnt, GL_SIZE);
    printf("flow(method): wilson_RK3_adaptive\n");
    printf("params      : num_meas=%d dt0=%.18g delta=%.3e dt_max=%.18g meas_each=%.18g time_bin=%.3e\n",
           p->num_meas, p->dt0, p->delta, p->dt_max, p->meas_each, p->time_bin);
    printf("output      : first column is conf_id; then plaq_norm at t=0, meas_each, 2*meas_each, ...\n");
    printf("plaq_norm   : plaq_raw / (GL_SIZE * 6 * 3)\n");
    
    fflush(stdout);
}

static void write_header(FILE *fp, const gradflow_adaptive_meas_params *p){
    fprintf(fp, "# conf_id");
    fprintf(fp, " t=0");
    for (int k = 1; k <= p->num_meas; k++) {
        fprintf(fp, " t=%.18g", p->meas_each * (double)k);
    }
    fprintf(fp, "\n");
    fflush(fp);
}


static void meas_cb_store_plaq(int meas_idx, double t, void *user_data)
{
    meas_store *S = (meas_store *)user_data;
    if (!S) return;

    // meas_idx=0 corresponds to the first target ~ meas_each*1
    const int out_idx = meas_idx + 1;
    if (out_idx < 1 || out_idx > S->num_meas) return;

    const double plaq = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);
    const double plaq_norm = plaq / GL_SIZE / 6.0 / 3.0;

    S->plaq_norm[out_idx] = plaq_norm;
    S->t_meas[out_idx] = t;

    if (devinfo.myrank == 0 && verbosity_lv >= 2) {
        const double target = S->meas_each * (double)out_idx;
        printf("  meas %d/%d  t=%.18lf (target=%.18lf, |dt|=%.3e)  plaq(norm)=%.18lf\n",
               out_idx, S->num_meas, t, target, fabs(t - target), plaq_norm);
        fflush(stdout);
    }
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "USAGE (AGF offline scan):\n"
            "  %s <input.set> <conf_prefix|conf_file> <use_ildg:0|1> <num_meas> <dt0> <delta> <dt_max> <meas_each> <time_bin> [options]\n"
            "\n"
            "Options:\n"
            "  --out <file>         Output file (default: gradflow_measure)\n"
            "  --conf-stride <N>    Use every N-th conf from the sorted list (default: 1)\n"
            "  -v <lvl>             Set verbosity_lv (default: keep current)\n"
            "\n"
            "Notes:\n"
            "  - If <conf_prefix|conf_file> is an existing file, only that file is processed.\n"
            "  - Otherwise scans current directory for files starting with prefix and having numeric suffix after last '.'\n"
            "    e.g. stored_conf.00010\n",
            prog);
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

    if (argc < 10) {
        if (devinfo.myrank == 0) {
            print_usage(argv[0]);
        }
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 1;
    }

    const char *setfile = argv[1];
    const char *prefix_or_file = argv[2];
    const int use_ildg = atoi(argv[3]);

    gradflow_adaptive_meas_params p;
    p.num_meas  = atoi(argv[4]);
    p.dt0       = atof(argv[5]);
    p.delta     = atof(argv[6]);
    p.dt_max    = atof(argv[7]);
    p.meas_each = atof(argv[8]);
    p.time_bin  = atof(argv[9]);

    const char *out_file = "gradflow_measure";
    int conf_stride = 1;

    //options pars
    for (int i = 10; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && (i + 1) < argc) {
            out_file = argv[++i];
        } else if (strcmp(argv[i], "--conf-stride") == 0 && (i + 1) < argc) {
            conf_stride = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0 && (i + 1) < argc) {
            verbosity_lv = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            if (devinfo.myrank == 0) print_usage(argv[0]);

#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 0;
        } else {
            if (devinfo.myrank == 0) {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_usage(argv[0]);
            }
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 1;
        }
    }
    if (p.num_meas <= 0 || p.dt0 <= 0.0 || p.delta <= 0.0 || p.dt_max <= 0.0 || p.meas_each <= 0.0 || p.time_bin < 0.0 || conf_stride <= 0) {
        if (devinfo.myrank == 0) fprintf(stderr, "ERROR: invalid parameters.\n");
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 1;
    }

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
    
    gradflow_workspace ws;
    if (gradflow_ws_alloc(&ws)) {
        if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to allocate gradient flow workspace.\n");
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 2;
    }

    //list of confs to process
    conf_entry *list = NULL;
    int nlist = 0;
    const int single_file_mode = file_exists(prefix_or_file);
    if (single_file_mode) {
        list = (conf_entry *)calloc(1, sizeof(*list));
        if (!list) {
            if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to allocate memory for conf list.\n");
            gradflow_ws_free(&ws);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        list[0].name = strdup(prefix_or_file);
        list[0].id = parse_conf_id_from_name(prefix_or_file);
        if (!list[0].name || list[0].id < 0) {
            if (devinfo.myrank == 0) fprintf(stderr, "ERROR: invalid conf file name: %s\n", prefix_or_file);
            free(list);
            gradflow_ws_free(&ws);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        nlist = 1;
    } else {
        if (list_confs_in_cwd(prefix_or_file, &list, &nlist) != 0) {
            if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to list conf files in current directory with prefix: %s\n", prefix_or_file);
            gradflow_ws_free(&ws);
            #ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        if (nlist == 0) {
            if (devinfo.myrank == 0) fprintf(stderr, "ERROR: no conf files found in current directory with prefix: %s\n", prefix_or_file);
            gradflow_ws_free(&ws);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
    }


    FILE *fp=NULL;
    if (devinfo.myrank == 0) {
        fp = fopen(out_file, "w");
        if (!fp) {
            fprintf(stderr, "ERROR: failed to open measure out file: %s: %s\n", out_file, strerror(errno));
            free_conf_list(list, nlist);
            gradflow_ws_free(&ws);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        log_measure_config(setfile, prefix_or_file, use_ildg, &p, conf_stride, out_file, nlist);
        write_header(fp, &p);
    }   

    //Allocate meas buffer
    const int nt = p.num_meas + 1;
    double *plaq_norm = (double *)malloc((size_t)nt * sizeof(double));
    double *t_meas = (double *)malloc((size_t)nt * sizeof(double));
    if (!plaq_norm || !t_meas) {
        if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to allocate measurement buffers\n");
        free(plaq_norm);
        free(t_meas);
        if (devinfo.myrank == 0 && fp) fclose(fp);
        free_conf_list(list, nlist);
        gradflow_ws_free(&ws);
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 2;
    }

    int processed = 0;
    for (int idx =0 ; idx < nlist; idx+=conf_stride) {
        const char *conf_in = list[idx].name;
        int file_conf_id= 0;
        if (devinfo.myrank == 0) {
            printf("\n===Processing conf: %s ===\n", conf_in);
            fflush(stdout);
        }
        if (read_conf_wrapper(conf_acc, conf_in, &file_conf_id, use_ildg)) {
            if (devinfo.myrank == 0) fprintf(stderr, "ERROR: failed to read conf file: %s\n", conf_in);
            continue;
        }

        #pragma acc update device(conf_acc[0:8])
        for (int k=0; k<nt; k++){
            plaq_norm[k] = NAN;
            t_meas[k] =NAN;
        }
        //t=0
        double plaq0 = calc_plaquette_soloopenacc(conf_acc, aux_conf_acc, local_sums);
        plaq_norm[0] = plaq0 / GL_SIZE / 6.0 / 3.0;
        t_meas[0] = 0.0;
        if (devinfo.myrank == 0) {
            printf("t=0  plaq(norm)=%.18lf\n", plaq_norm[0]);
            fflush(stdout);
        }

        meas_store S;
        S.num_meas = p.num_meas;
        S.meas_each = p.meas_each;
        S.time_bin = p.time_bin;
        S.plaq_norm = plaq_norm;
        S.t_meas = t_meas;

        if (devinfo.myrank == 0) {
             printf("AGF params: num_meas=%d dt0=%.6e delta=%.3e dt_max=%.6e meas_each=%.6e time_bin=%.3e\n",
                   p.num_meas, p.dt0, p.delta, p.dt_max, p.meas_each, p.time_bin);
            fflush(stdout);
        }

        gradflow_perform_measures_localobs_adaptive(conf_acc, &ws, &p, meas_cb_store_plaq, &S , NULL);
            
        long long conf_id = (file_conf_id != 0) ? (long long)file_conf_id : list[idx].id;

        //sanity: All measures filled?
        int missing = 0;
        for (int k = 0; k < nt; k++) {
            if (!isfinite(plaq_norm[k])) missing++;
        }
        if (missing != 0 && devinfo.myrank == 0) {
            fprintf(stderr, "WARNING: conf_id=%lld  missing %d/%d measurements\n", conf_id, missing, nt);
        }


        // Write as rows senno' Andrea si arrabbia
        if (devinfo.myrank == 0){
            fprintf(fp, "%lld", conf_id);
            for (int k =0; k <nt; k++) {
                fprintf(fp, " %.18lf", plaq_norm[k]);
            }
            fprintf(fp, "\n");
            fflush(fp);
            printf("Measured conf_id=%lld  plaq(t+0)=%.18lf  plaq(t_final=%.18lf)=%.18lf\n",
                   conf_id, plaq_norm[0], t_meas[p.num_meas], plaq_norm[p.num_meas]);
        }
        processed++;
    }
    if (devinfo.myrank == 0) {
        printf("\n \n");
        printf("\n=========================================\n");
        printf("\nDone. AGF offline: processed %d confs.\n", processed);
        printf("Output written to: %s\n", out_file);
        printf("Ciao!\n");
        fflush(stdout);
        fclose(fp);
    }

    free(plaq_norm);
    free(t_meas);
    free_conf_list(list, nlist);
    gradflow_ws_free(&ws);

#ifdef MULTIDEVICE
    MPI_Finalize();
#endif
    return 0;
}