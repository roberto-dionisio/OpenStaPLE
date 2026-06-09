#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "../Include/setting_file_parser.h"
#include "../OpenAcc/alloc_vars.h"
#include "../OpenAcc/geometry.h"
#include "../OpenAcc/io.h"
#include "../OpenAcc/deviceinit.h"
#include "../OpenAcc/cooling.h"
#include "../OpenAcc/su3_utilities.h"
#include "../Meas/gauge_meas.h"
#include "../Meas/topo_timeslice.h"
#include "../Mpi/multidev.h"

#ifndef __GNUC__
#include "openacc.h"
#endif

#ifdef MULTIDEVICE
#include <mpi.h>
#endif

int conf_id_iter;
int verbosity_lv;

static int gl_n_dir(int dir)
{
    switch(dir) {
        case 0: return GL_N0;
        case 1: return GL_N1;
        case 2: return GL_N2;
        default: return GL_N3;
    }
}


/*File and dir entries as it has been done for gradflowtest*/
typedef struct conf_entry_t { long long id; char *name; } conf_entry;

static long long parse_conf_id_from_name(const char *fname)
{
    const char *dot = strrchr(fname, '.');
    if(!dot || !dot[1]) return -1;
    char *endp = NULL;
    errno = 0;
    long long v = strtoll(dot + 1, &endp, 10);
    if(errno != 0 || endp == dot + 1 || (endp && *endp != '\0')) return -1;
    return v;
}

static int starts_with(const char *s, const char *prefix)
{
    if(!prefix || !*prefix) return 1;
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static char *path_join2(const char *dir, const char *name)
{
    if(!dir || !*dir || strcmp(dir, ".") == 0) return strdup(name);
    const size_t dl = strlen(dir);
    const size_t nl = strlen(name);
    const int need_slash = (dl > 0 && dir[dl - 1] != '/');
    const size_t out_len = dl + (need_slash ? 1 : 0) + nl + 1;
    char *out = (char *)malloc(out_len);
    if(!out) return NULL;
    if(need_slash) snprintf(out, out_len, "%s/%s", dir, name);
    else           snprintf(out, out_len, "%s%s",  dir, name);
    return out;
}

static int path_is_regular_file(const char *path)
{
    struct stat st;
    if(stat(path, &st) != 0) return 0;
    return S_ISREG(st.st_mode);
}

static int path_is_dir(const char *path)
{
    struct stat st;
    if(stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

static int cmp_conf_entry_by_id(const void *a, const void *b)
{
    const conf_entry *A = (const conf_entry *)a;
    const conf_entry *B = (const conf_entry *)b;
    if(A->id < B->id) return -1;
    if(A->id > B->id) return  1;
    return strcmp(A->name, B->name);
}

static void free_conf_list(conf_entry *list, int n)
{
    if(!list) return;
    for(int i = 0; i < n; i++) free(list[i].name);
    free(list);
}

static int list_confs_in_dir(const char *dirpath, const char *prefix,
                              conf_entry **out_list, int *out_n)
{
    *out_list = NULL; *out_n = 0;
    DIR *dir = opendir(dirpath);
    if(!dir) return 1;
    int cap = 128, n = 0;
    conf_entry *list = (conf_entry *)calloc((size_t)cap, sizeof(*list));
    if(!list) { closedir(dir); return 1; }
    struct dirent *de = NULL;
    while((de = readdir(dir)) != NULL) {
        const char *name = de->d_name;
        if(name[0] == '.') continue;
        if(!starts_with(name, prefix)) continue;
        if(de->d_type != DT_REG && de->d_type != DT_UNKNOWN) continue;
        char *fullpath = path_join2(dirpath, name);
        if(!fullpath) { free_conf_list(list, n); closedir(dir); return 1; }
        struct stat st;
        if(stat(fullpath, &st) != 0) { free(fullpath); continue; }
        long long id = parse_conf_id_from_name(name);
        if(id < 0) { free(fullpath); continue; }
        if(n == cap) {
            cap *= 2;
            conf_entry *tmp = (conf_entry *)realloc(list, (size_t)cap * sizeof(*list));
            if(!tmp) { free(fullpath); free_conf_list(list, n); closedir(dir); return 1; }
            list = tmp;
        }
        list[n].id = id; list[n].name = fullpath; n++;
    }
    closedir(dir);
    if(n == 0) { free(list); list = NULL; }
    else qsort(list, (size_t)n, sizeof(*list), cmp_conf_entry_by_id);
    *out_list = list; *out_n = n;
    return 0;
}

//Output
static void write_file_header(FILE *fp, const char *dir_label,
                               int Nt, int Ns)
{
    fprintf(fp, "# topo_timeslice_meas: momentum direction=%s  Nt=%d  Ns=%d\n",
            dir_label, Nt, Ns);
    fprintf(fp, "# p = 2*pi/Ns * k\n");
    fprintf(fp, "# conf_id  cool_step  k");
    for(int t = 0; t < Nt; t++)
        fprintf(fp, "  Q_re(t=%d) Q_im(t=%d)", t, t);
    fprintf(fp, "\n");
    fflush(fp);
}

/* Write one row for a given k value */
static void write_row(FILE *fp, long long conf_id, int cool_step, int k,
                      int Nt, int nk,
                      const double *Q_re, const double *Q_im)
{
    fprintf(fp, "%lld  %d  %d", conf_id, cool_step, k);
    for(int t = 0; t < Nt; t++)
        fprintf(fp, "  %.15e %.15e",
                Q_re[t * nk + k], Q_im[t * nk + k]);
    fprintf(fp, "\n");
}


static void measure_and_write(
    su3_soa *conf_to_meas,
    long long conf_id, int cool_step,
    int tmap, int xmap, int ymap, int zmap,
    int Nt, int Nx, int Ny, int Nz,
    double *Q_re_x, double *Q_im_x,
    double *Q_re_y, double *Q_im_y,
    double *Q_re_z, double *Q_im_z,
    FILE *fp_x, FILE *fp_y, FILE *fp_z)
{
    compute_topological_charge_density(conf_to_meas, auxbis_conf_acc, topo_loc);

#pragma acc update host(topo_loc[0:2])

    compute_topo_timeslice_momenta(topo_loc, xmap, Nt, Nx, Q_re_x, Q_im_x);
    compute_topo_timeslice_momenta(topo_loc, ymap, Nt, Ny, Q_re_y, Q_im_y);
    compute_topo_timeslice_momenta(topo_loc, zmap, Nt, Nz, Q_re_z, Q_im_z);

    if(devinfo.myrank != 0) return;

    const int nkx = Nx / 2 + 1;
    const int nky = Ny / 2 + 1;
    const int nkz = Nz / 2 + 1;

    for(int k = 0; k <= Nx / 2; k++)
        write_row(fp_x, conf_id, cool_step, k, Nt, nkx, Q_re_x, Q_im_x);
    for(int k = 0; k <= Ny / 2; k++)
        write_row(fp_y, conf_id, cool_step, k, Nt, nky, Q_re_y, Q_im_y);
    for(int k = 0; k <= Nz / 2; k++)
        write_row(fp_z, conf_id, cool_step, k, Nt, nkz, Q_re_z, Q_im_z);

    fflush(fp_x); fflush(fp_y); fflush(fp_z);
}


/*Usage ads in gradflow*/
static void print_usage(const char *prog)
{
    fprintf(stderr,
        "USAGE:\n"
        "  %s <input.set> <conf_prefix|conf_file> <use_ildg:0|1>"
        " <ncool> <cool_each> [options]\n"
        "\n"
        "  Measures Q(t,k) = Sum_{x_s,x_perp} q(x) exp(i*2*pi/Ns*k*x_s)\n"
        "  for k=0..Ns/2, at cool_step=0,cool_each,2*cool_each,...,ncool.\n"
        "  Produces three files (one per spatial direction).\n"
        "\n"
        "Options:\n"
        "  --out <prefix>       Output file prefix (default: topo_timeslice)\n"
        "  --conf-stride <N>    Use every N-th conf (default: 1)\n"
        "  -v <lvl>             Verbosity level\n",
        prog);
}

//driver executable for topotslice measurement 
int main(int argc, char **argv)
{
#ifdef MULTIDEVICE
    MPI_Init(&argc, &argv);
    init_multidev1D(&devinfo);
#else
    devinfo.myrank       = 0;
    devinfo.nranks       = 1;
    devinfo.myrank_world = 0;
#endif

    if(argc < 6) {
        if(devinfo.myrank == 0) print_usage(argv[0]);
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 1;
    }

    const char *setfile        = argv[1];
    const char *prefix_or_file = argv[2];
    const int   use_ildg       = atoi(argv[3]);
    const int   ncool          = atoi(argv[4]);
    const int   cool_each      = atoi(argv[5]);

    const char *out_prefix = "topo_timeslice";
    int conf_stride = 1;

    for(int i = 6; i < argc; i++) {
        if(strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if(strcmp(argv[i], "--conf-stride") == 0 && i + 1 < argc) {
            conf_stride = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            verbosity_lv = atoi(argv[++i]);
        } else if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            if(devinfo.myrank == 0) print_usage(argv[0]);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 0;
        } else {
            if(devinfo.myrank == 0)
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 1;
        }
    }

    if(ncool < 0 || cool_each <= 0 || conf_stride <= 0) {
        if(devinfo.myrank == 0)
            fprintf(stderr, "ERROR: ncool>=0, cool_each>0, conf_stride>0 required.\n");
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 1;
    }

    //geometry and dev setup
    set_global_vars_and_fermions_from_input_file((char *)setfile);

    if(geom_par.gnx == 0 || geom_par.gny == 0 ||
       geom_par.gnz == 0 || geom_par.gnt == 0) {
        if(devinfo.myrank_world == 0)
            fprintf(stderr,
                "WARNING: geometry not in input file, falling back to "
                "compile-time defaults.\n");
        geom_par.gnx  = GL_N0; geom_par.gny  = GL_N1;
        geom_par.gnz  = GL_N2; geom_par.gnt  = GL_N3;
        geom_par.xmap = 0;     geom_par.ymap = 1;
        geom_par.zmap = 2;     geom_par.tmap = 3;
    }

#ifndef __GNUC__
    acc_device_t my_device_type = acc_device_nvidia;
#ifdef MULTIDEVICE
    select_init_acc_device(my_device_type,
        (devinfo.single_dev_choice + devinfo.myrank_world) % devinfo.proc_per_node);
#else
    select_init_acc_device(my_device_type, devinfo.single_dev_choice);
#endif
#endif

    set_geom_glv(&geom_par);
    mem_alloc_core();
    mem_alloc_extended();
    mem_alloc_core_f();
    mem_alloc_extended_f();

    compute_nnp_and_nnm_openacc();
#pragma acc enter data copyin(nnp_openacc)
#pragma acc enter data copyin(nnm_openacc)

    const int tmap = geom_par.tmap;
    const int xmap = geom_par.xmap;
    const int ymap = geom_par.ymap;
    const int zmap = geom_par.zmap;

    const int Nt = gl_n_dir(tmap);
    const int Nx = gl_n_dir(xmap);
    const int Ny = gl_n_dir(ymap);
    const int Nz = gl_n_dir(zmap);

    const int nkx = Nx / 2 + 1;
    const int nky = Ny / 2 + 1;
    const int nkz = Nz / 2 + 1;

    double *Q_re_x = (double *)malloc((size_t)(Nt * nkx) * sizeof(double));
    double *Q_im_x = (double *)malloc((size_t)(Nt * nkx) * sizeof(double));
    double *Q_re_y = (double *)malloc((size_t)(Nt * nky) * sizeof(double));
    double *Q_im_y = (double *)malloc((size_t)(Nt * nky) * sizeof(double));
    double *Q_re_z = (double *)malloc((size_t)(Nt * nkz) * sizeof(double));
    double *Q_im_z = (double *)malloc((size_t)(Nt * nkz) * sizeof(double));

    if(!Q_re_x || !Q_im_x || !Q_re_y || !Q_im_y || !Q_re_z || !Q_im_z) {
        if(devinfo.myrank == 0) fprintf(stderr, "ERROR: malloc failed for Q buffers.\n");
        free(Q_re_x); free(Q_im_x); free(Q_re_y); free(Q_im_y);
        free(Q_re_z); free(Q_im_z);
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 2;
    }

    //output file names
    const size_t prefix_len = strlen(out_prefix);
    char *fname_x = (char *)malloc(prefix_len + 3);
    char *fname_y = (char *)malloc(prefix_len + 3);
    char *fname_z = (char *)malloc(prefix_len + 3);
    if(!fname_x || !fname_y || !fname_z) {
        if(devinfo.myrank == 0) fprintf(stderr, "ERROR: malloc failed for filenames.\n");
#ifdef MULTIDEVICE
        MPI_Finalize();
#endif
        return 2;
    }
    snprintf(fname_x, prefix_len + 3, "%s_x", out_prefix);
    snprintf(fname_y, prefix_len + 3, "%s_y", out_prefix);
    snprintf(fname_z, prefix_len + 3, "%s_z", out_prefix);

    FILE *fp_x = NULL, *fp_y = NULL, *fp_z = NULL;
    if(devinfo.myrank == 0) {
        fp_x = fopen(fname_x, "w");
        fp_y = fopen(fname_y, "w");
        fp_z = fopen(fname_z, "w");
        if(!fp_x || !fp_y || !fp_z) {
            fprintf(stderr, "ERROR: cannot open output files: %s\n",
                    strerror(errno));
            if(fp_x) fclose(fp_x);
            if(fp_y) fclose(fp_y);
            if(fp_z) fclose(fp_z);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        write_file_header(fp_x, "x", Nt, Nx);
        write_file_header(fp_y, "y", Nt, Ny);
        write_file_header(fp_z, "z", Nt, Nz);

        printf("=== topo_timeslice_meas ===\n");
        printf("setfile   : %s\n", setfile);
        printf("input     : %s\n", prefix_or_file);
        printf("use_ildg  : %d\n", use_ildg);
        printf("ncool     : %d\n", ncool);
        printf("cool_each : %d\n", cool_each);
        printf("out_prefix: %s\n", out_prefix);
        printf("Geometry  : Nt=%d Nx=%d Ny=%d Nz=%d (tmap=%d xmap=%d ymap=%d zmap=%d)\n",
               Nt, Nx, Ny, Nz, tmap, xmap, ymap, zmap);
        printf("Momenta   : kx=0..%d  ky=0..%d  kz=0..%d\n",
               Nx/2, Ny/2, Nz/2);
        fflush(stdout);
    }

    //build conf list
    conf_entry *list = NULL;
    int nlist = 0;

    if(path_is_regular_file(prefix_or_file)) {
        list = (conf_entry *)calloc(1, sizeof(*list));
        if(!list) {
            if(devinfo.myrank == 0) fprintf(stderr, "ERROR: malloc failed.\n");
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        list[0].name = strdup(prefix_or_file);
        list[0].id   = parse_conf_id_from_name(prefix_or_file);
        nlist = 1;
    } else {
        const char *scan_dir    = ".";
        const char *scan_prefix = prefix_or_file;
        char *dir_buf = NULL;

        if(path_is_dir(prefix_or_file)) {
            scan_dir    = prefix_or_file;
            scan_prefix = "";
        } else {
            const char *slash = strrchr(prefix_or_file, '/');
            if(slash) {
                const size_t dl = (size_t)(slash - prefix_or_file);
                dir_buf = (char *)malloc(dl + 1);
                if(!dir_buf) {
                    if(devinfo.myrank == 0) fprintf(stderr, "ERROR: OOM.\n");
#ifdef MULTIDEVICE
                    MPI_Finalize();
#endif
                    return 2;
                }
                memcpy(dir_buf, prefix_or_file, dl);
                dir_buf[dl] = '\0';
                scan_dir    = (dl == 0) ? "/" : dir_buf;
                scan_prefix = slash + 1;
            }
        }

        if(list_confs_in_dir(scan_dir, scan_prefix, &list, &nlist) != 0 || nlist == 0) {
            if(devinfo.myrank == 0)
                fprintf(stderr, "ERROR: no conf files found for: %s\n", prefix_or_file);
            free(dir_buf);
#ifdef MULTIDEVICE
            MPI_Finalize();
#endif
            return 2;
        }
        free(dir_buf);
    }

    if(devinfo.myrank == 0)
        printf("Found %d configuration(s).\n\n", nlist);

    
    int processed = 0;
    for(int idx = 0; idx < nlist; idx += conf_stride) {
        const char *conf_in = list[idx].name;
        int file_conf_id = 0;

        if(devinfo.myrank == 0) {
            printf("=== Processing conf: %s ===\n", conf_in);
            fflush(stdout);
        }

        if(read_conf_wrapper(conf_acc, conf_in, &file_conf_id, use_ildg)) {
            if(devinfo.myrank == 0)
                fprintf(stderr, "ERROR: failed to read conf: %s\n", conf_in);
            continue;
        }

#pragma acc update device(conf_acc[0:8])

        const long long conf_id = (file_conf_id != 0)
                                  ? (long long)file_conf_id
                                  : list[idx].id;

        // calda
        measure_and_write(
            conf_acc, conf_id, 0,
            tmap, xmap, ymap, zmap,
            Nt, Nx, Ny, Nz,
            Q_re_x, Q_im_x, Q_re_y, Q_im_y, Q_re_z, Q_im_z,
            fp_x, fp_y, fp_z);

        if(verbosity_lv >= 1 && devinfo.myrank == 0) {
            printf("  cool_step=0 written.\n");
            fflush(stdout);
        }

        // raffredata
        if(ncool > 0) {
            su3_soa *conf_to_use = conf_acc;
            for(int cs = 1; cs <= ncool; cs++) {
                //done in place
                cool_conf(conf_to_use, aux_conf_acc, auxbis_conf_acc);
                conf_to_use = aux_conf_acc;

                if(cs % cool_each == 0) {
                    measure_and_write(
                        aux_conf_acc, conf_id, cs,
                        tmap, xmap, ymap, zmap,
                        Nt, Nx, Ny, Nz,
                        Q_re_x, Q_im_x, Q_re_y, Q_im_y, Q_re_z, Q_im_z,
                        fp_x, fp_y, fp_z);

                    if(verbosity_lv >= 1 && devinfo.myrank == 0) {
                        printf("  cool_step=%d written.\n", cs);
                        fflush(stdout);
                    }
                }
            }
        }

        processed++;
        if(devinfo.myrank == 0) {
            printf("  Done conf_id=%lld\n\n", conf_id);
            fflush(stdout);
        }
    }

    if(devinfo.myrank == 0) {
        printf("=========================================\n");
        printf("Done. Processed %d conf(s).\n", processed);
        printf("Output: %s  %s  %s\n", fname_x, fname_y, fname_z);
        if(fp_x) fclose(fp_x);
        if(fp_y) fclose(fp_y);
        if(fp_z) fclose(fp_z);
    }

    free(Q_re_x); free(Q_im_x);
    free(Q_re_y); free(Q_im_y);
    free(Q_re_z); free(Q_im_z);
    free(fname_x); free(fname_y); free(fname_z);
    free_conf_list(list, nlist);

#ifdef MULTIDEVICE
    MPI_Finalize();
#endif
    return 0;
}
