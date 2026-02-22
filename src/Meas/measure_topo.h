#ifndef MEASURE_TOPO_H_
#define MEASURE_TOPO_H_

#include "../Include/common_defines.h"

typedef struct meastopo_param_t{

    int meascool;
    char pathcool[20];
    int coolmeasstep;
    int cool_measinterval;
    int cooleach;
    
    int measstout;
    char pathstout[20];
    double measrhostout; //ONLY FOR CHECKING
    int stoutmeasstep;
    int stout_measinterval;
    int stouteach;

    int measgradflow;        // 0/1
    int floweach;
    int    flow_num_meas;    // number of measurements to record
    double flow_dt0;         // initial dt
    double flow_delta;       // acceptance threshold
    double flow_dt_max;      // max dt
    double flow_meas_each;   // target spacing in t
    double flow_time_bin;    // bin around target times
    char pathflow[50]; 
    
} meastopo_param;

extern meastopo_param meastopo_params;

#endif
