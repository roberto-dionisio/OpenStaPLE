#ifndef STOUTING_H
#define STOUTING_H

#include "./struct_c_def.h"
#include "../Include/common_defines.h"

// if using GCC, there are some problems with __restrict.
#ifdef __GNUC__
#define __restrict
#endif

#pragma acc routine seq
static inline void conf_left_exp_multiply_to_su3_soa(__restrict const su3_soa * const cnf, const int idx,
																										 __restrict su3_soa * const  EXP,
																										 __restrict su3_soa * const cnf_out)
{
	single_su3 AUX;
	// multiply: U_new = EXP * U_old
	AUX.comp[0][0] = cnf->r0.c0[idx];
	AUX.comp[0][1] = cnf->r0.c1[idx];
	AUX.comp[0][2] = cnf->r0.c2[idx];
	AUX.comp[1][0] = cnf->r1.c0[idx];
	AUX.comp[1][1] = cnf->r1.c1[idx];
	AUX.comp[1][2] = cnf->r1.c2[idx];
	// rebuild third row
	AUX.comp[2][0] = conj(AUX.comp[0][1] * AUX.comp[1][2] - AUX.comp[0][2] * AUX.comp[1][1]);
	AUX.comp[2][1] = conj(AUX.comp[0][2] * AUX.comp[1][0] - AUX.comp[0][0] * AUX.comp[1][2]);
	AUX.comp[2][2] = conj(AUX.comp[0][0] * AUX.comp[1][1] - AUX.comp[0][1] * AUX.comp[1][0]);

	// multiply
	cnf_out->r0.c0[idx] = EXP->r0.c0[idx] * AUX.comp[0][0] + EXP->r0.c1[idx] * AUX.comp[1][0] + EXP->r0.c2[idx] * AUX.comp[2][0];
	cnf_out->r0.c1[idx] = EXP->r0.c0[idx] * AUX.comp[0][1] + EXP->r0.c1[idx] * AUX.comp[1][1] + EXP->r0.c2[idx] * AUX.comp[2][1];
	cnf_out->r0.c2[idx] = EXP->r0.c0[idx] * AUX.comp[0][2] + EXP->r0.c1[idx] * AUX.comp[1][2] + EXP->r0.c2[idx] * AUX.comp[2][2];

	cnf_out->r1.c0[idx] = EXP->r1.c0[idx] * AUX.comp[0][0] + EXP->r1.c1[idx] * AUX.comp[1][0] + EXP->r1.c2[idx] * AUX.comp[2][0];
	cnf_out->r1.c1[idx] = EXP->r1.c0[idx] * AUX.comp[0][1] + EXP->r1.c1[idx] * AUX.comp[1][1] + EXP->r1.c2[idx] * AUX.comp[2][1];
	cnf_out->r1.c2[idx] = EXP->r1.c0[idx] * AUX.comp[0][2] + EXP->r1.c1[idx] * AUX.comp[1][2] + EXP->r1.c2[idx] * AUX.comp[2][2];

}

void exp_minus_QA_times_conf(__restrict const su3_soa * const tu,
														 __restrict const tamat_soa * QA,
														 __restrict su3_soa * const tu_out,
														 __restrict su3_soa * const exp_aux);

void stout_isotropic(__restrict const su3_soa * const u, // input conf
										 __restrict su3_soa * const uprime, // output conf [stouted]
										 __restrict su3_soa * const local_staples, // parking variable
										 __restrict su3_soa * const auxiliary, // parking variable
										 __restrict tamat_soa * const tipdot, // parking variable
										 const int istopo); // istopo = {0,1} -> rho = {fermrho,toporho}


void compute_lambda(__restrict thmat_soa * const L, // Lambda --> ouput (this is for next step fermion force computation)
										__restrict const su3_soa   * const SP, // Sigma primo --> input (previous step fermion force)
										__restrict const su3_soa   * const U, // gauge configuration --> input
										__restrict const tamat_soa * const QA, // Cayley Hamilton Qs --> input (rho*ta(staples))
										__restrict su3_soa   * const TMP // parking variable
										);

#if (defined STOUT_FERMIONS) || (defined STOUT_TOPO)
void stout_wrapper(__restrict const su3_soa * const tconf_acc,
									 __restrict su3_soa * tstout_conf_acc_arr, const int istopo); // istopo = {0,1} ==> stout steps = {fermionic,topological} stout steps
#endif

void compute_sigma(__restrict const thmat_soa * const L, // Lambda --> ouput  (una cosa che serve per calcolare la forza fermionica successiva)
									 __restrict const su3_soa   * const U, // gauge configuration --> input
									 __restrict su3_soa   * const S,  // in input it is Sigma prime (input: previous step fermforce);in output it is Sigma
									 __restrict const tamat_soa * const QA, // Cayley hamilton Qs --> input (rho*ta(staples))
									 __restrict su3_soa   * const TMP, // parking variable
									 const int istopo // istopo = {0,1} -> rho = {fermrho,toporho}
									 );


#endif
