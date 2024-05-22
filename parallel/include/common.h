#ifndef COMMON_H
#define COMMON_H

#include "STRUCTURES.h"
#include "readfiles.h"
#include "initialization.h"
#include "initialization.h"
#include "fitting.h"

double func_Gaussian ( double inv_sigma, double dr);
void func_GTO ( double inv_sigma, double dr, int poly_degree, double func_out[3], int ifchebeshev, double maxdist);
void func_cheb(double dr, int num_bases, double** cheb_bases_dr, int maxdist, double Rcut);
// double  func_chebeshev(int poly_degree, double dr);
// double  func_chebeshev_grad(int poly_degree, double dr);
// double  func_chebeshev_hess(int poly_degree, double dr);
double volume_Hmat(double *H);
double checkPBC_Shifted (double x);
void V3M3mulV3(double x1[3], double H[9], double x2[3]);
double V3NormS(double x1[3]);
#endif