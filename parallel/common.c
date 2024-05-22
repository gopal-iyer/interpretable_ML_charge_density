//Fitting 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <ctype.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>  
#include <sys/time.h>
#include <sys/types.h>
#include "mkl.h"
// #include "common.c"
// #include "read_chgcar.c"
#include <stdbool.h>

#include "STRUCTURES.h"
#include "readfiles.h"
#include "initialization.h"
#include "fitting.h"
#include "common.h"

double func_Gaussian ( double inv_sigma, double dr)
{
  return (exp ( -inv_sigma * dr * dr ));
}

void func_GTO ( double inv_sigma, double dr, int poly_degree, double func_out[3], int ifchebeshev, double maxdist)
{
  int nproc, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double t0,t1;

  if (ifchebeshev == 0){
    // Function value
  // if (rank==0) t0 = MPI_Wtime();
  func_out[0] = exp ( -inv_sigma * dr * dr ) * pow(dr,poly_degree);
  // if (rank==0) t1 = MPI_Wtime();
  // if (rank==0) printf("time taken %1f\n",t1-t0);

  // first derivative
  if (poly_degree > 0){
    func_out[1] = -2 * inv_sigma * dr * exp ( -inv_sigma * dr * dr ) * pow(dr,poly_degree) +
    exp ( -inv_sigma * dr * dr ) *(poly_degree*pow(dr,poly_degree-1));
  } else {
    func_out[1] = -2 * inv_sigma * dr * exp ( -inv_sigma * dr * dr ) * pow(dr,poly_degree);
  }

  // Second derivative
  if (poly_degree==0){
    func_out[2] = exp ( -inv_sigma * dr * dr ) *(4*inv_sigma*inv_sigma*dr*dr - 2*inv_sigma);
  }
  if (poly_degree==1){
    func_out[2] = exp ( -inv_sigma * dr * dr ) *(4*inv_sigma*inv_sigma*dr*dr*dr-6*inv_sigma*dr);
  }
  if (poly_degree>1){
    func_out[2] = exp ( -inv_sigma * dr * dr ) *(-2*inv_sigma*(poly_degree+2)*pow(dr,poly_degree)+
      4*inv_sigma*inv_sigma*pow(dr,poly_degree+2) + poly_degree*(poly_degree-1)*pow(dr,poly_degree-2));
  }

  // return (exp ( -inv_sigma * dr * dr ) * pow(dr,poly_degree));
  }
  else{
    if (rank==0) printf("wrong function called for chebeshev evaluation\n");
  }
  
}

void func_cheb(double dr, int num_bases, double** cheb_bases_dr, int maxdist, double rcut) 
{
  for (int i = 0; i < num_bases; i++){
    for (int j = 0; j <3; j++){
      cheb_bases_dr[i][j] = 0.0;
    }
  }
  double dr_orig = dr;
  double temp1, temp2, temp3;
  dr = cos(3.14159265359*0.5*dr/maxdist);
  cheb_bases_dr[0][0] = 1.0;
  cheb_bases_dr[1][0] = dr;

  cheb_bases_dr[0][1] = 0.0;
  cheb_bases_dr[1][1] = 1.0;

  cheb_bases_dr[0][2] = 0.0;
  cheb_bases_dr[1][2] = 0.0;

  for (int i = 2; i < num_bases; i++){
    for (int j = 0; j <3; j++){
      if (j==0){
        cheb_bases_dr[i][j] = 2*dr*cheb_bases_dr[i-1][0] - cheb_bases_dr[i-2][0];
      } else if (j==1){
        cheb_bases_dr[i][j] = 2*cheb_bases_dr[i-1][0] + 2*dr*cheb_bases_dr[i-1][1]
                           - cheb_bases_dr[i-2][1];
        cheb_bases_dr[i][j] = cheb_bases_dr[i][j] * (-1*sin(dr)) * (1/maxdist);
        
      } else if (j==2) {
        cheb_bases_dr[i][j] = 4*cheb_bases_dr[i-1][1] + 2*dr*cheb_bases_dr[i-1][2] 
                           - cheb_bases_dr[i-2][2];
        cheb_bases_dr[i][j] = -1.0 *(1/maxdist) * (cheb_bases_dr[i][j-1]*cos(dr)*(1/maxdist) - 
                              sin(dr)*sin(dr)*(1/maxdist)*cheb_bases_dr[i][j]);
      }
    }
  }
  for (int i = 0; i < num_bases; i++){
    temp1 = cheb_bases_dr[i][0];
    temp2 = cheb_bases_dr[i][1];
    temp3 = cheb_bases_dr[i][2];

    cheb_bases_dr[i][0] = temp1 * exp(-4.0 * dr_orig*dr_orig/(rcut*rcut));
    cheb_bases_dr[i][1] = exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp2  +
                              (-4.0*2.0*dr_orig/(rcut*rcut)) *exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) *
                               temp1;
    cheb_bases_dr[i][1] = exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp3  +    
                         (-4.0*2.0*dr_orig/(rcut*rcut)) *exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp2 +
                         (-4.0*2.0*dr_orig/(rcut*rcut)) *exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp2 +
                         pow((-4.0*2.0*dr_orig/(rcut*rcut)), 2) *exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp1 +
                         (-4.0*2.0*1.0/(rcut*rcut)) *exp(-4.0 * dr_orig*dr_orig/(rcut*rcut)) * temp1;
    if (dr_orig > rcut){
      cheb_bases_dr[i][0] = 0.0;
      cheb_bases_dr[i][1] = 0.0;
      cheb_bases_dr[i][2] = 0.0;
    }

  }

}


// double func_chebeshev(int poly_degree, double dr)
// { dr = cos(dr/maxdist);
//   if (0){
//     if (poly_degree == 0){
//       return 1.0;
//     } else if (poly_degree == 1){
//       return dr;
//     } else{
//       double temp = 2*dr*func_chebeshev(poly_degree-1, dr) - func_chebeshev(poly_degree-2, dr);
//       return temp;
//     }
//   } else {

//     if (poly_degree >= 0)
//     {
//       Tn[0] = 1.0;  /* Tn contains chebyshev_poly*smoothing_function */
//       dTn_ds[0] = 0.00;  /* dTn_ds contains only derivative of chebyshev_poly */
//     }
//   }
  

// }


// double func_chebeshev_grad(int poly_degree, double dr)
// {
//   if (poly_degree == 0){
//     return 0.0;
//   } else if (poly_degree == 1){
//     return 1.0;
//   } else{
//     double temp = 2*func_chebeshev(poly_degree-1, dr) + 2*dr*func_chebeshev_grad(poly_degree-1, dr)
//                 - func_chebeshev_grad(poly_degree-2, dr);
//     return temp;
//   }
// }

// double func_chebeshev_hess(int poly_degree, double dr)
// {
//   if (poly_degree == 0){
//     return 0.0;
//   } else if (poly_degree == 1){
//     return 0.0;
//   } else{
//     double temp = 4*func_chebeshev_grad(poly_degree-1, dr) + 2*dr*func_chebeshev_hess(poly_degree-1, dr)
//                 - func_chebeshev_hess(poly_degree-2, dr);
//     return temp;
//   }
// }




double volume_Hmat(double *H)
{
  double detH, invH[3][3], H3d[3][3];

  H3d[0][0] = H[0];
  H3d[0][1] = H[1];
  H3d[0][2] = H[2];
  H3d[1][0] = H[3];
  H3d[1][1] = H[4];
  H3d[1][2] = H[5];
  H3d[2][0] = H[6];
  H3d[2][1] = H[7];
  H3d[2][2] = H[8];

  invH[0][0] = H3d[1][1]*H3d[2][2] - H3d[1][2]*H3d[2][1];
  invH[0][1] = H3d[2][1]*H3d[0][2] - H3d[0][1]*H3d[2][2];
  invH[0][2] = H3d[0][1]*H3d[1][2] - H3d[0][2]*H3d[1][1];

  detH = H3d[0][0]*invH[0][0] + H3d[0][1]*invH[1][0] + H3d[0][2]*invH[2][0];

  return detH;
}

double checkPBC_Shifted (double x)
{
  if (x < -0.5)
    {
      do
  {
    x = x + 1;
  }
      while (x < -0.5); 
    }
  else if (x > 0.5)
    {
      do
  {
    x = x - 1;
  }
      while (x > 0.5); 
    }
  return x; 
}


void V3M3mulV3(double x1[3], double H[9], double x2[3])
{
  double H1[3][3];
  H1[0][0] = H[0];
  H1[0][1] = H[1];
  H1[0][2] = H[2];
  H1[1][0] = H[3];
  H1[1][1] = H[4];
  H1[1][2] = H[5];
  H1[2][0] = H[6];
  H1[2][1] = H[7];
  H1[2][2] = H[8];

  x2[0] = x1[0]*H1[0][0] + x1[1]*H1[1][0] + x1[2]*H1[2][0];
  x2[1] = x1[0]*H1[0][1] + x1[1]*H1[1][1] + x1[2]*H1[2][1];
  x2[2] = x1[0]*H1[0][2] + x1[1]*H1[1][2] + x1[2]*H1[2][2];

}

double V3NormS(double x1[3])
{
  double norm = x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2];
  return sqrt(norm);
}