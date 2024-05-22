


#define L_STRING 512

#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>





typedef struct
{
  int Nx;
  int Ny;
  int Nz;
  int cell_typ; // 0: orthogonal and 1: non-orthogonal
  int natoms;
  int chg_step_jumpX;
  int chg_step_jumpY;
  int chg_step_jumpZ;
  int total_grid_trained;
  int Nx_used;
  int Ny_used;
  int Nz_used;
  int *atom_typ;
  double LatVec[9];
  double Lx;
  double Ly;
  double Lz;
  // double V;
  double dx;
  double dy;
  double dz;
  double *rho_total;
  double *rho_train;
  double *rho_train_procs;
  double *drho_x_train;
  double *drho_y_train;
  double *drho_z_train;
  double *drho_x_procs;
  double *drho_y_procs;
  double *drho_z_procs;
  double *ddrho_x_train;
  double *ddrho_y_train;
  double *ddrho_z_train;
  double *ddrho_x_procs;
  double *ddrho_y_procs;
  double *ddrho_z_procs;
  double *atom_pos;
  double *basis_matrixG;
  
  // char DenFileName[L_STRING];
} DenFile_param_Obj;




typedef struct
{
  // int np_Ndx;
  // int np_Ndy;
  // int np_Ndz;
  int node_begin_x;
  int node_end_x;
  int node_begin_y;
  int node_end_y;
  int node_begin_z;
  int node_end_z;
  int node_num_x;
  int node_num_y;
  int node_num_z;
  int coords_cart[3];
} DenFile_paral_Obj;



// typedef struct
// {
//   double *rho;
//   double *atom_pos;
// } DenFile_rho_atompos_Obj;


typedef struct
{
  int isC11;
  int isC22;
  int isC23;
  int isC33;
  int isC34;
  int isgradient;
  int ishessian;
  int numC11_coeffs;
  int numC22_coeffs;
  int numC23_coeffs;
  int numC33_coeffs;
  int numC34_coeffs;
  int ischeb;
  int numG_C11;
  int numG_C22;
  int numG_C33;
  int poly_order_C11;
  int poly_order_C22;
  int poly_order_C33;
  int num_bases_C11;
  int num_bases_C22;
  int num_bases_C33;
  int ifwrite;
  int start_idx_den_file;
  int end_idx_den_file;
  long int total_grid_points_train;
  long int total_grid_points_train_x;
  long int total_grid_points_train_y;
  long int total_grid_points_train_z;
  int num_files;
  int nelem;
  int num_coeff;
  int num_DenFiles;
  int num_processors;
  int iftraining;
  int ifpredict;
  int fname_len;
  int *NDx;
  int *NDy;
  int *NDz;
  int *natoms;
  int *Nx_used;
  int *Ny_used;
  int *Nz_used;
  int count1;
  int count2;
  int count3;
  int *grid_trained_per_procs;
  int *node_begin_x;
  int *node_end_x;
  int *node_begin_y;
  int *node_end_y;
  int *node_begin_z;
  int *node_end_z;
  int *node_num_x;
  int *node_num_y;
  int *node_num_z;
  int *chg_jump_x;
  int *chg_jump_y;
  int *chg_jump_z;
  int *I_idx;
  int *J_idx;
  int *K_idx;
  int *file_idx;
  int *der_idx;
  long int grid_points_procs;
  double chg_step_jump_dist;
  double lambda;
  double *Amat;
  double *Amat_1D;
  double *Amat_1D_root;
  double *Amat_1D_root_sorted;
  int *Amat_pos;
  int *Amat_pos_root;
  double **Amat_domain;
  double *bvec;
  double *bvec_domain;
  double *bvec_domain_rho_only;
  double *bvec_domainx;
  double *bvec_domainy;
  double *bvec_domainz;
  double *bvec_domainxx;
  double *bvec_domainyy;
  double *bvec_domainzz;
  double *bvec_root;
  int *bvec_pos;
  int *bvec_pos_root;
  double *bvec_root_sorted;
  double *alpha_C11;
  double *beta_C11;
  double *alpha_C22;
  double *beta_C22;
  double *alpha_C33;
  double *beta_C33;
  double **Gaussian_Width_C11;
  double **Gaussian_Width_C22;
  double **Gaussian_Width_C33;
  double *fit_coeffs;
  double Rcut;
  double time_start;
  double time_end;
  char den_file_name[L_STRING];
  char coeff_file_name[L_STRING];
  char input_filename[L_STRING];
  char ExecRoot[L_STRING];

} main_Obj;

typedef struct
{
  int isC11;
  int isC22;
  int isC23;
  int isC33;
  int isC34;
  int isgradient;
  int ishessian;
  int ischeb;
  int numG_C11;
  int numG_C22;
  int numG_C33;
  int poly_order_C11;
  int poly_order_C22;
  int poly_order_C33;
  int ifwrite;
  int start_idx_den_file;
  int end_idx_den_file;
  int nelem;
  int iftraining;
  int ifpredict;
  double chg_step_jump_dist;
  double lambda;
  double Rcut;
  char den_file_name[L_STRING];
  char coeff_file_name[L_STRING];
  char ExecRoot[L_STRING];
  char filename[L_STRING];
} InputFile_Obj;


#endif