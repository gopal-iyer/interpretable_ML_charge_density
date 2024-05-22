
#ifndef FITTING_H
#define FITTING_H

#include "STRUCTURES.h"
#include "readfiles.h"
#include "initialization.h"


void fitting(main_Obj *pMAIN_Obj);

void Amat_Calc(main_Obj *pMAIN_Obj);

void Setum_Comms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *fparal_Obj, int file_rank);

void Den_data_gather(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_rank);

void Truncate_data_gather(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_rank);

void calc_domain_grids_total(main_Obj *pMAIN_Obj);

int find_chem_idx(int atom_idx, DenFile_param_Obj *file_param_Obj, main_Obj *pMAIN_Obj);

void Calc_C11_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index);

void print_Amat(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj);

void Gather_Amat(main_Obj *pMAIN_Obj);

void print_Amat_distributed(main_Obj *pMAIN_Obj);

void Gather_Amat_distributed(main_Obj *pMAIN_Obj);

void sort_Amat(main_Obj *pMAIN_Obj);

void solve_linear_system(main_Obj *pMAIN_Obj);

void Calc_C22_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index);

void Calc_C33_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index);
void free_memory(DenFile_param_Obj *file_param_Obj);

#endif 