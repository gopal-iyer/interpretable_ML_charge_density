#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include "STRUCTURES.h"
#include "readfiles.h"
// #include "fitting.h"


void Initialize(main_Obj *fmain_common_Obj, int argc, char *argv[]);
void Input_MPI_create(MPI_Datatype *fINPUT_OBJ_MPI);
void check_input(InputFile_Obj *fInput_Obj, int argc, char *argv[]);
void Copy_Input_to_main_Obj(InputFile_Obj *fInput_Obj, main_Obj *ffmain_common_Obj);
void Gaussian_width_initialize(main_Obj *ffmain_common_Obj);
void Grid_pts_calc(main_Obj *pMAIN_Obj);

#endif 