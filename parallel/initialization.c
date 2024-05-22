//  Initializes the structures
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
#include "fitting.h"
#include "initialization.h"
#include "common.h"

#define N_Inputfile 27
#define L_STRING 512

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


void Initialize(main_Obj *fmain_common_Obj, int argc, char *argv[]){

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	InputFile_Obj Input_Obj;
	MPI_Datatype INPUT_OBJ_MPI;
	MPI_Request req;
	#ifdef DEBUG
    if (rank==0) printf("Creating the input broadcast object \n");
  #endif
	Input_MPI_create(&INPUT_OBJ_MPI);

	
	// MPI_Barrier(MPI_COMM_WORLD);


	if (rank==0) {

		
		check_input(&Input_Obj, argc, argv);  // Reads the input file name and the directory where the executable is located


		read_input(&Input_Obj, fmain_common_Obj);	
		// printf("rank %d, alpha :(%1f,%1f)\n",rank,fmain_common_Obj->beta[0],fmain_common_Obj->beta[1]);

		#ifdef DEBUG
      printf("Broadcasting the input object\n");
    #endif
		MPI_Bcast(&Input_Obj, 1, INPUT_OBJ_MPI, 0, MPI_COMM_WORLD);
		#ifdef DEBUG
      		printf("Broadcasting alpha\n");
   		#endif
		MPI_Bcast(fmain_common_Obj->alpha_C11, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->alpha_C22, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->alpha_C33, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		#ifdef DEBUG
         	printf("Broadcasting beta\n");
        #endif
		MPI_Bcast(fmain_common_Obj->beta_C11, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->beta_C22, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->beta_C33, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	} else {	
		MPI_Bcast(&Input_Obj, 1, INPUT_OBJ_MPI, 0, MPI_COMM_WORLD);
		fmain_common_Obj->alpha_C11 = (double *)malloc(Input_Obj.nelem * sizeof(double));
    	fmain_common_Obj->beta_C11 = (double *)malloc(Input_Obj.nelem * sizeof(double));
    	fmain_common_Obj->alpha_C22 = (double *)malloc(Input_Obj.nelem * sizeof(double));
    	fmain_common_Obj->beta_C22 = (double *)malloc(Input_Obj.nelem * sizeof(double));
    	fmain_common_Obj->alpha_C33 = (double *)malloc(Input_Obj.nelem * sizeof(double));
    	fmain_common_Obj->beta_C33 = (double *)malloc(Input_Obj.nelem * sizeof(double));
		MPI_Bcast(fmain_common_Obj->alpha_C11, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->alpha_C22, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->alpha_C33, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->beta_C11, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->beta_C22, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(fmain_common_Obj->beta_C33, Input_Obj.nelem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	}
	// int err =MPI_Barrier(MPI_COMM_WORLD);
	#ifdef DEBUG
    if (rank==0) printf("Free the INPUT Obj \n");
  #endif
	MPI_Type_free(&INPUT_OBJ_MPI);

	#ifdef DEBUG
    if (rank==0) printf("Copy the Input Obj to Main Obj \n");
  #endif
	Copy_Input_to_main_Obj(&Input_Obj, fmain_common_Obj);	

	#ifdef DEBUG
    if (rank==0) printf("Initialize the Gaussian width\n");
  #endif
	Gaussian_width_initialize(fmain_common_Obj);

	#ifdef DEBUG
    if (rank==0) printf("Allocating memory for chg_jump_x, chg_jump_y and chg_jump_z\n");
  #endif
	fmain_common_Obj->chg_jump_x = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->chg_jump_y = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->chg_jump_z = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));

	#ifdef DEBUG
    if (rank==0) printf("Calculating the grid points in training\n");
  #endif
	Grid_pts_calc(fmain_common_Obj);

	#ifdef DEBUG
    if (rank==0) printf("Allocating memory for node_begin_xyz, node_end_xyz\n");
  #endif
	fmain_common_Obj->node_begin_x = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_end_x = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_begin_y = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_end_y = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_begin_z = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_end_z = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_num_x = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_num_y = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));
	fmain_common_Obj->node_num_z = (int *) malloc(fmain_common_Obj->num_files * sizeof(int));

	// printf("coming here \n");
}



void Input_MPI_create(MPI_Datatype *fINPUT_OBJ_MPI){
	InputFile_Obj Input_Obj_temp;
	MPI_Datatype InputFile_vars[N_Inputfile] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
											    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
											    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, 
											    MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,   
											    MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_CHAR, MPI_CHAR,
											    MPI_CHAR, MPI_CHAR};

	int blens[N_Inputfile] = {1, 1, 1, 1, 1,
							  1, 1, 1, 1, 1, 
							  1, 1, 1, 1, 1,   
							  1, 1, 1, 1, 1,
							  1, 1, 1, L_STRING, L_STRING,    
							  L_STRING, L_STRING};

	MPI_Aint addr[N_Inputfile],disps[N_Inputfile], base;
	int i = 0;

	MPI_Get_address(&Input_Obj_temp, &base);
	MPI_Get_address(&Input_Obj_temp.isC11, addr + i++);
	MPI_Get_address(&Input_Obj_temp.isC22, addr + i++);
	MPI_Get_address(&Input_Obj_temp.isC23, addr + i++);
	MPI_Get_address(&Input_Obj_temp.isC33, addr + i++);
	MPI_Get_address(&Input_Obj_temp.isC34, addr + i++);
	MPI_Get_address(&Input_Obj_temp.isgradient, addr + i++);
	MPI_Get_address(&Input_Obj_temp.ishessian, addr + i++);
	MPI_Get_address(&Input_Obj_temp.ischeb, addr + i++);
	MPI_Get_address(&Input_Obj_temp.numG_C11, addr + i++);
	MPI_Get_address(&Input_Obj_temp.numG_C22, addr + i++);
	MPI_Get_address(&Input_Obj_temp.numG_C33, addr + i++);
	MPI_Get_address(&Input_Obj_temp.poly_order_C11, addr + i++);
	MPI_Get_address(&Input_Obj_temp.poly_order_C22, addr + i++);
	MPI_Get_address(&Input_Obj_temp.poly_order_C33, addr + i++);
	MPI_Get_address(&Input_Obj_temp.ifwrite, addr + i++);
	MPI_Get_address(&Input_Obj_temp.start_idx_den_file, addr + i++);
	MPI_Get_address(&Input_Obj_temp.end_idx_den_file, addr + i++);
	MPI_Get_address(&Input_Obj_temp.nelem, addr + i++);
	MPI_Get_address(&Input_Obj_temp.iftraining, addr + i++);
	MPI_Get_address(&Input_Obj_temp.ifpredict, addr + i++);
	MPI_Get_address(&Input_Obj_temp.chg_step_jump_dist, addr + i++);
	MPI_Get_address(&Input_Obj_temp.lambda, addr + i++);
	MPI_Get_address(&Input_Obj_temp.Rcut, addr + i++);
	MPI_Get_address(&Input_Obj_temp.den_file_name, addr + i++);
	MPI_Get_address(&Input_Obj_temp.coeff_file_name, addr + i++);
	MPI_Get_address(&Input_Obj_temp.ExecRoot, addr + i++);
	MPI_Get_address(&Input_Obj_temp.filename, addr + i++);

	for (i = 0; i < N_Inputfile; i++) {
        disps[i] = addr[i] - base;
    }

    MPI_Type_create_struct(N_Inputfile, blens, disps, InputFile_vars, fINPUT_OBJ_MPI);
    MPI_Type_commit(fINPUT_OBJ_MPI);

}
// what if the datatype is of memory type or pointer type

void check_input(InputFile_Obj *fInput_Obj, int argc, char *argv[]){
	int i, rootfile_len;
	char name_flag = 'N';
	char *libbase = argv[0];
	char *pch;
	pch = strrchr(libbase,'/'); // find last occurrence of '/'
	rootfile_len = strlen(libbase) - strlen(pch);
	char rootfile[rootfile_len];

	for (int i=0;i<rootfile_len;i++){
		rootfile[i] = *(libbase+i);
	}

	memcpy(fInput_Obj->ExecRoot, rootfile, strlen(rootfile)+1);

    if (argc != 2) {
    	name_flag = 'Y';
    } else {
    	memcpy(fInput_Obj->filename, argv[1], strlen(argv[1])+1);
    }

    if (name_flag == 'Y') {
        exit(EXIT_FAILURE);
    }

}

void Copy_Input_to_main_Obj(InputFile_Obj *fInput_Obj, main_Obj *ffmain_common_Obj){
	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	ffmain_common_Obj->isC11 = fInput_Obj->isC11;
	ffmain_common_Obj->isC22 = fInput_Obj->isC22;
	ffmain_common_Obj->isC23 = fInput_Obj->isC23;
	ffmain_common_Obj->isC33 = fInput_Obj->isC33;
	ffmain_common_Obj->isC34 = fInput_Obj->isC34;
	ffmain_common_Obj->isgradient = fInput_Obj->isgradient;
	ffmain_common_Obj->ishessian = fInput_Obj->ishessian;
	ffmain_common_Obj->numG_C11 = fInput_Obj->numG_C11;
	ffmain_common_Obj->numG_C22 = fInput_Obj->numG_C22;
	ffmain_common_Obj->ischeb = fInput_Obj->ischeb;
	ffmain_common_Obj->numG_C33 = fInput_Obj->numG_C33;
	ffmain_common_Obj->poly_order_C11 = fInput_Obj->poly_order_C11;
	ffmain_common_Obj->poly_order_C22 = fInput_Obj->poly_order_C22;
	ffmain_common_Obj->poly_order_C33 = fInput_Obj->poly_order_C33;
	ffmain_common_Obj->ifwrite = fInput_Obj->ifwrite;
	ffmain_common_Obj->start_idx_den_file = fInput_Obj->start_idx_den_file;
	ffmain_common_Obj->end_idx_den_file = fInput_Obj->end_idx_den_file;
	ffmain_common_Obj->nelem = fInput_Obj->nelem;
	ffmain_common_Obj->iftraining = fInput_Obj->iftraining;
	ffmain_common_Obj->ifpredict = fInput_Obj->ifpredict;
	ffmain_common_Obj->chg_step_jump_dist = fInput_Obj->chg_step_jump_dist;
	ffmain_common_Obj->lambda = fInput_Obj->lambda;
	ffmain_common_Obj->Rcut = fInput_Obj->Rcut;
	strncpy(ffmain_common_Obj->den_file_name, fInput_Obj->den_file_name, sizeof(ffmain_common_Obj->den_file_name));
	strncpy(ffmain_common_Obj->coeff_file_name, fInput_Obj->coeff_file_name, sizeof(ffmain_common_Obj->coeff_file_name));
	strncpy(ffmain_common_Obj->ExecRoot, fInput_Obj->ExecRoot, sizeof(ffmain_common_Obj->ExecRoot));
	strncpy(ffmain_common_Obj->input_filename, fInput_Obj->filename, sizeof(ffmain_common_Obj->input_filename));
	//ffmain_common_Obj->den_file_name = fInput_Obj->den_file_name;
	// ffmain_common_Obj->coeff_file_name = fInput_Obj->coeff_file_name;
	// ffmain_common_Obj->ExecRoot = fInput_Obj->ExecRoot;
	// ffmain_common_Obj->filename = fInput_Obj->filename;
}

void Gaussian_width_initialize(main_Obj *ffmain_common_Obj) {
	int num_Gaussians_c22,num_Gaussians_c23, num_pairs, num_Gaussians_c33, num_Gaussians_c34, num_triplets;
	int nproc, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (1){
		ffmain_common_Obj->Gaussian_Width_C11 = (double **) malloc (ffmain_common_Obj->numG_C11 * sizeof ( double *));
		ffmain_common_Obj->Gaussian_Width_C22 = (double **) malloc (ffmain_common_Obj->numG_C22 * sizeof ( double *));
		ffmain_common_Obj->Gaussian_Width_C33 = (double **) malloc (ffmain_common_Obj->numG_C33 * sizeof ( double *));
		for (int k = 0; k < ffmain_common_Obj->numG_C11; k++)
	    	ffmain_common_Obj->Gaussian_Width_C11[k] = (double *) malloc ( ffmain_common_Obj->nelem * sizeof ( double ));
	    for (int k = 0; k < ffmain_common_Obj->numG_C22; k++)
	    	ffmain_common_Obj->Gaussian_Width_C22[k] = (double *) malloc ( ffmain_common_Obj->nelem * sizeof ( double ));
	    for (int k = 0; k < ffmain_common_Obj->numG_C33; k++)
	    	ffmain_common_Obj->Gaussian_Width_C33[k] = (double *) malloc ( ffmain_common_Obj->nelem * sizeof ( double ));
	    #ifdef DEBUG
	      if (rank==0) printf("Gaussian width:\n");
	      if (rank==0) printf("-----------------------------------\n");
	    #endif
	    for (int j = 0; j < ffmain_common_Obj->nelem; j++){
	    	#ifdef DEBUG
	        if (rank==0) printf("Element: %d\nWidth: ",j);
	      #endif
	    	for (int i = 0; i < ffmain_common_Obj->numG_C11; i++){
	    		double var1 = pow(ffmain_common_Obj->beta_C11[j], i);
				  ffmain_common_Obj->Gaussian_Width_C11[i][j] = var1 * ffmain_common_Obj->alpha_C11[j];
				  #ifdef DEBUG
	        	if (rank==0) printf("%1f ",ffmain_common_Obj->Gaussian_Width_C11[i][j]);
	      	#endif
	    	}  
	    	#ifdef DEBUG
	        if (rank==0) printf("\n");
	      #endif
	    }
	    for (int j = 0; j < ffmain_common_Obj->nelem; j++){
	    	#ifdef DEBUG
	        if (rank==0) printf("Element: %d\nWidth: ",j);
	      #endif
	    	for (int i = 0; i < ffmain_common_Obj->numG_C22; i++){
	    		double var1 = pow(ffmain_common_Obj->beta_C22[j], i);
				  ffmain_common_Obj->Gaussian_Width_C22[i][j] = var1 * ffmain_common_Obj->alpha_C22[j];
				  #ifdef DEBUG
	        	if (rank==0) printf("%1f ",ffmain_common_Obj->Gaussian_Width_C22[i][j]);
	      	#endif
	    	}  
	    	#ifdef DEBUG
	        if (rank==0) printf("\n");
	      #endif
	    }
	    for (int j = 0; j < ffmain_common_Obj->nelem; j++){
	    	#ifdef DEBUG
	        if (rank==0) printf("Element: %d\nWidth: ",j);
	      #endif
	    	for (int i = 0; i < ffmain_common_Obj->numG_C33; i++){
	    		double var1 = pow(ffmain_common_Obj->beta_C33[j], i);
				  ffmain_common_Obj->Gaussian_Width_C33[i][j] = var1 * ffmain_common_Obj->alpha_C33[j];
				  #ifdef DEBUG
	        	if (rank==0) printf("%1f ",ffmain_common_Obj->Gaussian_Width_C33[i][j]);
	      	#endif
	    	}  
	    	#ifdef DEBUG
	        if (rank==0) printf("\n");
	      #endif
	    }
	    if (rank==0) printf("-----------------------------------\n");
	}
		
    	if (ffmain_common_Obj->ischeb!=1){
    		ffmain_common_Obj->num_bases_C11 = ffmain_common_Obj->numG_C11 *(1+ffmain_common_Obj->poly_order_C11);
    		ffmain_common_Obj->num_bases_C22 = ffmain_common_Obj->numG_C22 *(1+ffmain_common_Obj->poly_order_C22);
    		ffmain_common_Obj->num_bases_C33 = ffmain_common_Obj->numG_C33 *(1+ffmain_common_Obj->poly_order_C33);
    	} else {
    		ffmain_common_Obj->num_bases_C11 = ffmain_common_Obj->poly_order_C11;
    		ffmain_common_Obj->num_bases_C22 = ffmain_common_Obj->poly_order_C22;
    		ffmain_common_Obj->num_bases_C33 = ffmain_common_Obj->poly_order_C33;
    	}
    	
      ffmain_common_Obj->numC11_coeffs = ffmain_common_Obj->nelem * ffmain_common_Obj->num_bases_C11;
      #ifdef DEBUG
      	if (rank==0) printf("Number of C11 coeff :%d\n",ffmain_common_Obj->numC11_coeffs);
    	#endif
      if (ffmain_common_Obj->isC22 == 0) {
      	ffmain_common_Obj->numC22_coeffs = 0;
      } else {
      	num_Gaussians_c22 = (ffmain_common_Obj->num_bases_C22 * (ffmain_common_Obj->num_bases_C22 + 1))/2;
      	num_pairs = (ffmain_common_Obj->nelem * (ffmain_common_Obj->nelem + 1))/2;
      	ffmain_common_Obj->numC22_coeffs = num_Gaussians_c22 * num_pairs;;
      }
      #ifdef DEBUG
      	if (rank==0) printf("Number of C22 coeff :%d\n",ffmain_common_Obj->numC22_coeffs);
    	#endif

      if (ffmain_common_Obj->isC23 == 0) {
      	ffmain_common_Obj->numC23_coeffs = 0;
      } else {
      	num_Gaussians_c23 = (ffmain_common_Obj->num_bases_C22 * (ffmain_common_Obj->num_bases_C22 + 1))/2;
      	num_pairs = (ffmain_common_Obj->nelem * (ffmain_common_Obj->nelem + 1))/2;
      	ffmain_common_Obj->numC23_coeffs = num_Gaussians_c23 * num_pairs;;
      }
      #ifdef DEBUG
      	if (rank==0) printf("Number of C23 coeff :%d\n",ffmain_common_Obj->numC23_coeffs);
    	#endif

      if (ffmain_common_Obj->isC33 == 0) {
      	ffmain_common_Obj->numC33_coeffs = 0;
      } else {
      	num_Gaussians_c33 = (ffmain_common_Obj->num_bases_C33*(ffmain_common_Obj->num_bases_C33 + 1)*(ffmain_common_Obj->num_bases_C33 + 2))/6;
      	num_triplets = (ffmain_common_Obj->nelem * (ffmain_common_Obj->nelem + 1) * (ffmain_common_Obj->nelem + 2))/6;
      	ffmain_common_Obj->numC33_coeffs = num_Gaussians_c33 * num_triplets;
      }
      #ifdef DEBUG
      	if (rank==0) printf("Number of C33 coeff :%d\n",ffmain_common_Obj->numC33_coeffs);
    	#endif

      if (ffmain_common_Obj->isC34 == 0) {
      	ffmain_common_Obj->numC34_coeffs = 0;
      } else {
      	num_Gaussians_c34 = (ffmain_common_Obj->num_bases_C33*(ffmain_common_Obj->num_bases_C33 + 1)*(ffmain_common_Obj->num_bases_C33 + 2))/6;
      	num_triplets = (ffmain_common_Obj->nelem * (ffmain_common_Obj->nelem + 1) * (ffmain_common_Obj->nelem + 2))/6;
      	ffmain_common_Obj->numC34_coeffs = num_Gaussians_c34 * num_triplets;
      }
      #ifdef DEBUG
      	if (rank==0) printf("Number of C34 coeff :%d\n",ffmain_common_Obj->numC34_coeffs);
    	#endif


      ffmain_common_Obj->num_coeff = 1 + ffmain_common_Obj->numC11_coeffs +
      								 ffmain_common_Obj->numC22_coeffs + 
      								 ffmain_common_Obj->numC23_coeffs +
      								 + ffmain_common_Obj->numC33_coeffs + ffmain_common_Obj->numC34_coeffs;
      #ifdef DEBUG
      	if (rank==0) printf("Total Number of coeff :%d\n",ffmain_common_Obj->num_coeff);
    	#endif
      ffmain_common_Obj->num_files = ffmain_common_Obj->end_idx_den_file - ffmain_common_Obj->start_idx_den_file +1;
      #ifdef DEBUG
      	if (rank==0) printf("Total Number of files in training :%d\n",ffmain_common_Obj->num_files);
    	#endif
}

void Grid_pts_calc(main_Obj *pMAIN_Obj) {
	int rank, num_files, num_elements, num_atoms;
	char buffer[L_STRING], line[L_STRING];
	int  *natoms_per_species = NULL;
	int offset;
	int i, j;
	char *p;
	long int total_grid=0, total_grid_trained=0;
	double Hmat[3][3], L_x, L_y, L_z, mesh_x, mesh_y, mesh_z;
	int grid_nx_train, grid_ny_train, grid_nz_train;
	long int total_grid_trained_x=0, total_grid_trained_y=0, total_grid_trained_z=0;
	char a1[30], a2[30], a3[30];
	int grid_nx, grid_ny, grid_nz, tot_grd_size;
	int step_jump_x, step_jump_y, step_jump_z;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int fname_len = 0;
	FILE *fp;

	#ifdef DEBUG
     if (rank==0) printf("Allocating memory for pMAIN_Obj->Nx_used of size num_files\n");
  #endif

	pMAIN_Obj->Nx_used = (int *)malloc(pMAIN_Obj->num_files * sizeof(int));  // look while debug
	pMAIN_Obj->Ny_used = (int *)malloc(pMAIN_Obj->num_files * sizeof(int));  // look while debug
	pMAIN_Obj->Nz_used = (int *)malloc(pMAIN_Obj->num_files * sizeof(int));  // look while debug
	pMAIN_Obj->natoms = (int *)malloc(pMAIN_Obj->num_files * sizeof(int));  // look while debug

	
	num_files = pMAIN_Obj->num_files;
	char *fname = pMAIN_Obj->den_file_name;

	
	for (int fl=0; fl < L_STRING; fl++) {
		if (pMAIN_Obj->den_file_name[fl] != '\0'){
			fname_len = fname_len+1;
		}
	}
	pMAIN_Obj->fname_len = fname_len;


	if (rank==0) {		
		for (i=0; i<num_files; i++) {
			for (int fl=0; fl < L_STRING; fl++) {
				if (fl < fname_len){
					// do nothing
				} else {
					fname[fl] = '\0';
				}
			}
			
			snprintf(buffer, 10,"%d",i+pMAIN_Obj->start_idx_den_file);
			strcat(fname,buffer);
			// printf("fname is %s\n",fname[i]);
			#ifdef DEBUG
     		printf("Reading density file: %s\n",fname);
  			#endif
			fp = fopen(fname,"r");

			if (fp == NULL) {
				fprintf(stderr,"CHGCAR file named '%s' cannot be opened\n",fname);
				exit(-1);
			}

			fgets (line, sizeof (line), fp);
			fgets (line, sizeof (line), fp);


			for (j = 0; j < 3; j++){
				fgets (line, sizeof (line), fp);
				sscanf (line, "%s%s%s", a1, a2, a3);
				Hmat[j][0] = atof (a1);
     		    Hmat[j][1] = atof (a2);
      			Hmat[j][2] = atof (a3);
			}

			fgets (line, sizeof (line), fp);

			p = line;
  			num_elements = 0;
  			while (sscanf(p, "%s%n", a1, &offset) == 1) {
      			p += offset;
      			num_elements++;
    		}

    		if (num_elements!=pMAIN_Obj->nelem){
    			fprintf(stderr,"Error: number of elements in the CHGCAR file '%s' is not the same as specified in input file\n",fname);
    			exit(-1);
    		}
    		// assert(num_elements==pMAIN_Obj->nelem);

    		fgets (line, sizeof (line), fp);

  			p = line;
  			natoms_per_species = (int *) malloc (num_elements * sizeof(int));
  			num_atoms = 0;
  			int jj = 0;
  			while (sscanf(p, "%s%n", a1, &offset) == 1) {
      			natoms_per_species[jj++] = atoi(a1);
      			p += offset;
      			num_atoms += atoi(a1);
    		}

    	    fgets (line, sizeof (line), fp);

		  for (int ii = 0; ii < num_atoms; ii++)
		    {
		      fgets (line, sizeof (line), fp);  /* do nothing. These lines contain atomic positions */
		      //sscanf(line, "%s%s%s", a1, a2, a3);\
		      // pos_chg1->pos[3*i+0] = atof(a1);
		      // pos_chg1->pos[3*i+1] = atof(a2);
		      // pos_chg1->pos[3*i+2] = atof(a3);
		    }
		  
		  fgets (line, sizeof (line), fp);  /* do nothing. This line is empty */
		  // printf("%s", line);
		  
		  // -------- GET CHGCAR GRID SIZE --------
		  fgets (line, sizeof (line), fp);
		  sscanf (line, "%s%s%s", a1, a2, a3);

		  fclose (fp);

		  grid_nx = atoi(a1);
		  grid_ny = atoi(a2);
		  grid_nz = atoi(a3);


		  // printf("LOOPC1 %s %s %s\n", a1, a2, a3);		  
		  tot_grd_size = grid_nx*grid_ny;
		  tot_grd_size *= grid_nz;          // long integers 
		  total_grid = total_grid + tot_grd_size;

		  L_x =  sqrt(Hmat[0][0]*Hmat[0][0] + Hmat[0][1]*Hmat[0][1] + Hmat[0][2]*Hmat[0][2]); // in Angstrom
		  L_y =  sqrt(Hmat[1][0]*Hmat[1][0] + Hmat[1][1]*Hmat[1][1] + Hmat[1][2]*Hmat[1][2]); // in Angstrom
		  L_z =  sqrt(Hmat[2][0]*Hmat[2][0] + Hmat[2][1]*Hmat[2][1] + Hmat[2][2]*Hmat[2][2]); // in Angstrom

		  mesh_x = L_x/grid_nx;
		  mesh_y = L_y/grid_ny;
		  mesh_z = L_z/grid_nz;

		  


		  step_jump_x =  pMAIN_Obj->chg_step_jump_dist / mesh_x;
		  step_jump_y =  pMAIN_Obj->chg_step_jump_dist / mesh_y;
		  step_jump_z =  pMAIN_Obj->chg_step_jump_dist / mesh_z;

		  step_jump_x = MAX(1,step_jump_x);
		  step_jump_y = MAX(1,step_jump_y);
		  step_jump_z = MAX(1,step_jump_z);

		  
		  pMAIN_Obj->chg_jump_x[i] = step_jump_x;
		  pMAIN_Obj->chg_jump_y[i] = step_jump_y;
		  pMAIN_Obj->chg_jump_z[i] = step_jump_z;
		 
		  grid_nx_train = (int) (grid_nx - 1)/step_jump_x + 1;
		  grid_ny_train = (int) (grid_ny - 1)/step_jump_y + 1;
		  grid_nz_train = (int) (grid_nz - 1)/step_jump_z + 1;

		  total_grid_trained_x = total_grid_trained_x + grid_nx_train;
		  total_grid_trained_y = total_grid_trained_y + grid_ny_train;
		  total_grid_trained_z = total_grid_trained_z + grid_nz_train;

		  total_grid_trained = total_grid_trained + grid_nx_train * grid_ny_train * grid_nz_train;

		  pMAIN_Obj->Nx_used[i] = grid_nx_train;
		  pMAIN_Obj->Ny_used[i] = grid_ny_train;
		  pMAIN_Obj->Nz_used[i] = grid_nz_train;
		  pMAIN_Obj->natoms[i] = num_atoms; // look while debug
		  #ifdef DEBUG
     		printf("total_grid_trained+= %ld\n",total_grid_trained);
     		printf("pMAIN_Obj->Nx_used[i] %d\n",pMAIN_Obj->Nx_used[i]);
     		printf("pMAIN_Obj->Ny_used[i] %d\n",pMAIN_Obj->Ny_used[i]);
     		printf("pMAIN_Obj->Nz_used[i] %d\n",pMAIN_Obj->Nz_used[i]);
     		printf("pMAIN_Obj->natoms[i] %d\n",pMAIN_Obj->natoms[i]);
     		printf("pMAIN_Obj->chg_jump_x[i] %d\n",pMAIN_Obj->chg_jump_x[i]);
     		printf("pMAIN_Obj->chg_jump_y[i] %d\n",pMAIN_Obj->chg_jump_y[i]);
     		printf("pMAIN_Obj->chg_jump_z[i] %d\n",pMAIN_Obj->chg_jump_z[i]);
  		#endif

		  free(natoms_per_species);
	 
    }
      pMAIN_Obj->total_grid_points_train   = total_grid_trained;
      pMAIN_Obj->total_grid_points_train_x = total_grid_trained_x;
      pMAIN_Obj->total_grid_points_train_y = total_grid_trained_y;
      pMAIN_Obj->total_grid_points_train_z = total_grid_trained_z;
      #ifdef DEBUG
     		printf("pMAIN_Obj->total_grid_points_train %ld\n",pMAIN_Obj->total_grid_points_train);
     		printf("total_grid_points_train_x %ld\n",pMAIN_Obj->total_grid_points_train_x);
     		printf("total_grid_points_train_y %ld\n",pMAIN_Obj->total_grid_points_train_y);
     		printf("total_grid_points_train_z %ld\n",pMAIN_Obj->total_grid_points_train_z);
     		printf("Broadcasting all read values from processor 0\n");
  		#endif
      MPI_Bcast(pMAIN_Obj->Nx_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD);  // look while debug
      MPI_Bcast(pMAIN_Obj->Ny_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->Nz_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->natoms,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->chg_jump_x,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->chg_jump_y,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->chg_jump_z,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 

      MPI_Bcast(&pMAIN_Obj->total_grid_points_train, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
		} else {
	  MPI_Bcast(pMAIN_Obj->Nx_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD);  // look while debug
      MPI_Bcast(pMAIN_Obj->Ny_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->Nz_used, pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->natoms,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(pMAIN_Obj->chg_jump_x,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->chg_jump_y,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD); 
      MPI_Bcast(pMAIN_Obj->chg_jump_z,  pMAIN_Obj->num_files, MPI_INT, 0, MPI_COMM_WORLD);  

	  MPI_Bcast(&pMAIN_Obj->total_grid_points_train, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&pMAIN_Obj->total_grid_points_train_z, 1, MPI_INT, 0, MPI_COMM_WORLD);


		}
		

	}