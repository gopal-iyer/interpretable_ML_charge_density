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
  
#define L_STRING 512
#define rho_tol 0.000001
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
// #define DEBUG TRUE
void fitting(main_Obj *pMAIN_Obj) {
	
	// pMAIN_Obj->bvec = (double *)malloc( pMAIN_Obj->total_grid_points_train * sizeof(double) );
	// pMAIN_Obj->Amat = (double *)malloc( pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff * sizeof(double) );
	int nprocs, rank;
	int rows_Amat;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	// int numD;

	// numD = pMAIN_Obj->num_bases;

	#ifdef DEBUG
      if (rank==0) printf("Calculating domain grids\n");
    #endif
	calc_domain_grids_total(pMAIN_Obj);

	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0)
		rows_Amat = pMAIN_Obj->grid_points_procs;
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0)
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1)
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1)
		rows_Amat = 7*pMAIN_Obj->grid_points_procs;
	#ifdef DEBUG
      if (rank==0) printf("rows_Amat %d\n",rows_Amat);
      if (rank==0) printf("Allocating memory for Amat_domain:  %d*%d\n",rows_Amat,pMAIN_Obj->num_coeff);
      if (rank==0) printf("Allocating memory for pMAIN_Obj->bvec_domain:  %d\n",rows_Amat);
    #endif

	pMAIN_Obj->Amat_domain = (double **)malloc( rows_Amat * sizeof(double *) );
	pMAIN_Obj->bvec_domain = (double *)malloc( rows_Amat * sizeof(double) );

	pMAIN_Obj->bvec_domain_rho_only = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainx = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainy = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainz = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainxx = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainyy = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );
	pMAIN_Obj->bvec_domainzz = (double *)malloc( pMAIN_Obj->grid_points_procs * sizeof(double) );

	#ifdef DEBUG
      if (rank==0) printf("Allocating memory for Amat_1D:  %d\n",rows_Amat *pMAIN_Obj->num_coeff);
      if (rank==0) printf("Allocating memory for Amat_pos:  %d\n",rows_Amat *pMAIN_Obj->num_coeff);
      if (rank==0) printf("Allocating memory for bvec_pos:  %d\n",rows_Amat);
    #endif
	pMAIN_Obj->Amat_1D = (double *)malloc( rows_Amat *pMAIN_Obj->num_coeff* sizeof(double *) );
	pMAIN_Obj->Amat_pos = (int *)malloc( rows_Amat *pMAIN_Obj->num_coeff* sizeof(int *) );
	pMAIN_Obj->bvec_pos = (int *)malloc( rows_Amat * sizeof(int *) );
	#ifdef DEBUG
      if (rank==0) printf("Allocating memory for I_idx, J_idx, K_idx:  %d, %d, %d\n",rows_Amat, rows_Amat, rows_Amat);
      if (rank==0) printf("Allocating memory for file_idx:  %d\n",rows_Amat);
      if (rank==0) printf("Allocating memory for der_idx:  %d\n",rows_Amat);
    #endif
	pMAIN_Obj->I_idx = (int *)malloc( rows_Amat * sizeof(int) );
	pMAIN_Obj->J_idx = (int *)malloc( rows_Amat * sizeof(int) );
	pMAIN_Obj->K_idx = (int *)malloc( rows_Amat * sizeof(int) );
	pMAIN_Obj->file_idx = (int *)malloc( rows_Amat * sizeof(int) );
	pMAIN_Obj->der_idx = (int *)malloc( rows_Amat * sizeof(int) );



	for (int j = 0; j<rows_Amat; j++){
		pMAIN_Obj->Amat_domain[j] = (double *)malloc( pMAIN_Obj->num_coeff * sizeof(double) );
	}


	if (pMAIN_Obj->Amat_domain == NULL)
	{
	  printf("ERROR: Insufficient memory\n");
	}

	if (pMAIN_Obj->bvec_domain == NULL)
	{
	  printf("ERROR: Insufficient memory\n");
	}

	int *recvcounts, *displs, *sendbuf;
	sendbuf = (int *) malloc(1*sizeof(int));
	sendbuf[0] = pMAIN_Obj->grid_points_procs;
	if (rank==0)  pMAIN_Obj->grid_trained_per_procs = (int *) malloc(nprocs*sizeof(int));
	if (rank==0) recvcounts = (int *) malloc(nprocs*sizeof(int));
	if (rank==0) displs = (int *) malloc(nprocs*sizeof(int));
	if (rank==0) {
		for (int pp = 0; pp<nprocs; pp++) {
			displs[pp] = pp;
			recvcounts[pp] = 1;
		}
	}

	
	if (rank==0) MPI_Gatherv( sendbuf, 1, MPI_INT, pMAIN_Obj->grid_trained_per_procs, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank!=0) MPI_Gatherv( sendbuf, 1, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, MPI_COMM_WORLD);

	#ifdef DEBUG
      if (rank==0) printf("grid_trained_per_procs broadcasted to all processors\n");
      if (rank==0) printf("grid_trained_per_procs: ");
      for (int i=0;i<nprocs;i++){
      	if (rank==0) printf("%d",pMAIN_Obj->grid_trained_per_procs[i]);
      }
    #endif

    #ifdef DEBUG
      if (rank==0) printf("\n");
    #endif

	if (rank==0) free(recvcounts);
	if (rank==0) free(displs);
	free(sendbuf);

	// for (int j = 0; j < pMAIN_Obj->grid_points_procs; j++)
	// 	pMAIN_Obj->bvec_domain[j]=0;
	// for (int j = 0; j < pMAIN_Obj->grid_points_procs * pMAIN_Obj->num_coeff; j++)
	// 	pMAIN_Obj->Amat_domain[j]=0;
	#ifdef DEBUG
      if (rank==0) printf("Calculating Amat\n");
    #endif
	Amat_Calc(pMAIN_Obj);

}

void Amat_Calc(main_Obj *pMAIN_Obj) {

	int nproc, rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	DenFile_paral_Obj paral_Obj;
	DenFile_param_Obj file_param_Obj;
	char *fname;
	char buffer[L_STRING], line[L_STRING];
	double Hmat[3][3], L_x, L_y, L_z;
	
	int i, j;
	// printf("came here\n");

	pMAIN_Obj->count1 = 0;
	pMAIN_Obj->count2 = 0;
	pMAIN_Obj->count3 = 0;

	for (i=0; i<pMAIN_Obj->num_files; i++) {
		double t0, t1, t2, t3, t4, t5, t6;
		if (rank==0) t0 = MPI_Wtime();
		#ifdef DEBUG
      	if (rank==0) printf("Setting up communication\n");
    	#endif
		Setum_Comms(pMAIN_Obj, &paral_Obj, i);
		// MPI_Barrier(MPI_COMM_WORLD);
		if (rank==0) t1 = MPI_Wtime();

		if (rank == 0) printf("Setting up communication took %.3f s.\n ", t1 - t0); 
     	Den_data_gather(pMAIN_Obj, &paral_Obj, &file_param_Obj, i);
     	// MPI_Barrier(MPI_COMM_WORLD);
     	if (rank==0) t2 = MPI_Wtime();		
		if (rank == 0) printf("Reading density took %.3f s.\n", t2 - t1); 
    	// MPI_Barrier(MPI_COMM_WORLD);
		Truncate_data_gather(pMAIN_Obj, &paral_Obj, &file_param_Obj, i);
		if (rank==0) t3 = MPI_Wtime();
		if (rank == 0) printf("Communicating density took %.3f s.\n", t3 - t2); 		
    	// MPI_Barrier(MPI_COMM_WORLD);
		Calc_C11_terms(pMAIN_Obj, &paral_Obj, &file_param_Obj, i);
		if (rank==0) t4 = MPI_Wtime();
		if (rank==0) printf("calculation of c11 took %.3f s. for rank %d\n", t4 - t3, rank); 
		// MPI_Barrier(MPI_COMM_WORLD);
		if (pMAIN_Obj->isC22==1) Calc_C22_terms(pMAIN_Obj, &paral_Obj, &file_param_Obj, i);
		if (rank==0) t5 = MPI_Wtime();
		if (rank==0 && pMAIN_Obj->isC22==1) printf("calculation of c22 took %.3f s. for rank %d\n", t5 - t4, rank); 
		if (pMAIN_Obj->isC33==1) Calc_C33_terms(pMAIN_Obj, &paral_Obj, &file_param_Obj, i);
		if (rank==0) t6 = MPI_Wtime();
		if (rank==0 && pMAIN_Obj->isC33) printf("calculation of c33 took %.3f s. for rank %d\n", t6 - t5, rank); 

		free_memory(&file_param_Obj);
		
	}


	// Gather_Amat(pMAIN_Obj);   // if full Amat is desired
	// solve_linear_system(pMAIN_Obj);  // if full Amat is desired

	 print_Amat_distributed(pMAIN_Obj);

	 Gather_Amat_distributed(pMAIN_Obj);


}


void free_memory(DenFile_param_Obj *file_param_Obj) {
	free(file_param_Obj->atom_pos);
	free(file_param_Obj->atom_typ);
	free(file_param_Obj->rho_train_procs);
	free(file_param_Obj->drho_x_procs);
	free(file_param_Obj->drho_y_procs);
	free(file_param_Obj->drho_z_procs);
	free(file_param_Obj->ddrho_x_procs);
	free(file_param_Obj->ddrho_y_procs);
	free(file_param_Obj->ddrho_z_procs);

}

void Setum_Comms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *fparal_Obj, int file_rank){
 // sets up the communicator for a given density file
	int nprocs, rank;
	int periods[3]= {1,1,1}, dims[3];
	int rank_cart;
	int dims_cart[3],periods_cart[3], coords_cart[3];
	int coord_procs_x, coord_procs_y, coord_procs_z;
	int node_begin_x, node_end_x, node_begin_y, node_end_y, node_begin_z, node_end_z;
	int Nx, Ny, Nz, Nd;
	int rem_x, rem_y, rem_z;
	int add_x, add_y, add_z;
	int add_num_x, add_num_y, add_num_z;

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Comm ThreeD_cartesian_comm;

	Nx = pMAIN_Obj->Nx_used[file_rank];
	Ny = pMAIN_Obj->Ny_used[file_rank];
	Nz = pMAIN_Obj->Nz_used[file_rank];
	Nd = Nx+Ny+Nz;

	dims[0] = cbrt(nprocs);
	dims[1] = cbrt(nprocs);
	dims[2] = cbrt(nprocs);
	// MPI_Dims_create(nproc, 3, dims); 
	int err1 = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &ThreeD_cartesian_comm);
	int err2 = MPI_Cart_get(ThreeD_cartesian_comm, 3, dims_cart, periods_cart, coords_cart);

	#ifdef DEBUG	
    	if (err1 != MPI_SUCCESS) printf(" MPI_Cart_create failed from processor %d and file_rank %d\n \n", rank, file_rank);	
    	if (err2 != MPI_SUCCESS) printf(" MPI_Cart_get failed from processor %d and file_rank %d\n \n", rank, file_rank) ;		
	#endif

	fparal_Obj->coords_cart[0]=coords_cart[0];
	fparal_Obj->coords_cart[1]=coords_cart[1];
	fparal_Obj->coords_cart[2]=coords_cart[2];

	coord_procs_x = coords_cart[0];
	coord_procs_y = coords_cart[1];
	coord_procs_z = coords_cart[2];

	rem_x = Nx - ((Nx/dims[0])*dims[0]);
	rem_y = Ny - ((Ny/dims[1])*dims[1]);
	rem_z = Nz - ((Nz/dims[2])*dims[2]);
	add_x = MIN(coord_procs_x, rem_x);
	add_y = MIN(coord_procs_y, rem_y);
	add_z = MIN(coord_procs_z, rem_z);
	add_num_x = MIN(rem_x/(coord_procs_x+1),1);
	add_num_y = MIN(rem_y/(coord_procs_y+1),1);
	add_num_z = MIN(rem_z/(coord_procs_z+1),1);
 // Nodes to be covered by the current prcoessor
	node_begin_x = coord_procs_x*(Nx/dims[0])+add_x;
	node_end_x = node_begin_x+(Nx/dims[0])-1+add_num_x;
	node_begin_y = coord_procs_y*(Ny/dims[1])+add_y;
	node_end_y = node_begin_y+(Ny/dims[1])-1+add_num_y;
	node_begin_z = coord_procs_z*(Nz/dims[2])+add_z;
	node_end_z = node_begin_z+(Nz/dims[2])-1+add_num_z;

	fparal_Obj->node_begin_x = node_begin_x;
	fparal_Obj->node_end_x = node_end_x;
	fparal_Obj->node_begin_y = node_begin_y;
	fparal_Obj->node_end_y = node_end_y;
	fparal_Obj->node_begin_z = node_begin_z;
	fparal_Obj->node_end_z = node_end_z;
	fparal_Obj->node_num_x = node_end_x - node_begin_x + 1;
	fparal_Obj->node_num_y = node_end_y - node_begin_y + 1;
	fparal_Obj->node_num_z = node_end_z - node_begin_z + 1;

	pMAIN_Obj->node_begin_x[file_rank] = node_begin_x;
	pMAIN_Obj->node_end_x[file_rank] = node_end_x;
	pMAIN_Obj->node_begin_y[file_rank] = node_begin_y;
	pMAIN_Obj->node_end_y[file_rank] = node_end_y;
	pMAIN_Obj->node_begin_z[file_rank] = node_begin_z;
	pMAIN_Obj->node_end_z[file_rank] = node_end_z;
	pMAIN_Obj->node_num_x[file_rank] = node_end_x - node_begin_x + 1;
	pMAIN_Obj->node_num_y[file_rank] = node_end_y - node_begin_y + 1;
	pMAIN_Obj->node_num_z[file_rank] = node_end_z - node_begin_z + 1;

	#ifdef DEBUG
    	printf("pMAIN_Obj->node_begin_x,y,z: %d %d %d for rank %d and file %d\n",
    		pMAIN_Obj->node_begin_x[file_rank],pMAIN_Obj->node_begin_y[file_rank],
    		pMAIN_Obj->node_begin_z[file_rank],rank,file_rank);
    	printf("pMAIN_Obj->node_end_x,y,z: %d %d %d for rank %d and file %d\n",
    		pMAIN_Obj->node_end_x[file_rank],pMAIN_Obj->node_end_y[file_rank],
    		pMAIN_Obj->node_end_z[file_rank],rank,file_rank);
    	printf("pMAIN_Obj->node_num_x,y,z: %d %d %d for rank %d and file %d\n",
    		pMAIN_Obj->node_num_x[file_rank],pMAIN_Obj->node_num_y[file_rank],
    		pMAIN_Obj->node_num_z[file_rank],rank,file_rank);
  	#endif
	MPI_Comm_free(&ThreeD_cartesian_comm);
}

void Den_data_gather(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_rank) {
	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


	FILE *fp;
	char buffer[L_STRING], line[L_STRING];
	int  *natoms_per_species = NULL;
	int offset;
	int i, j;
	char *p;
	long int total_grid=0, total_grid_trained=0;
	double Hmat[3][3], L_x, L_y, L_z, mesh_x, mesh_y, mesh_z;
	int grid_nx_train, grid_ny_train, grid_nz_train;
	int total_grid_trained_x=0, total_grid_trained_y=0, total_grid_trained_z=0;
	int num_files = pMAIN_Obj->num_files;
	char a1[30], a2[30], a3[30];
	int num_elements, num_atoms, grid_nx, grid_ny, grid_nz;
	int step_jump_x, step_jump_y, step_jump_z;
	int pos_x_m, pos_x_p, pos_y_m, pos_y_p, pos_z_m, pos_z_p;
	double *grad_weights, *grad_weights2;
  	int FDn=6;
  	int w, ig,jg,kg,igm,jgm,kgm,igp,jgp,kgp;
  	int pos_3d;

	char *fname = pMAIN_Obj->den_file_name;


	if (rank==0) {
		for (int fl=0; fl < L_STRING; fl++) {
			if (fl < pMAIN_Obj->fname_len){
				// do nothing
			} else {
				fname[fl] = '\0';
			}
		}

		snprintf(buffer, 10,"%d",file_rank+pMAIN_Obj->start_idx_den_file);
		strcat(fname,buffer);

		#ifdef DEBUG
    		printf("Reading density file: %s\n",fname);
  		#endif
		fp = fopen(fname,"r");

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


		fgets (line, sizeof (line), fp);
		p = line;
  		natoms_per_species = (int *) malloc (num_elements * sizeof(int));
  		num_atoms = 0;
  		int io=0;
  		while (sscanf(p, "%s%n", a1, &offset) == 1) {
  			natoms_per_species[io++] = atoi(a1);
  			p += offset;
  			num_atoms += atoi(a1);
  		}
  		#ifdef DEBUG
    		printf("Allocating memory for file_param_Obj->atom_pos : %d\n",3*num_atoms);
    		printf("Atom pos : ");
  		#endif
 		file_param_Obj->atom_pos = (double *) malloc (3*num_atoms*sizeof(double));

  		fgets (line, sizeof (line), fp);
  		for (int iat = 0; iat < num_atoms; iat++) {
  			fgets (line, sizeof (line), fp);  /* do nothing. These lines contain atomic positions */
		    sscanf(line, "%s%s%s", a1, a2, a3);\
		    file_param_Obj->atom_pos[3*iat+0] = atof(a1);
		    file_param_Obj->atom_pos[3*iat+1] = atof(a2);
		    file_param_Obj->atom_pos[3*iat+2] = atof(a3);
		    #ifdef DEBUG
	    		printf("%1f %1f %1f",file_param_Obj->atom_pos[3*iat+0],
	    			file_param_Obj->atom_pos[3*iat+1],file_param_Obj->atom_pos[3*iat+2]);
	    		printf("\n");
  			#endif
  		}

  		fgets (line, sizeof (line), fp);
  		fgets (line, sizeof (line), fp);
  		sscanf (line, "%s%s%s", a1, a2, a3);
  		grid_nx = atoi(a1);
  		grid_ny = atoi(a2);
  		grid_nz = atoi(a3);

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

		grid_nx_train = (int) (grid_nx - 1)/step_jump_x + 1;
		grid_ny_train = (int) (grid_ny - 1)/step_jump_y + 1;
		grid_nz_train = (int) (grid_nz - 1)/step_jump_z + 1;

		total_grid_trained = grid_nx_train * grid_ny_train * grid_nz_train;

		file_param_Obj->rho_total = (double *) malloc (grid_nx * grid_ny * grid_nz * sizeof (double));
		file_param_Obj->rho_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->drho_x_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->drho_y_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->drho_z_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_x_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_y_train = (double *) malloc (total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_z_train = (double *) malloc (total_grid_trained * sizeof (double));
		#ifdef DEBUG
    		printf("Allocating memory for file_param_Obj->rho_total : %d\n",grid_nx * grid_ny * grid_nz);
    		printf("Allocating memory for file_param_Obj->rho_train : %ld\n",total_grid_trained);
    		printf("Allocating memory for drho_x,y,x_train, ddrho_x,y,z_train : %ld\n",total_grid_trained);
  		#endif
    	#ifdef DEBUG
    		printf("Initializing file_param_Obj->drho_x,y,z_train and file_param_Obj->ddrho_x,y,z_train to zero\n");	
  		#endif
		for (j=0; j<total_grid_trained;j++){
			file_param_Obj->drho_x_train[j]=0.0;
			file_param_Obj->drho_y_train[j]=0.0;
			file_param_Obj->drho_z_train[j]=0.0;
			file_param_Obj->ddrho_x_train[j]=0.0;
			file_param_Obj->ddrho_y_train[j]=0.0;
			file_param_Obj->ddrho_z_train[j]=0.0;
		}

		#ifdef DEBUG
    		printf("Reading rho from the file into file_param_Obj->rho_total\n");	
  		#endif
		int idx = 0;
  		while (fgets (line, sizeof (line), fp) && !strstr (line, "augmentation")){
  			p = line;
  			while (sscanf(p, "%s%n", a1, &offset) == 1) {
  			 	file_param_Obj->rho_total[idx++] = atof(a1);
  			 	p += offset;
  			 }
  		}
  		
  		#ifdef DEBUG
    		printf("Calculating weights for FD gradient and Laplacian\n");	
  		#endif
    	#ifdef DEBUG
    		printf("Gradient weigths: ");	
  		#endif
  		grad_weights = (double *)malloc(FDn*sizeof(double));
  		grad_weights2 = (double *)malloc((FDn+1)*sizeof(double));
  		for (w=0;w<FDn;w++){
  			grad_weights[w] = pow(-1.0,w+2) * (1.0/(w+1))
  							 * pow(tgamma(FDn),2) *(1/tgamma(FDn-w-1)) *(1/tgamma(FDn+w+1));
  			#ifdef DEBUG
    			printf("%1f ",grad_weights[w]);	
  			#endif
  		}
  		#ifdef DEBUG
    		printf("\n");	
  		#endif
    	#ifdef DEBUG
    		printf("Laplacian weigths: ");	
  		#endif
  		grad_weights2[0] = 0;
  		for (w = 0; w < FDn; w++){
  			grad_weights2[0] = grad_weights2[0] - 1/pow(w+1, 2);
  			
  		}
  		#ifdef DEBUG
    		printf("%1f ",grad_weights2[0]);	
  		#endif
  		for (w = 1; w < FDn+1; w++){
  			grad_weights2[w] = 2*pow(-1.0,w+1) * (1.0/(w*w))
  							 * pow(tgamma(FDn),2) *(1/tgamma(FDn-w)) *(1/tgamma(FDn+w));
  			#ifdef DEBUG
    			printf("%1f ",grad_weights2[w]);	
  			#endif
  		}

  		#ifdef DEBUG
    		printf("\nSave rho_train corrresponding to step_jump into file_param_Obj->rho_train\n");
    		printf("Calculate drho_x,y,z and ddrho_xx,yy,zz\n");
  		#endif
  		idx = 0;   /// try to make it better
  		for (kg = 0; kg < grid_nz; kg=kg+step_jump_z) {
  			for (jg = 0; jg < grid_ny; jg=jg+step_jump_y) {
  				for (ig = 0; ig < grid_nx; ig=ig+step_jump_x) {
  					pos_3d = ig + jg*grid_nx + kg*grid_nx*grid_ny;
  					// double rem_x = ig/step_jump_x; 
  					// double rem_y = jg/step_jump_y;
  					// double rem_z = kg/step_jump_z;

  					// if (rem_x == 0.0 && rem_y == 0.0 && rem_z == 0.0) {
  						file_param_Obj->rho_train[idx]=file_param_Obj->rho_total[pos_3d];

  						file_param_Obj->ddrho_x_train[idx] = file_param_Obj->ddrho_x_train[idx] +
						2*grad_weights2[0]* (1/(mesh_x*mesh_x))*file_param_Obj->rho_total[pos_3d];

						file_param_Obj->ddrho_y_train[idx] = file_param_Obj->ddrho_y_train[idx] +
						2*grad_weights2[0]* (1/(mesh_y*mesh_y))*file_param_Obj->rho_total[pos_3d];

						file_param_Obj->ddrho_z_train[idx] = file_param_Obj->ddrho_z_train[idx] +
						2*grad_weights2[0]* (1/(mesh_z*mesh_z))*file_param_Obj->rho_total[pos_3d];

  						for (w=0;w<FDn;w++){
  							igm = ig-w-1;
  							igp = ig+w+1;
  							jgm = jg-w-1;
  							jgp = jg+w+1;
  							kgm = kg-w-1;
  							kgp = kg+w+1;

  							
  							if (igm < 0) igm = grid_nx + igm;
  							if (igm > grid_nx - 1) igm = igm - grid_nx;
  							pos_x_m = (igm) + jg*grid_nx + kg*grid_nx*grid_ny;

  							if (igp < 0) igp = grid_nx + igp;
  							if (igp > grid_nx - 1) igp = igp - grid_nx;
  							pos_x_p = (igp) + jg*grid_nx + kg*grid_nx*grid_ny;

  							if (jgm < 0) jgm = grid_ny + jgm;
  							if (jgm > grid_ny - 1) jgm = jgm - grid_ny;
  							pos_y_m = (ig) + jgm*grid_nx + kg*grid_nx*grid_ny;

  							if (jgp  < 0) jgp  = grid_ny + jgp ;
  							if (jgp  > grid_ny - 1) jgp  = jgp  - grid_ny;
  							pos_y_p = (ig) + jgp*grid_nx + kg*grid_nx*grid_ny;

  							if (kgm < 0) kgm = grid_nz + kgm;
  							if (kgm > grid_nz - 1) kgm = kgm - grid_nz;
  							pos_z_m = (ig) + jg*grid_nx + kgm*grid_nx*grid_ny;

  							if (kgp < 0) kgp = grid_nz + kgp;
  							if (kgp > grid_nz - 1) kgp = kgp - grid_nz;
  							pos_z_p = (ig) + jg*grid_nx + kgp*grid_nx*grid_ny;

  							file_param_Obj->drho_x_train[idx] = file_param_Obj->drho_x_train[idx] +
  							grad_weights[w] * (1/mesh_x) *
  							(file_param_Obj->rho_total[pos_x_p]-file_param_Obj->rho_total[pos_x_m]);

  							file_param_Obj->drho_y_train[idx] = file_param_Obj->drho_y_train[idx] +
  							grad_weights[w] * (1/mesh_y) *
  							(file_param_Obj->rho_total[pos_y_p]-file_param_Obj->rho_total[pos_y_m]);

  							file_param_Obj->drho_z_train[idx] = file_param_Obj->drho_z_train[idx] +
  							grad_weights[w] * (1/mesh_z) *
  							(file_param_Obj->rho_total[pos_z_p]-file_param_Obj->rho_total[pos_z_m]);

  							file_param_Obj->ddrho_x_train[idx] = file_param_Obj->ddrho_x_train[idx] +
  							  grad_weights2[w+1]* (1/(mesh_x*mesh_x)) * 
  							(file_param_Obj->rho_total[pos_x_p] + file_param_Obj->rho_total[pos_x_m]);

  							file_param_Obj->ddrho_y_train[idx] = file_param_Obj->ddrho_y_train[idx] +
  							  grad_weights2[w+1]* (1/(mesh_y*mesh_y)) *
  							(file_param_Obj->rho_total[pos_y_p] + file_param_Obj->rho_total[pos_y_m]);

  							file_param_Obj->ddrho_z_train[idx] = file_param_Obj->ddrho_z_train[idx] +
  							  grad_weights2[w+1] * (1/(mesh_z*mesh_z)) *
  							 (file_param_Obj->rho_total[pos_z_p] + file_param_Obj->rho_total[pos_z_m]);

  						}
  						idx=idx+1;
  					// }
  				}
  			}
  		}

  		free(grad_weights);
  		free(grad_weights2);
  		file_param_Obj->Nx = grid_nx;
  		file_param_Obj->Ny = grid_ny;
  		file_param_Obj->Nz = grid_nz;

  		file_param_Obj->Nx_used = grid_nx_train;
  		file_param_Obj->Ny_used = grid_ny_train;
  		file_param_Obj->Nz_used = grid_nz_train;
  		#ifdef DEBUG
    		printf("file_param_Obj->Nx,Ny,Nz %d %d %d\n",file_param_Obj->Nx,
    			file_param_Obj->Ny,file_param_Obj->Nz);
    		printf("file_param_Obj->Nx_used,Ny_used,Nz_used %d %d %d\n",file_param_Obj->Nx_used,
    			file_param_Obj->Ny_used,file_param_Obj->Nz_used);
  		#endif
  // 		file_param_Obj->cell_typ = 0;
  // 		if (Hmat[0][1] == 0.0 && Hmat[0][2] == 0.0 && Hmat[1][0] == 0.0 && Hmat[1][2] == 0.0 && Hmat[2][0] == 0.0 && Hmat[2][1] == 0.0){
  // 			file_param_Obj->cell_typ = 0;
  // 		} else {
  // 			file_param_Obj->cell_typ = 1;
  // 		}

  // 		#ifdef DEBUG
		// 	printf("Cell typ is %d from processor %d and file_rank %d are: \n", file_param_Obj->cell_typ, rank, file_rank);
		// #endif
  		file_param_Obj->natoms = num_atoms;
  		file_param_Obj->chg_step_jumpX = step_jump_x;
  		file_param_Obj->chg_step_jumpY = step_jump_y;
  		file_param_Obj->chg_step_jumpZ = step_jump_z;
  		#ifdef DEBUG
    		printf("file_param_Obj->natoms %d \n",file_param_Obj->natoms);
    		printf("file_param_Obj->chg_step_jumpX,Y,Z %d %d %d\n",file_param_Obj->chg_step_jumpX,
    			file_param_Obj->chg_step_jumpY,file_param_Obj->chg_step_jumpZ);
  		#endif

    	#ifdef DEBUG
    		printf("file_param_Obj->LatVec: \n");
  		#endif

  		double Hmat_linearized[9];
  		for (int hi = 0; hi<3;hi++){
  			for (int hj = 0; hj<3;hj++){
  				Hmat_linearized[3*hi+hj] = Hmat[hi][hj];
  				file_param_Obj->LatVec[3*hi+hj] = Hmat[hi][hj];
  				
  			}
  			#ifdef DEBUG
    			printf("%1f %1f %1f \n",file_param_Obj->LatVec[3*hi+0],
    				file_param_Obj->LatVec[3*hi+1],file_param_Obj->LatVec[3*hi+2]);
  			#endif
  		}
  		// file_param_Obj->LatVec = Hmat_linearized;
  		file_param_Obj->Lx = L_x;
  		file_param_Obj->Ly = L_y;
  		file_param_Obj->Lz = L_z;

  		file_param_Obj->dx = mesh_x;
  		file_param_Obj->dy = mesh_y;
  		file_param_Obj->dz = mesh_z;

  		#ifdef DEBUG
    		printf("file_param_Obj->Lx,y,z: %1f %1f %1f\n",file_param_Obj->Lx,
    			file_param_Obj->Ly,file_param_Obj->Lz);
    		printf("file_param_Obj->dx,y,z: %1f %1f %1f\n",file_param_Obj->dx,
    			file_param_Obj->dy,file_param_Obj->dz);
    		printf("file_param_Obj->atom_typ: ");
  		#endif

  		file_param_Obj->atom_typ = (int *) malloc (num_elements * sizeof(int));
  		for (int nels = 0; nels < num_elements; nels++) {
  			file_param_Obj->atom_typ[nels] = natoms_per_species[nels];
  			#ifdef DEBUG
    			printf("%d ",file_param_Obj->atom_typ[nels]);
  			#endif
  		}


		file_param_Obj->total_grid_trained = total_grid_trained;
 		#ifdef DEBUG
    		printf("\n");
    		printf("file_param_Obj->total_grid_trained: %d",file_param_Obj->total_grid_trained);
  			printf("Broadcasting read values from the density files\n");
  		#endif
  		MPI_Bcast(&file_param_Obj->Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		// MPI_Bcast(&file_param_Obj->cell_typ, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpX, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpY, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpZ, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nx_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ny_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nz_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->total_grid_trained, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->LatVec, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->atom_typ, pMAIN_Obj->nelem, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->atom_pos, 3*file_param_Obj->natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->rho_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->drho_x_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->drho_y_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->drho_z_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->ddrho_x_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->ddrho_y_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->ddrho_z_train, total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {		
		file_param_Obj->atom_typ = (int *) malloc (pMAIN_Obj->nelem * sizeof(int));

		MPI_Bcast(&file_param_Obj->Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		// MPI_Bcast(&file_param_Obj->cell_typ, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpX, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpY, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->chg_step_jumpZ, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nx_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ny_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Nz_used, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->total_grid_trained, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->LatVec, 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->Lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->dz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  		MPI_Bcast(file_param_Obj->atom_typ, pMAIN_Obj->nelem , MPI_INT, 0, MPI_COMM_WORLD);
  		MPI_Bcast(&file_param_Obj->natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
  		file_param_Obj->atom_pos = (double *) malloc (3*file_param_Obj->natoms*sizeof(double));
  		MPI_Bcast(file_param_Obj->atom_pos, 3*file_param_Obj->natoms, MPI_DOUBLE, 0, MPI_COMM_WORLD); 		
		file_param_Obj->rho_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->drho_x_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->drho_y_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->drho_z_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_x_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_y_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));
		file_param_Obj->ddrho_z_train = (double *) malloc (file_param_Obj->total_grid_trained * sizeof (double));

		MPI_Bcast(file_param_Obj->rho_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->drho_x_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->drho_y_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->drho_z_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->ddrho_x_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->ddrho_y_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(file_param_Obj->ddrho_z_train, file_param_Obj->total_grid_trained, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	if (rank==0) free(natoms_per_species);
}


void Truncate_data_gather(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_rank) {
	int rank, procs_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	file_param_Obj->rho_train_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->drho_x_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->drho_y_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->drho_z_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->ddrho_x_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->ddrho_y_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));
	file_param_Obj->ddrho_z_procs = (double *) malloc (paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z * sizeof (double));

	#ifdef DEBUG
    	printf("Allocating memory for file_param_Obj->rho_train_procs: %d for rank %d\n",
    		paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z,rank);
    	// if (rank==0) printf("Allocating memory for drho_x,y,z_procs and ddrho_x,y,z_procs: %d\n",paral_Obj->node_num_x * paral_Obj->node_num_y * paral_Obj->node_num_z);
  	#endif

	int idx=0,idx1=0;
    
	// printf("file_param_Obj->N_used %d %d %d, Rank %d\n",file_param_Obj->Nx_used,
	// 	file_param_Obj->Ny_used,file_param_Obj->Nz_used, rank);

	// printf("paral_Obj->node_begin_x (%d %d) (%d %d) (%d %d), Rank %d\n",paral_Obj->node_begin_x,paral_Obj->node_end_x,
	// 	paral_Obj->node_begin_y,paral_Obj->node_end_y, paral_Obj->node_begin_z,paral_Obj->node_end_z,rank);
	#ifdef DEBUG
    	if (rank==0) printf("Storing rho and drho and ddrho values into file_param_Obj->rho_train_procs, drho_x,y,z_procs and drho_x,y,z_procs\n");
 	#endif
    for (int kg = 0; kg < file_param_Obj->Nz_used; kg++) {
		for (int jg = 0; jg < file_param_Obj->Ny_used; jg++) {
			for (int ig = 0; ig < file_param_Obj->Nx_used; ig++) {
				idx1=idx1+1;
				int pos_3d = ig + jg*file_param_Obj->Nx_used + kg*file_param_Obj->Nx_used*file_param_Obj->Ny_used;
				if (ig >= paral_Obj->node_begin_x && ig <= paral_Obj->node_end_x && jg >= paral_Obj->node_begin_y && jg <= paral_Obj->node_end_y && kg >= paral_Obj->node_begin_z && kg <= paral_Obj->node_end_z) {
					file_param_Obj->rho_train_procs[idx] = file_param_Obj->rho_train[pos_3d];
					file_param_Obj->drho_x_procs[idx] = file_param_Obj->drho_x_train[pos_3d];
					file_param_Obj->drho_y_procs[idx] = file_param_Obj->drho_y_train[pos_3d];
					file_param_Obj->drho_z_procs[idx] = file_param_Obj->drho_z_train[pos_3d];
					file_param_Obj->ddrho_x_procs[idx] = file_param_Obj->ddrho_x_train[pos_3d];
					file_param_Obj->ddrho_y_procs[idx] = file_param_Obj->ddrho_y_train[pos_3d];
					file_param_Obj->ddrho_z_procs[idx] = file_param_Obj->ddrho_z_train[pos_3d];
					idx=idx+1;
				}
			}
		}
	}
	#ifdef DEBUG
    	printf("Number of values for rho actually stored is %d for rank %d\n",idx,rank);
    	if (rank==0) printf("Free memory for file_param_Obj->rho_total, file_param_Obj->drho_x_train and file_param_Obj->ddrho_x_train");
  	#endif
	// MPI_Barrier(MPI_COMM_WORLD);
	// free(file_param_Obj->rho_train);

	if (rank==0){
		free(file_param_Obj->rho_total);
		free(file_param_Obj->rho_train);
		free(file_param_Obj->drho_x_train);
		free(file_param_Obj->drho_y_train);
		free(file_param_Obj->drho_z_train);
		free(file_param_Obj->ddrho_x_train);
		free(file_param_Obj->ddrho_y_train);
		free(file_param_Obj->ddrho_z_train);
	} else {
		free(file_param_Obj->rho_train);
		free(file_param_Obj->drho_x_train);
		free(file_param_Obj->drho_y_train);
		free(file_param_Obj->drho_z_train);
		free(file_param_Obj->ddrho_x_train);
		free(file_param_Obj->ddrho_y_train);
		free(file_param_Obj->ddrho_z_train);
	}
}

void calc_domain_grids_total(main_Obj *pMAIN_Obj) {

	int i, j, rank, nprocs;
	int Nx, Ny, Nz, Nd;
	int rem_x, rem_y, rem_z;
	int add_x, add_y, add_z;
	int add_num_x, add_num_y, add_num_z;
	int dims_cart[3],periods_cart[3], coords_cart[3];
	int node_begin_x, node_end_x, node_begin_y, node_end_y, node_begin_z, node_end_z;
	int coord_procs_x, coord_procs_y, coord_procs_z;
	int rank_cart;
	int periods[3]= {1,1,1}, dims[3];
	int ND[3];
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	pMAIN_Obj->grid_points_procs = 0;

	for (i=0; i < pMAIN_Obj->num_files; i++) {	
		Nx = pMAIN_Obj->Nx_used[i];
		Ny = pMAIN_Obj->Ny_used[i];
		Nz = pMAIN_Obj->Nz_used[i];
		Nd = Nx+Ny+Nz;
		double exp = 1/3;
		dims[0] = cbrt(nprocs);
		dims[1] = cbrt(nprocs);
		dims[2] = cbrt(nprocs);
		MPI_Comm cartesian_comm_3D;
		// periods = {1,1,1};
		// printf("rank %d, nprocs %d, dims (%d, %d, %d), Nused (%d, %d, %d) \n",rank,nprocs,dims[0],dims[1],dims[2], Nx, Ny, Nz);

		MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cartesian_comm_3D);

		MPI_Cart_get(cartesian_comm_3D, 3, dims_cart, periods_cart, coords_cart);
		
		// MPI_Cart_coords(cartesian_comm_3D, rank, 3, coords_cart);
		coord_procs_x = coords_cart[0];
		coord_procs_y = coords_cart[1];
		coord_procs_z = coords_cart[2];
		rem_x = Nx - ((Nx/dims[0])*dims[0]), rem_y = Ny - ((Ny/dims[1])*dims[1]), rem_z = Nz - ((Nz/dims[2])*dims[2]);
		add_x = MIN(coord_procs_x, rem_x), add_y = MIN(coord_procs_y, rem_y), add_z = MIN(coord_procs_z, rem_z);
		add_num_x = MIN(rem_x/(coord_procs_x+1),1), add_num_y = MIN(rem_y/(coord_procs_y+1),1), add_num_z = MIN(rem_z/(coord_procs_z+1),1);

	 // Nodes to be covered by the current prcoessor
		node_begin_x = coord_procs_x*(Nx/dims[0])+add_x;
		node_end_x = node_begin_x+(Nx/dims[0])-1+add_num_x;
		node_begin_y = coord_procs_y*(Ny/dims[1])+add_y;
		node_end_y = node_begin_y+(Ny/dims[1])-1+add_num_y;
		node_begin_z = coord_procs_z*(Nz/dims[2])+add_z;
		node_end_z = node_begin_z+(Nz/dims[2])-1+add_num_z;

		ND[0] = (node_end_x-node_begin_x) + 1;
		ND[1] = (node_end_y-node_begin_y) + 1;
		ND[2] = (node_end_z-node_begin_z) + 1;


		pMAIN_Obj->grid_points_procs = pMAIN_Obj->grid_points_procs + ND[0]* ND[1] * ND[2];

		MPI_Comm_free(&cartesian_comm_3D);
	}
}

void Calc_C11_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index) {
	int grid_i, grid_j, grid_k, atom_idx;
	double box_volume, dx_r[3], dxc[3],  corr11_vec[pMAIN_Obj->nelem * pMAIN_Obj->num_bases_C11];
	int idx_1D, chem_idx;
	int i, j ,k, i_elem, i_Gauss, coeff_track;
	double dxr[3];
	double dr;
	int nproc, rank;
	double temp0, temp1, temp2;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	box_volume = volume_Hmat(file_param_Obj->LatVec);
	int NDx, NDy, NDz, der_terms;
	double **corr_derivatives;
	int Gauss_idx, poly_degree;
	double inv_sigma, func_out[3];
	int numD;
	double maxdist;
	double **cheb_bases_dr;


	cheb_bases_dr = (double **) malloc(pMAIN_Obj->num_bases_C11*sizeof(double*));
	for (int bas =0; bas < pMAIN_Obj->num_bases_C11; bas++){
		cheb_bases_dr[bas] = (double *) malloc(3*sizeof(double));
	}

	maxdist = sqrt(file_param_Obj->Lx * file_param_Obj->Lx + file_param_Obj->Ly * file_param_Obj->Ly + file_param_Obj->Lz * file_param_Obj->Lz);

	numD = pMAIN_Obj->num_bases_C11;
	
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1)
		der_terms = 6;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0)
		der_terms = 0;

	if (pMAIN_Obj->isgradient == 1 || pMAIN_Obj->ishessian == 1){
		corr_derivatives = (double **)malloc(der_terms *sizeof(double*));
		for(i=0; i<der_terms; i++)
			corr_derivatives[i] = (double *)malloc(pMAIN_Obj->nelem * pMAIN_Obj->num_bases_C11 *sizeof(double));
	}
	#ifdef DEBUG
    	if (rank==0) printf("Inside Calc_C11_terms\n");
    	if (rank==0) printf("der_terms %d\n",der_terms);
    	if (rank==0) printf("Memory for corr_derivatives: %d*%d\n",der_terms,pMAIN_Obj->nelem * pMAIN_Obj->num_bases_C11);
  	#endif
	

	NDx = paral_Obj->node_num_x ;
	NDy = paral_Obj->node_num_y ;
	NDz = paral_Obj->node_num_z ;
	
	#ifdef DEBUG
    	if (rank==0) printf("NDx,y,z %d %d %d\n",NDx,NDy,NDz);
  	#endif

	if (file_index==0){
		pMAIN_Obj->count2=0;
		pMAIN_Obj->count1=0;
		pMAIN_Obj->count3=0;
	} else {
		pMAIN_Obj->count2=0;
		pMAIN_Obj->count1=0;
		pMAIN_Obj->count3=0;
		for(i=0;i<file_index;i++) {
			pMAIN_Obj->count2 = pMAIN_Obj->count2 + pMAIN_Obj->node_num_x[i] *pMAIN_Obj->node_num_y[i]
			 *pMAIN_Obj->node_num_z[i]*(1+der_terms);
			 pMAIN_Obj->count1 = pMAIN_Obj->count1 + pMAIN_Obj->node_num_x[i] *pMAIN_Obj->node_num_y[i]
			 *pMAIN_Obj->node_num_z[i]*(1+der_terms);
			 pMAIN_Obj->count3 = pMAIN_Obj->count3 + pMAIN_Obj->node_num_x[i] *pMAIN_Obj->node_num_y[i]
			 *pMAIN_Obj->node_num_z[i];
		}
	}

	#ifdef DEBUG
    	if (rank==0) printf("pMAIN_Obj->count1, pMAIN_Obj->count2 %d %d\n",pMAIN_Obj->count1,pMAIN_Obj->count2);
  	#endif


	for (int grid_k = 0; grid_k <NDz; grid_k++) {
		// if (rank==0) printf("grid_k %d/%d, num_bases %d\n",grid_k,NDz,pMAIN_Obj->num_bases);
		for (int grid_j = 0; grid_j < NDy; grid_j++) {
			for (int grid_i = 0; grid_i < NDx; grid_i++) {

				idx_1D = grid_i + grid_j *  NDx + grid_k * NDy * NDx;
				pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->rho_train_procs[idx_1D]/box_volume;	
				pMAIN_Obj->bvec_domain_rho_only[pMAIN_Obj->count3] = file_param_Obj->rho_train_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainx[pMAIN_Obj->count3] = file_param_Obj->drho_x_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainy[pMAIN_Obj->count3] = file_param_Obj->drho_y_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainz[pMAIN_Obj->count3] = file_param_Obj->drho_z_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainxx[pMAIN_Obj->count3] = file_param_Obj->ddrho_x_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainyy[pMAIN_Obj->count3] = file_param_Obj->ddrho_y_procs[idx_1D]/box_volume;
				pMAIN_Obj->bvec_domainzz[pMAIN_Obj->count3] = file_param_Obj->ddrho_z_procs[idx_1D]/box_volume;
				// pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][0] = 1.0;
				pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;//(grid_i+paral_Obj->node_begin_x)*file_param_Obj->chg_step_jumpX
				pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
				pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;

				pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
				pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 0;
				pMAIN_Obj->count1=pMAIN_Obj->count1+1;
				pMAIN_Obj->count3=pMAIN_Obj->count3+1;
				
				if (pMAIN_Obj->isgradient==1){
					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->drho_x_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 1;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;

					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->drho_y_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 2;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;

					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->drho_z_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 3;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;

					if (pMAIN_Obj->ishessian==1){
						pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_x_procs[idx_1D]/box_volume;	
						pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
						pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
						pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
						pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
						pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 4;
						pMAIN_Obj->count1=pMAIN_Obj->count1+1;

						pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_y_procs[idx_1D]/box_volume;	
						pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
						pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
						pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
						pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
						pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 5;
						pMAIN_Obj->count1=pMAIN_Obj->count1+1;

						pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_z_procs[idx_1D]/box_volume;	
						pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
						pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
						pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
						pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
						pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 6;
						pMAIN_Obj->count1=pMAIN_Obj->count1+1;
					}
				} else if (pMAIN_Obj->ishessian==1){
					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_x_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 1;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;

					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_y_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 2;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;

					pMAIN_Obj->bvec_domain[pMAIN_Obj->count1] = file_param_Obj->ddrho_z_procs[idx_1D]/box_volume;	
					pMAIN_Obj->I_idx[pMAIN_Obj->count1] = grid_i;
					pMAIN_Obj->J_idx[pMAIN_Obj->count1] = grid_j;
					pMAIN_Obj->K_idx[pMAIN_Obj->count1] = grid_k;
					pMAIN_Obj->file_idx[pMAIN_Obj->count1] = file_index;
					pMAIN_Obj->der_idx[pMAIN_Obj->count1] = 3;
					pMAIN_Obj->count1=pMAIN_Obj->count1+1;
				}
				
				// coeff_track=coeff_track+1;
				for (i=0;i<pMAIN_Obj->nelem * pMAIN_Obj->num_bases_C11;i++) {
					corr11_vec[i]=0.0;
					for(j=0;j<der_terms;j++){
						corr_derivatives[j][i] = 0.0;
					}
				}
				temp0 = (float) (grid_i+paral_Obj->node_begin_x) *
				 ((float) file_param_Obj->chg_step_jumpX / (float) file_param_Obj->Nx);

				temp1 = (float) (grid_j+paral_Obj->node_begin_y) 
				* ((float) file_param_Obj->chg_step_jumpY / (float) file_param_Obj->Ny);

				temp2 = (float) (grid_k+paral_Obj->node_begin_z) 
				* ((float) file_param_Obj->chg_step_jumpZ / (float) file_param_Obj->Nz);

				for (atom_idx = 0; atom_idx < file_param_Obj->natoms ; atom_idx++) {
					
					dxr[0] = temp0 - file_param_Obj->atom_pos[3*atom_idx+ 0];
					dxr[1] = temp1 - file_param_Obj->atom_pos[3*atom_idx+ 1];
					dxr[2] = temp2 - file_param_Obj->atom_pos[3*atom_idx+ 2];
					dxr[0] = checkPBC_Shifted(dxr[0]); // PBC, i.e. -0.50 < dxr < 0.50 
	        dxr[1] = checkPBC_Shifted(dxr[1]); // PBC, i.e. -0.50 < dxr < 0.50 
			  	dxr[2] = checkPBC_Shifted(dxr[2]); // PBC, i.e. -0.50 < dxr < 0.50 

			  	V3M3mulV3(dxr, file_param_Obj->LatVec, dxc);
			  	dr = V3NormS(dxc);
			  	
			  	func_cheb(dr, pMAIN_Obj->num_bases_C11, cheb_bases_dr, maxdist,pMAIN_Obj->Rcut);

			  		// dr = sqrt(file_param_Obj->Lx * file_param_Obj->Lx * dxr[0] * dxr[0] + file_param_Obj->Ly * file_param_Obj->Ly * dxr[1] * dxr[1] + file_param_Obj->Lz * file_param_Obj->Lz * dxr[2] * dxr[2]) ;
			  		// if(rank==0 && atom_idx==2 && grid_j==1 && grid_k ==1) printf("dr for atom 3 is %1f for grid_i %d\n", dr,grid_i);
			  	chem_idx=find_chem_idx(atom_idx, file_param_Obj, pMAIN_Obj);		
			  	for (i_Gauss = 0; i_Gauss < pMAIN_Obj->num_bases_C11; i_Gauss++){
			  		if (pMAIN_Obj->ischeb!=1){
			  			Gauss_idx =  i_Gauss / (1+pMAIN_Obj->poly_order_C11);
			  			poly_degree =  i_Gauss % (1+pMAIN_Obj->poly_order_C11);
			  			inv_sigma = pMAIN_Obj->Gaussian_Width_C11[Gauss_idx][chem_idx];
			  			func_GTO(inv_sigma, dr, poly_degree, func_out, pMAIN_Obj->ischeb, maxdist);
			  		} else{
			  			func_out[0] = cheb_bases_dr[i_Gauss][0];
			  			func_out[1] = cheb_bases_dr[i_Gauss][1];
			  			func_out[2] = cheb_bases_dr[i_Gauss][2];
			  		}
			  		
			  		corr11_vec[chem_idx*numD + i_Gauss] = 
			  		corr11_vec[chem_idx*numD + i_Gauss] + func_out[0];

			  		if (pMAIN_Obj->isgradient==1) {
			  		corr_derivatives[0][chem_idx*numD + i_Gauss] = 
			  		    corr_derivatives[0][chem_idx*numD + i_Gauss] +
			  		    func_out[1]*dxc[0]/dr;

			  		corr_derivatives[1][chem_idx*numD + i_Gauss] = 
			  		    corr_derivatives[1][chem_idx*numD + i_Gauss] +
			  		    func_out[1]*dxc[1]/dr;

			  		corr_derivatives[2][chem_idx*numD + i_Gauss] = 
			  		    corr_derivatives[2][chem_idx*numD + i_Gauss] +
			  		    func_out[1]*dxc[2]/dr;

			  		if (pMAIN_Obj->ishessian==1) {
			  		  corr_derivatives[3][chem_idx*numD + i_Gauss] = 
								corr_derivatives[3][chem_idx*numD + i_Gauss] +
								func_out[2]*(dxc[0])*(dxc[0])*(1/dr)*(1/dr) +
								func_out[1]*(1/dr)+
								func_out[1]*(dxc[0])*(dxc[0])*(1/dr)*(-1/dr)*(1/dr);

							corr_derivatives[4][chem_idx*numD + i_Gauss] = 
								corr_derivatives[4][chem_idx*numD + i_Gauss] +
								func_out[2]*(dxc[1])*(dxc[1])*(1/dr)*(1/dr) +
								func_out[1]*(1/dr)+
								func_out[1]*(dxc[1])*(dxc[1])*(1/dr)*(-1/dr)*(1/dr);

							corr_derivatives[5][chem_idx*numD + i_Gauss] = 
								corr_derivatives[5][chem_idx*numD + i_Gauss] +
								func_out[2]*(dxc[2])*(dxc[2])*(1/dr)*(1/dr) +
								func_out[1]*(1/dr)+
								func_out[1]*(dxc[2])*(dxc[2])*(1/dr)*(-1/dr)*(1/dr);

			  		    	}	
			  		    } else if (pMAIN_Obj->ishessian==1) {

			  		    		corr_derivatives[0][chem_idx*numD + i_Gauss] = 
										corr_derivatives[0][chem_idx*numD + i_Gauss] +
										func_out[2]*(dxc[0])*(dxc[0])*(1/dr)*(1/dr) +
										func_out[1]*(1/dr)+
										func_out[1]*(dxc[0])*(dxc[0])*(1/dr)*(-1/dr)*(1/dr);

										corr_derivatives[1][chem_idx*numD + i_Gauss] = 
										corr_derivatives[1][chem_idx*numD + i_Gauss] +
										func_out[2]*(dxc[1])*(dxc[1])*(1/dr)*(1/dr) +
										func_out[1]*(1/dr)+
										func_out[1]*(dxc[1])*(dxc[1])*(1/dr)*(-1/dr)*(1/dr);

										corr_derivatives[2][chem_idx*numD + i_Gauss] = 
										corr_derivatives[2][chem_idx*numD + i_Gauss] +
										func_out[2]*(dxc[2])*(dxc[2])*(1/dr)*(1/dr) +
										func_out[1]*(1/dr)+
										func_out[1]*(dxc[2])*(dxc[2])*(1/dr)*(-1/dr)*(1/dr);
			  		    }
			  		}
				}	

				for (i=0; i<der_terms+1; i++){
					coeff_track = 0;
					if (i==0) pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track]=1.0;
					if (i>0) pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track]=0.0;
					coeff_track = 1;
					for (i_elem = 0; i_elem < pMAIN_Obj->nelem; i_elem++){
						for (i_Gauss = 0; i_Gauss < numD; i_Gauss++) {
							if (i==0) {
								pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] =
								corr11_vec[i_elem * numD + i_Gauss];	
							} else{
									pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] =
									corr_derivatives[i-1][i_elem * numD + i_Gauss];
							}	
							coeff_track=coeff_track+1;
							// fprintf(fptr1,"%1f  ",pMAIN_Obj->Amat_domain[pMAIN_Obj->count2]);						
						}
					}
					// pMAIN_Obj->count1 = pMAIN_Obj->count1+1;
					pMAIN_Obj->count2 = pMAIN_Obj->count2+1;
				}							
			}			
		}
		// printf("%d\n",grid_k);
	}

	if (pMAIN_Obj->isgradient == 1 || pMAIN_Obj->ishessian == 1){
		for(i=0; i<der_terms; i++)
			free(corr_derivatives[i]);
		free(corr_derivatives);
	}
	for (int bas =0; bas < pMAIN_Obj->num_bases_C11; bas++){
		free(cheb_bases_dr[bas]);
	}


}



void Calc_C22_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index) {
	int grid_i, grid_j, grid_k; 
	double box_volume, dx_r[3], dxc[3],dxc1[3];
	double **corr22_mat,**corr23_mat;
	int idx_1D, chem_idx;
	int i, j ,k, m, i_elem, i_Gauss;
	int coeff_idx1,coeff_idx2,atom_idx1,atom_idx,iv1,iv2,iv11,iv22,coeff_track;
	double var1, var2,var3,var4,var5,var5_C23,sigma1_ij,G1_ij,G2_ij, temp0, temp1, temp2;
	double dxr[3],dxr1[3];
	double dr, dr1;
	int nproc, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	box_volume = volume_Hmat(file_param_Obj->LatVec);
	int NDx, NDy, NDz, der_terms;
	double *dist_ij;
	double var1x, var1y, var1z, var3x, var3y, var3z, var1xx, var1yy, var1zz, var3xx, var3yy, var3zz;
	double var5x,var5y,var5z,var5xx,var5yy,var5zz,var5x_C23,var5y_C23,var5z_C23,var5xx_C23,var5yy_C23,var5zz_C23;
	double atom1_x, atom1_y, atom1_z, atom2_x, atom2_y, atom2_z;
	double ***corr_derivatives, ***corr_derivatives_C23;
	int Gauss_index_1, Gauss_index_2, Poly_order_1, Poly_order_2;
	int Gauss_index, Poly_order;
	double func_out1[3], func_out2[3], func_out3[3];
	double **basis_matrixG1;
	double dvar1,dvar2,ddvar1,ddvar2,dvar3,dvar4,ddvar3,ddvar4;
	int numD, Poly_order_ij;
	double maxdist;
	double **cheb_bases_dr23, **cheb_bases_dr22;

	cheb_bases_dr23 = (double **) malloc((2*(pMAIN_Obj->num_bases_C22-1)+1)*sizeof(double*));
	cheb_bases_dr22 = (double **) malloc(pMAIN_Obj->num_bases_C22*sizeof(double*));
	for (int bas =0; bas < 2*(pMAIN_Obj->num_bases_C22-1)+1; bas++){
		cheb_bases_dr23[bas] = (double *) malloc(3*sizeof(double));
	}
	for (int bas =0; bas < pMAIN_Obj->num_bases_C22; bas++){
		cheb_bases_dr22[bas] = (double *) malloc(3*sizeof(double));
	}

	numD = pMAIN_Obj->num_bases_C22;
	maxdist = sqrt(file_param_Obj->Lx * file_param_Obj->Lx + file_param_Obj->Ly * file_param_Obj->Ly + file_param_Obj->Lz * file_param_Obj->Lz);
	// printf("11.came here\n");
	// file_param_Obj->basis_matrixG = (double *)malloc(file_param_Obj->natoms*numD*sizeof(double));
	// printf("22.came here\n");

	basis_matrixG1 = (double **)malloc(3*sizeof(double*));
	for(i=0; i<3 ; i++){
		basis_matrixG1[i] = (double *)malloc(file_param_Obj->natoms*numD*sizeof(double));
	}

	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1)
		der_terms = 6;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0)
		der_terms = 0;


	if (pMAIN_Obj->isgradient == 1 || pMAIN_Obj->ishessian == 1) {
		corr_derivatives = (double ***)malloc(der_terms*sizeof(double**));
		if(pMAIN_Obj->isC23==1) corr_derivatives_C23 = (double ***)malloc(der_terms*sizeof(double**));	
		for(i=0; i<der_terms ; i++){
			corr_derivatives[i] = (double **)malloc(pMAIN_Obj->nelem * pMAIN_Obj->nelem*sizeof(double*));
			if(pMAIN_Obj->isC23==1) corr_derivatives_C23[i] = (double **)malloc(pMAIN_Obj->nelem * pMAIN_Obj->nelem*sizeof(double*));
			for(j=0; j<pMAIN_Obj->nelem * pMAIN_Obj->nelem ; j++){
				corr_derivatives[i][j] = (double *)malloc(numD * numD*sizeof(double));
				if(pMAIN_Obj->isC23==1) corr_derivatives_C23[i][j] = (double *)malloc(numD * numD*sizeof(double));
			}
		}
	}


	typedef struct
	{
	  	double *mat;
	} corr23G;

	corr23G *Gij_struct;

	dist_ij = (double *) malloc(file_param_Obj->natoms * file_param_Obj->natoms * sizeof(double));
	Gij_struct = (corr23G *) malloc (file_param_Obj->natoms * file_param_Obj->natoms* sizeof (corr23G));

	for (i = 0; i < file_param_Obj->natoms * file_param_Obj->natoms; i++) 
		Gij_struct[i].mat = (double *) malloc (numD * numD * sizeof(double));

	if (pMAIN_Obj->isC23==1) {	
		for (atom_idx = 0; atom_idx < file_param_Obj->natoms; atom_idx++){
			for (atom_idx1 = (atom_idx + 1); atom_idx1 < file_param_Obj->natoms; atom_idx1++){
				dxr[0] = file_param_Obj->atom_pos[3*atom_idx + 0] - file_param_Obj->atom_pos[3*atom_idx1 + 0]; // dimensionless 
				dxr[1] = file_param_Obj->atom_pos[3*atom_idx + 1] - file_param_Obj->atom_pos[3*atom_idx1 + 1];
				dxr[2] = file_param_Obj->atom_pos[3*atom_idx + 2] - file_param_Obj->atom_pos[3*atom_idx1 + 2];
				
				dxr[0] = checkPBC_Shifted(dxr[0]); // PBC, i.e. -0.50 < dxr < 0.50 
	      dxr[1] = checkPBC_Shifted(dxr[1]); // PBC, i.e. -0.50 < dxr < 0.50 
			  dxr[2] = checkPBC_Shifted(dxr[2]); // PBC, i.e. -0.50 < dxr < 0.50 

			  V3M3mulV3(dxr, file_param_Obj->LatVec, dxc);
			  dr = V3NormS(dxc);

			  func_cheb(dr, 2*numD-2+1, cheb_bases_dr23, maxdist,pMAIN_Obj->Rcut);

				dist_ij[atom_idx * file_param_Obj->natoms + atom_idx1] = dr;
				dist_ij[atom_idx1 * file_param_Obj->natoms + atom_idx] = dr;

				i = find_chem_idx(atom_idx, file_param_Obj, pMAIN_Obj);
				j = find_chem_idx(atom_idx1, file_param_Obj, pMAIN_Obj);
				iv1 = atom_idx*file_param_Obj->natoms + atom_idx1;
				for (coeff_idx1 = 0; coeff_idx1 < numD; coeff_idx1++) {
					for (coeff_idx2 = 0; coeff_idx2 < numD; coeff_idx2++) {
						if (pMAIN_Obj->ischeb!= 1){
							Gauss_index_1 =  coeff_idx1 / (1+pMAIN_Obj->poly_order_C22);
							Gauss_index_2 =  coeff_idx2 / (1+pMAIN_Obj->poly_order_C22);
							Poly_order_1  =  coeff_idx1 % (1+pMAIN_Obj->poly_order_C22);
							Poly_order_2  =  coeff_idx2 % (1+pMAIN_Obj->poly_order_C22);
						
							Poly_order_ij = Poly_order_1+Poly_order_2;
							sigma1_ij = pMAIN_Obj->Gaussian_Width_C22[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C22[Gauss_index_2][j];
							sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C22[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C22[Gauss_index_2][j]);
							func_GTO(sigma1_ij, dr, Poly_order_ij, func_out1, pMAIN_Obj->ischeb, maxdist);
						} else {
							Poly_order_ij = coeff_idx1 + coeff_idx2;
							// func_GTO(0.0, dr, Poly_order_ij, func_out1, pMAIN_Obj->ischeb, maxdist);
							func_out1[0] = cheb_bases_dr23[Poly_order_ij][0];
							func_out1[1] = cheb_bases_dr23[Poly_order_ij][1];
							func_out1[2] = cheb_bases_dr23[Poly_order_ij][2];
						}
						
						Gij_struct[iv1].mat[coeff_idx1 * numD + coeff_idx2] = func_out1[0];
					}
				}
			}
		}

		corr23_mat = (double **)malloc(pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double *));
		for (i=0; i<pMAIN_Obj->nelem * pMAIN_Obj->nelem; i++){
			corr23_mat[i] = (double *)malloc(numD * numD* sizeof(double));
		}
	}


	// printf("entered c22 calc function by proc rank %d\n",rank);

	corr22_mat = (double **)malloc(pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double *));
	

	for (i=0; i<pMAIN_Obj->nelem * pMAIN_Obj->nelem; i++){
		corr22_mat[i] = (double *)malloc(numD * numD* sizeof(double));
	}

		

	NDx = paral_Obj->node_num_x ;
	NDy = paral_Obj->node_num_y ;
	NDz = paral_Obj->node_num_z ;

	if (file_index==0){
		pMAIN_Obj->count2=0;
	} else {
		pMAIN_Obj->count2=0;
		for(i=0;i<file_index;i++) {
			pMAIN_Obj->count2 = pMAIN_Obj->count2 + pMAIN_Obj->node_num_x[i] *pMAIN_Obj->node_num_y[i]
			 *pMAIN_Obj->node_num_z[i]*(1+der_terms);
		}
	}
	double **pos_store;
	pos_store = (double **)malloc(4 * sizeof(double *));
	for(i=0;i<4;i++) {
		pos_store[i] = (double *)malloc(file_param_Obj->natoms*sizeof(double));
	}

	for (int grid_k = 0; grid_k <NDz; grid_k++) {
		for (int grid_j = 0; grid_j < NDy; grid_j++) {
			for (int grid_i = 0; grid_i < NDx; grid_i++) {
				// printf("1.came here\n");
				temp0 = (float) (grid_i+paral_Obj->node_begin_x) * ((float) file_param_Obj->chg_step_jumpX / (float) file_param_Obj->Nx);
				temp1 = (float) (grid_j+paral_Obj->node_begin_y) * ((float) file_param_Obj->chg_step_jumpY / (float) file_param_Obj->Ny);
				temp2 = (float) (grid_k+paral_Obj->node_begin_z) * ((float) file_param_Obj->chg_step_jumpZ / (float) file_param_Obj->Nz);
				for (atom_idx = 0; atom_idx < file_param_Obj->natoms ; atom_idx++) {			
						dxr[0] = temp0 - file_param_Obj->atom_pos[3*atom_idx+ 0];
						dxr[1] = temp1 - file_param_Obj->atom_pos[3*atom_idx+ 1];
						dxr[2] = temp2 - file_param_Obj->atom_pos[3*atom_idx+ 2];
						dxr[0] = checkPBC_Shifted(dxr[0]); // PBC, i.e. -0.50 < dxr < 0.50 
		        dxr[1] = checkPBC_Shifted(dxr[1]); // PBC, i.e. -0.50 < dxr < 0.50 
				  	dxr[2] = checkPBC_Shifted(dxr[2]); // PBC, i.e. -0.50 < dxr < 0.50 

			  		V3M3mulV3(dxr, file_param_Obj->LatVec, dxc);
			  		dr = V3NormS(dxc);
			  		func_cheb(dr, pMAIN_Obj->num_bases_C22, cheb_bases_dr22, maxdist,pMAIN_Obj->Rcut);
			  		pos_store[0][atom_idx] = dxc[0];
			  		pos_store[1][atom_idx] = dxc[1];
			  		pos_store[2][atom_idx] = dxc[2];
			  		pos_store[3][atom_idx] = dr;
			  		// dr = sqrt(file_param_Obj->Lx * file_param_Obj->Lx * dxr[0] * dxr[0] + file_param_Obj->Ly * file_param_Obj->Ly * dxr[1] * dxr[1] + file_param_Obj->Lz * file_param_Obj->Lz * dxr[2] * dxr[2]) ;
			  		// if(rank==0 && atom_idx==2 && grid_j==1 && grid_k ==1) printf("dr for atom 3 is %1f for grid_i %d\n", dr,grid_i);
			  		chem_idx=find_chem_idx(atom_idx, file_param_Obj, pMAIN_Obj);		
			  		for (i_Gauss = 0; i_Gauss < numD; i_Gauss++){
			  			if (pMAIN_Obj->ischeb!=1){
			  				Gauss_index =  i_Gauss / (1+pMAIN_Obj->poly_order_C22);
				  			Poly_order =  i_Gauss % (1+pMAIN_Obj->poly_order_C22);
				  			func_GTO(pMAIN_Obj->Gaussian_Width_C22[Gauss_index][chem_idx], dr, Poly_order, func_out2, pMAIN_Obj->ischeb, maxdist);
			  			} else {
			  				// func_GTO(0.0, dr, i_Gauss, func_out2, pMAIN_Obj->ischeb, maxdist);
			  				func_out2[0] = cheb_bases_dr22[i_Gauss][0];
			  				func_out2[1] = cheb_bases_dr22[i_Gauss][0];
			  				func_out2[2] = cheb_bases_dr22[i_Gauss][0];
			  			}		  			
			  			basis_matrixG1[0][atom_idx*numD + i_Gauss] 
			  			= func_out2[0]; 
			  			basis_matrixG1[1][atom_idx*numD + i_Gauss] 
			  			= func_out2[1]; 
			  			basis_matrixG1[2][atom_idx*numD + i_Gauss] 
			  			= func_out2[2];
			  		}
				}


				for (i = 0; i < pMAIN_Obj->nelem; i++){
					for (j = 0; j < pMAIN_Obj->nelem; j++){
						for (k = 0; k < numD; k++){
							for (m = 0; m < numD; m++){
								corr22_mat[i*pMAIN_Obj->nelem+j][k*numD + m] = 0.0;
								if(pMAIN_Obj->isC23==1) corr23_mat[i*pMAIN_Obj->nelem+j][k*numD + m] = 0.0;
								for (int ii=0; ii<der_terms; ii++){
									corr_derivatives[ii][i*pMAIN_Obj->nelem+j][k*numD + m] = 0.0;
									if (pMAIN_Obj->isC23==1) corr_derivatives_C23[ii][i*pMAIN_Obj->nelem+j][k*numD + m] = 0.0;
								}
							}
						}
					}
				}
				
				for (coeff_idx1 = 0; coeff_idx1 < numD; coeff_idx1++){
					for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD; coeff_idx2++){
						for (atom_idx = 0; atom_idx < file_param_Obj->natoms; atom_idx++){
							for (atom_idx1 = (atom_idx +1); atom_idx1 < file_param_Obj->natoms; atom_idx1++){
								// printf("1. came here %d/%d\n",coeff_idx1,numD);
								iv1 = atom_idx * file_param_Obj->natoms + atom_idx1;
								i = find_chem_idx(atom_idx, file_param_Obj, pMAIN_Obj);
								j = find_chem_idx(atom_idx1, file_param_Obj, pMAIN_Obj);

								dxc[0] = pos_store[0][atom_idx];
								dxc[1] = pos_store[1][atom_idx];
								dxc[2] = pos_store[2][atom_idx];

								dxc1[0] = pos_store[0][atom_idx1];
								dxc1[1] = pos_store[1][atom_idx1];
								dxc1[2] = pos_store[2][atom_idx1];

								dr = pos_store[3][atom_idx];
								dr1 = pos_store[3][atom_idx1];

								var1 = basis_matrixG1[0][atom_idx*numD + coeff_idx1];
								var2 = basis_matrixG1[0][atom_idx1*numD + coeff_idx2];

								dvar1 = basis_matrixG1[1][atom_idx*numD + coeff_idx1];
								dvar2 = basis_matrixG1[1][atom_idx1*numD + coeff_idx2];
								
								ddvar1 = basis_matrixG1[2][atom_idx*numD + coeff_idx1];
								ddvar2 = basis_matrixG1[2][atom_idx1*numD + coeff_idx2];
								// printf("2. came here %d/%d\n",coeff_idx1,numD);

								if (pMAIN_Obj->isgradient==1) {
									var1x = var1*dvar2*dxc1[0]*(1/dr1) + dvar1*var2*dxc[0]*(1/dr);
									var1y = var1*dvar2*dxc1[1]*(1/dr1) + dvar1*var2*dxc[1]*(1/dr);
									var1z = var1*dvar2*dxc1[2]*(1/dr1) + dvar1*var2*dxc[2]*(1/dr);
								}
								if (pMAIN_Obj->ishessian==1){
									var1xx = var1*ddvar2*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1)+
													 dvar1*dvar2*dxc[0]*dxc1[0]*(1/dr)*(1/dr1)+
													 var1*dvar2*(1/dr1)+
													 var1*dvar2*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar1*var2*dxc[0]*dxc[0]*(1/dr)*(1/dr)+
													 dvar1*dvar2*dxc[0]*dxc1[0]*(1/dr)*(1/dr1)+
													 dvar1*var2*(1/dr)+
													 dvar1*var2*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

									var1yy = var1*ddvar2*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1)+
													 dvar1*dvar2*dxc[1]*dxc1[1]*(1/dr)*(1/dr1)+
													 var1*dvar2*(1/dr1)+
													 var1*dvar2*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar1*var2*dxc[1]*dxc[1]*(1/dr)*(1/dr)+
													 dvar1*dvar2*dxc[1]*dxc1[1]*(1/dr)*(1/dr1)+
													 dvar1*var2*(1/dr)+
													 dvar1*var2*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

									var1zz = var1*ddvar2*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1)+
													 dvar1*dvar2*dxc[2]*dxc1[2]*(1/dr)*(1/dr1)+
													 var1*dvar2*(1/dr1)+
													 var1*dvar2*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar1*var2*dxc[2]*dxc[2]*(1/dr)*(1/dr)+
													 dvar1*dvar2*dxc[2]*dxc1[2]*(1/dr)*(1/dr1)+
													 dvar1*var2*(1/dr)+
													 dvar1*var2*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);

								}										
								// printf("3. came here %d/%d\n",coeff_idx1,numD);
								if(pMAIN_Obj->isC23==1) {
									iv11 = coeff_idx1 * numD + coeff_idx2;
									iv22 =	coeff_idx2 * numD + coeff_idx1;
									G1_ij = Gij_struct[iv1].mat[iv11];
									G2_ij = Gij_struct[iv1].mat[iv22];
								}
								

								var3 = basis_matrixG1[0][atom_idx*numD + coeff_idx2];
								var4 = basis_matrixG1[0][atom_idx1*numD + coeff_idx1];

								dvar3 = basis_matrixG1[1][atom_idx*numD + coeff_idx2];
								dvar4 = basis_matrixG1[1][atom_idx1*numD + coeff_idx1];

								ddvar3 = basis_matrixG1[2][atom_idx*numD + coeff_idx2];
								ddvar4 = basis_matrixG1[2][atom_idx1*numD + coeff_idx1];
								// if (grid_i==0 && grid_j==0 && grid_k==0 && i==0 && j==0 && coeff_idx1==0 && coeff_idx2==0) printf("var is (%1f %1f %1f %1f %1f) atom_idx and atom_idx1 is %d and %d for (0,0,0) and elem=0\n",var1,var2,var3,var4,var5,atom_idx,atom_idx1);
								
								if (pMAIN_Obj->isgradient==1) {
									var3x = var3*dvar4*dxc1[0]*(1/dr1) + dvar3*var4*dxc[0]*(1/dr);
									var3y = var3*dvar4*dxc1[1]*(1/dr1) + dvar3*var4*dxc[1]*(1/dr);
									var3z = var3*dvar4*dxc1[2]*(1/dr1) + dvar3*var4*dxc[2]*(1/dr);
								}

								if (pMAIN_Obj->ishessian==1){
									var3xx = var3*ddvar4*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1)+
													 dvar3*dvar4*dxc[0]*dxc1[0]*(1/dr)*(1/dr1)+
													 var3*dvar4*(1/dr1)+
													 var3*dvar4*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar3*var4*dxc[0]*dxc[0]*(1/dr)*(1/dr)+
													 dvar3*dvar4*dxc[0]*dxc1[0]*(1/dr)*(1/dr1)+
													 dvar3*var4*(1/dr)+
													 dvar3*var4*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);
													 
									var3yy = var3*ddvar4*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1)+
													 dvar3*dvar4*dxc[1]*dxc1[1]*(1/dr)*(1/dr1)+
													 var3*dvar4*(1/dr1)+
													 var3*dvar4*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar3*var4*dxc[1]*dxc[1]*(1/dr)*(1/dr)+
													 dvar3*dvar4*dxc[1]*dxc1[1]*(1/dr)*(1/dr1)+
													 dvar3*var4*(1/dr)+
													 dvar3*var4*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

									var3zz = var3*ddvar4*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1)+
													 dvar3*dvar4*dxc[2]*dxc1[2]*(1/dr)*(1/dr1)+
													 var3*dvar4*(1/dr1)+
													 var3*dvar4*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
													 ddvar3*var4*dxc[2]*dxc[2]*(1/dr)*(1/dr)+
													 dvar3*dvar4*dxc[2]*dxc1[2]*(1/dr)*(1/dr1)+
													 dvar3*var4*(1/dr)+
													 dvar3*var4*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);
								}					
								
								var5 = 0.50*(var1*var2 + var3*var4);
								// printf("4. came here %d/%d\n",coeff_idx1,numD);
								if (pMAIN_Obj->isgradient==1) {
									var5x = 0.5*(var1x + var3x);
									var5y = 0.5*(var1y + var3y);
									var5z = 0.5*(var1z + var3z);
								}
								if (pMAIN_Obj->ishessian==1) {
									var5xx = 0.5*(var1xx + var3xx);
									var5yy = 0.5*(var1yy + var3yy);
									var5zz = 0.5*(var1zz + var3zz);
								}

								if(pMAIN_Obj->isC23==1){
									var5_C23 = 0.50*(var1*var2*G1_ij + var3*var4*G2_ij);
									if (pMAIN_Obj->isgradient==1) {
										var5x_C23 = 0.5*(var1x*G1_ij + var3x*G2_ij);
										var5y_C23 = 0.5*(var1y*G1_ij + var3y*G2_ij);
										var5z_C23 = 0.5*(var1z*G1_ij + var3z*G2_ij);
									}
									if (pMAIN_Obj->ishessian==1) {
										var5xx_C23 = 0.5*(var1xx*G1_ij + var3xx*G2_ij);
										var5yy_C23 = 0.5*(var1yy*G1_ij + var3yy*G2_ij);
										var5zz_C23 = 0.5*(var1zz*G1_ij + var3zz*G2_ij);
									}
								} 

								m = coeff_idx1*numD + coeff_idx2;
								if (i <= j){
									corr22_mat[i*pMAIN_Obj->nelem + j][m] += var5;
									if (pMAIN_Obj->isgradient==1) {
										corr_derivatives[0][i*pMAIN_Obj->nelem + j][m] += var5x;
										corr_derivatives[1][i*pMAIN_Obj->nelem + j][m] += var5y;
										corr_derivatives[2][i*pMAIN_Obj->nelem + j][m] += var5z;
										if (pMAIN_Obj->ishessian==1) {
											corr_derivatives[3][i*pMAIN_Obj->nelem + j][m] += var5xx;
											corr_derivatives[4][i*pMAIN_Obj->nelem + j][m] += var5yy;
											corr_derivatives[5][i*pMAIN_Obj->nelem + j][m] += var5zz;
										}
											
									} else if (pMAIN_Obj->ishessian==1) {
										corr_derivatives[0][i*pMAIN_Obj->nelem + j][m] += var5xx;
										corr_derivatives[1][i*pMAIN_Obj->nelem + j][m] += var5yy;
										corr_derivatives[2][i*pMAIN_Obj->nelem + j][m] += var5zz;
									}
									

									if(pMAIN_Obj->isC23==1){
										corr23_mat[i*pMAIN_Obj->nelem + j][m] += var5_C23;
										if (pMAIN_Obj->isgradient==1) {
											corr_derivatives_C23[0][i*pMAIN_Obj->nelem + j][m] += var5x_C23;
											corr_derivatives_C23[1][i*pMAIN_Obj->nelem + j][m] += var5y_C23;
											corr_derivatives_C23[2][i*pMAIN_Obj->nelem + j][m] += var5z_C23;
											if (pMAIN_Obj->ishessian==1) {
												corr_derivatives_C23[3][i*pMAIN_Obj->nelem + j][m] += var5xx_C23;
												corr_derivatives_C23[4][i*pMAIN_Obj->nelem + j][m] += var5yy_C23;
												corr_derivatives_C23[5][i*pMAIN_Obj->nelem + j][m] += var5zz_C23;
											}
											
										} else if (pMAIN_Obj->ishessian==1) {
											corr_derivatives_C23[0][i*pMAIN_Obj->nelem + j][m] += var5xx_C23;
											corr_derivatives_C23[1][i*pMAIN_Obj->nelem + j][m] += var5yy_C23;
											corr_derivatives_C23[2][i*pMAIN_Obj->nelem + j][m] += var5zz_C23;
										}
									} 
								} else {
									corr22_mat[j*pMAIN_Obj->nelem + i][m] += var5;
									if (pMAIN_Obj->isgradient==1) {
										corr_derivatives[0][j*pMAIN_Obj->nelem + i][m] += var5x;
										corr_derivatives[1][j*pMAIN_Obj->nelem + i][m] += var5y;
										corr_derivatives[2][j*pMAIN_Obj->nelem + i][m] += var5z;
										if (pMAIN_Obj->ishessian==1) {
											corr_derivatives[3][j*pMAIN_Obj->nelem + i][m] += var5xx;
											corr_derivatives[4][j*pMAIN_Obj->nelem + i][m] += var5yy;
											corr_derivatives[5][j*pMAIN_Obj->nelem + i][m] += var5zz;
										}
											
									} else if (pMAIN_Obj->ishessian==1) {
										corr_derivatives[0][j*pMAIN_Obj->nelem + i][m] += var5xx;
										corr_derivatives[1][j*pMAIN_Obj->nelem + i][m] += var5yy;
										corr_derivatives[2][j*pMAIN_Obj->nelem + i][m] += var5zz;
									}
									if(pMAIN_Obj->isC23==1){
										corr23_mat[j*pMAIN_Obj->nelem + i][m] += var5_C23;
										if (pMAIN_Obj->isgradient==1) {
											corr_derivatives_C23[0][j*pMAIN_Obj->nelem + i][m] += var5x_C23;
											corr_derivatives_C23[1][j*pMAIN_Obj->nelem + i][m] += var5y_C23;
											corr_derivatives_C23[2][j*pMAIN_Obj->nelem + i][m] += var5z_C23;
											if (pMAIN_Obj->ishessian==1) {
												corr_derivatives_C23[3][j*pMAIN_Obj->nelem + i][m] += var5xx_C23;
												corr_derivatives_C23[4][j*pMAIN_Obj->nelem + i][m] += var5yy_C23;
												corr_derivatives_C23[5][j*pMAIN_Obj->nelem + i][m] += var5zz_C23;
											}
											
										} else if (pMAIN_Obj->ishessian==1) {
											corr_derivatives_C23[0][j*pMAIN_Obj->nelem + i][m] += var5xx_C23;
											corr_derivatives_C23[1][j*pMAIN_Obj->nelem + i][m] += var5yy_C23;
											corr_derivatives_C23[2][j*pMAIN_Obj->nelem + i][m] += var5zz_C23;
										}
									} 
								}
								// printf("5. came here %d/%d\n",coeff_idx1,numD);
							}
						}
					}
				}

				for (int dd = 0; dd<der_terms+1; dd++){
					coeff_track = 1 + pMAIN_Obj->numC11_coeffs;

					for (i = 0; i < pMAIN_Obj->nelem; i++){
						for (j = i; j < pMAIN_Obj->nelem; j++){
							for (coeff_idx1 = 0; coeff_idx1 < numD; coeff_idx1++){
								for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD; coeff_idx2++){
									
									m = coeff_idx1*numD + coeff_idx2;
									
									if (dd==0){
										var1 = corr22_mat[i*pMAIN_Obj->nelem + j][m];
									} else {
										var1 = corr_derivatives[dd-1][i*pMAIN_Obj->nelem + j][m];
									}	
									pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] = var1;
									coeff_track=coeff_track+1;
								}
								
							}
							
						}
					}

					if(pMAIN_Obj->isC23==1) {
						for (i = 0; i < pMAIN_Obj->nelem; i++){
							for (j = i; j < pMAIN_Obj->nelem; j++){
								for (coeff_idx1 = 0; coeff_idx1 < numD; coeff_idx1++){
									for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD; coeff_idx2++){
										m = coeff_idx1*numD + coeff_idx2;;
										if (dd==0) {
											var1 = corr23_mat[i*pMAIN_Obj->nelem + j][m];
										} else {
											var1 = corr_derivatives_C23[dd-1][i*pMAIN_Obj->nelem + j][m];
										}
										
										pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] = var1;
										
										coeff_track=coeff_track+1;
									}
								}
							}
						}
					}
					pMAIN_Obj->count2 = pMAIN_Obj->count2+1	;
				}							
			}
			
		}
		// printf("%d/%d\n",grid_k,NDz);
	}

	if (pMAIN_Obj->isgradient == 1 || pMAIN_Obj->ishessian == 1) {
		for(i=0; i<der_terms ; i++){
			for(j=0; j<pMAIN_Obj->nelem * pMAIN_Obj->nelem ; j++){
				free(corr_derivatives[i][j]) ;
				if(pMAIN_Obj->isC23==1) free(corr_derivatives_C23[i][j]);
			}
		}
		for(i=0; i<der_terms ; i++) {
			free(corr_derivatives[i]) ;
			if(pMAIN_Obj->isC23==1) free(corr_derivatives_C23[i]);
		}
		free(corr_derivatives) ;
		if(pMAIN_Obj->isC23==1) free(corr_derivatives_C23);
	}

	for(i=0; i<3 ; i++){
		free(basis_matrixG1[i]);
	}
	free(basis_matrixG1);

	for(i=0;i<4;i++) {
		free(pos_store[i]);
	}
	free(pos_store);



	free(dist_ij);

	for (i = 0; i < file_param_Obj->natoms * file_param_Obj->natoms; i++)
		free(Gij_struct[i].mat);
	free(Gij_struct);


	for (i=0; i<pMAIN_Obj->nelem * pMAIN_Obj->nelem; i++){
		if(pMAIN_Obj->isC23==1) free(corr23_mat[i]);
		free(corr22_mat[i]);
	}

	if(pMAIN_Obj->isC23==1) free(corr23_mat);
	free(corr22_mat);
	
	for (int bas =0; bas < 2*(pMAIN_Obj->num_bases_C22-1); bas++){
		free(cheb_bases_dr23[bas]);
	}
	for (int bas =0; bas < pMAIN_Obj->num_bases_C22; bas++){
		free(cheb_bases_dr22[bas]);
	}
	free(cheb_bases_dr23);
	free(cheb_bases_dr22);

}

void Calc_C33_terms(main_Obj *pMAIN_Obj, DenFile_paral_Obj *paral_Obj, DenFile_param_Obj *file_param_Obj, int file_index) {
	int grid_i, grid_j, grid_k;
	double box_volume, dx_r[3], dxc[3],dxc1[3],dxc2[3];
	double **corr33_mat, **corr34_mat, ***der_corr33_mat, ***der_corr34_mat, *dist_ij;
	int idx_1D, chem_idx;
	int i, j ,k, m, n, i_elem, i_Gauss, p;
	int coeff_idx1,coeff_idx2,coeff_idx3,atom_idx1,atom_idx,atom_idx2,iv1,iv2,iv3,iv4,coeff_track;
	double var1, var2,var3,var4,var5,var5_C23,sigma1_ij,G1_ij,G1_ik,G1_jk;
	double dvar1, dvar2, dvar3, ddvar1, ddvar2, ddvar3;
	double dxr[3],dxr1[3],dxr2[3];
	double dr, dr1, dr2, dr3, dr_ij, dr_ik, dr_jk, temp0, temp1, temp2;
	int nproc, rank, der_terms, Poly_order_ij;
	double t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16, t_c33, t_c34;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	box_volume = volume_Hmat(file_param_Obj->LatVec);
	int NDx, NDy, NDz;
	int Gauss_index, Poly_order,Gauss_index_1, Poly_order_1,Gauss_index_2, Poly_order_2,Gauss_index_3, Poly_order_3;
	double var1x, var1y, var1z, var3x, var3y, var3z, var1xx, var1yy, var1zz, var4_x, var4_y, var4_z, 
	var4_xx, var4_yy, var4_zz, var5_x, var5_y, var5_z, var5_xx, var5_yy, var5_zz;
	double atom1_x, atom1_y, atom1_z, atom2_x, atom2_y, atom2_z, atom3_x, atom3_y, atom3_z;
	double **basis_matrixG1, func_out[3];
	int numD, numD3;
	double maxdist;

	maxdist = sqrt(file_param_Obj->Lx * file_param_Obj->Lx + 
								file_param_Obj->Ly * file_param_Obj->Ly + 
								file_param_Obj->Lz * file_param_Obj->Lz);
	
	// numD = pMAIN_Obj->num_bases;
	numD3 = pMAIN_Obj->num_bases_C33;


	double **cheb_bases_dr34, **cheb_bases_dr33;

	cheb_bases_dr34 = (double **) malloc((2*(pMAIN_Obj->num_bases_C33-1)+1)*sizeof(double*));
	cheb_bases_dr33 = (double **) malloc(pMAIN_Obj->num_bases_C33*sizeof(double*));

	for (int bas =0; bas < 2*(pMAIN_Obj->num_bases_C33-1)+1; bas++){
		cheb_bases_dr34[bas] = (double *) malloc(3*sizeof(double));
	}
	for (int bas =0; bas < pMAIN_Obj->num_bases_C33; bas++){
		cheb_bases_dr33[bas] = (double *) malloc(3*sizeof(double));
	}
	double **bases_ij;
	bases_ij = (double **)malloc(file_param_Obj->natoms * file_param_Obj->natoms * sizeof(double*));
	for (int bas=0; bas <file_param_Obj->natoms * file_param_Obj->natoms; bas++){
		bases_ij[bas] = (double *)malloc((2*(pMAIN_Obj->num_bases_C33-1)+1)*sizeof(double));
	}


	// file_param_Obj->basis_matrixG = (double *)realloc(file_param_Obj->basis_matrixG, file_param_Obj->natoms*pMAIN_Obj->numG*sizeof(double));
	basis_matrixG1 = (double **)malloc(3*sizeof(double*));
	for(i=0; i<3 ; i++){
		basis_matrixG1[i] = (double *)malloc(file_param_Obj->natoms*numD3*sizeof(double));
	}

	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1)
		der_terms = 3;
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1)
		der_terms = 6;
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0)
		der_terms = 0;  

	corr33_mat = (double**) malloc (pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double*));
	if(pMAIN_Obj->isC34==1) corr34_mat = (double**) malloc (pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double*));
	for (i = 0; i < pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem; i++){
			corr33_mat[i] = (double*) malloc (numD3 * numD3 * numD3 * sizeof (double));
			if(pMAIN_Obj->isC34==1) corr34_mat[i] = (double*) malloc (numD3 * numD3 * numD3 * sizeof (double));
	}

	if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1){
		der_corr33_mat = (double***) malloc (der_terms * sizeof(double**));
		if(pMAIN_Obj->isC34==1) der_corr34_mat = (double***) malloc (der_terms * sizeof(double**));
		for(i = 0; i<der_terms; i++){
			der_corr33_mat[i] = (double**) malloc (pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double*));
			if(pMAIN_Obj->isC34==1) der_corr34_mat[i] = (double**) malloc (pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem * sizeof(double*));
			for(j = 0; j<pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem; j++) {
				der_corr33_mat[i][j] = (double*) malloc (numD3 * numD3 * numD3 * sizeof (double));
				if(pMAIN_Obj->isC34==1) der_corr34_mat[i][j] = (double*) malloc (numD3 * numD3 * numD3 * sizeof (double));
			}
		}
	}
	
		
	iv1 = pMAIN_Obj->nelem * pMAIN_Obj->nelem;
  iv2 = pMAIN_Obj->nelem;
  iv3 = numD3 * numD3;
  iv4 = numD3;


	NDx = paral_Obj->node_num_x ;
	NDy = paral_Obj->node_num_y ;
	NDz = paral_Obj->node_num_z ;

	dist_ij = (double *) malloc(file_param_Obj->natoms * file_param_Obj->natoms * sizeof(double));
	for (atom_idx = 0; atom_idx < file_param_Obj->natoms; atom_idx++){
		for (atom_idx1 = (atom_idx + 1); atom_idx1 < file_param_Obj->natoms; atom_idx1++){
			dxr[0] = file_param_Obj->atom_pos[3*atom_idx + 0] - file_param_Obj->atom_pos[3*atom_idx1 + 0]; // dimensionless 
			dxr[1] = file_param_Obj->atom_pos[3*atom_idx + 1] - file_param_Obj->atom_pos[3*atom_idx1 + 1];
			dxr[2] = file_param_Obj->atom_pos[3*atom_idx + 2] - file_param_Obj->atom_pos[3*atom_idx1 + 2];
			
			dxr[0] = checkPBC_Shifted(dxr[0]); // PBC, i.e. -0.50 < dxr < 0.50 
	    dxr[1] = checkPBC_Shifted(dxr[1]); // PBC, i.e. -0.50 < dxr < 0.50 
			dxr[2] = checkPBC_Shifted(dxr[2]); // PBC, i.e. -0.50 < dxr < 0.50 


			V3M3mulV3(dxr, file_param_Obj->LatVec, dxc);
			dr = V3NormS(dxc);
			
			func_cheb(dr, 2*(pMAIN_Obj->num_bases_C33-1)+1, cheb_bases_dr34, maxdist,pMAIN_Obj->Rcut);

			dist_ij[atom_idx * file_param_Obj->natoms + atom_idx1] = dr;
			dist_ij[atom_idx1 * file_param_Obj->natoms + atom_idx] = dr;

			for (int ff = 0; ff<2*(pMAIN_Obj->num_bases_C33-1)+1; ff++){
				bases_ij[atom_idx * file_param_Obj->natoms + atom_idx1][ff] = cheb_bases_dr34[ff][0];
			}
		}
	}


		
	if (file_index==0){
		pMAIN_Obj->count2=0;
	} else {
		pMAIN_Obj->count2=0;
		for(i=0;i<file_index;i++) {
			pMAIN_Obj->count2 = pMAIN_Obj->count2 + pMAIN_Obj->node_num_x[i] *pMAIN_Obj->node_num_y[i]
			 *pMAIN_Obj->node_num_z[i]*(1+der_terms);
		}
	}
	// if (pMAIN_Obj->isC34  == 1 && rank==0) printf("C34 is present\n");
	// if (pMAIN_Obj->isC34  != 1 && rank==0) printf("C34 is not present\n");
	double **pos_store;
	pos_store = (double **)malloc(4 * sizeof(double *));
	for(i=0;i<4;i++) {
		pos_store[i] = (double *)malloc(file_param_Obj->natoms*sizeof(double));
	}
	// printf("natoms %d\n",file_param_Obj->natoms);

	for (int grid_k = 0; grid_k <NDz; grid_k++) {
		 // if (rank==0) printf("grid_k %d/%d\n",grid_k,NDz);
		for (int grid_j = 0; grid_j < NDy; grid_j++) {
			// if (rank==0) printf("grid_j %d/%d\n",grid_j,NDy);
			for (int grid_i = 0; grid_i < NDx; grid_i++) {
				// if (rank==0) printf("grid_i %d/%d\n",grid_i,NDx);

				iv1 = pMAIN_Obj->nelem * pMAIN_Obj->nelem;
    		iv2 = pMAIN_Obj->nelem;
    		iv3 = numD3 * numD3;
    		iv4 = numD3;

    		temp0 = (float) (grid_i+paral_Obj->node_begin_x) * ((float) file_param_Obj->chg_step_jumpX / (float) file_param_Obj->Nx);
				temp1 = (float) (grid_j+paral_Obj->node_begin_y) * ((float) file_param_Obj->chg_step_jumpY / (float) file_param_Obj->Ny);
				temp2 = (float) (grid_k+paral_Obj->node_begin_z) * ((float) file_param_Obj->chg_step_jumpZ / (float) file_param_Obj->Nz);

    			for (atom_idx = 0; atom_idx < file_param_Obj->natoms ; atom_idx++) {
						dxr[0] = temp0 - file_param_Obj->atom_pos[3*atom_idx+ 0];
						dxr[1] = temp1 - file_param_Obj->atom_pos[3*atom_idx+ 1];
						dxr[2] = temp2 - file_param_Obj->atom_pos[3*atom_idx+ 2];
						dxr[0] = checkPBC_Shifted(dxr[0]); // PBC, i.e. -0.50 < dxr < 0.50 
	          dxr[1] = checkPBC_Shifted(dxr[1]); // PBC, i.e. -0.50 < dxr < 0.50 
			  		dxr[2] = checkPBC_Shifted(dxr[2]); // PBC, i.e. -0.50 < dxr < 0.50 

			  		V3M3mulV3(dxr, file_param_Obj->LatVec, dxc);
			  		dr = V3NormS(dxc);
			  		func_cheb(dr, pMAIN_Obj->num_bases_C33, cheb_bases_dr33, maxdist,pMAIN_Obj->Rcut);

			  		// if (grid_k==0 && grid_j==0 && grid_i==0 && rank==0){
			  		// 	for (int mm=0;mm<pMAIN_Obj->num_bases;mm++){
			  		// 		printf("%1f %1f %1f\n",cheb_bases_dr33[mm][0],cheb_bases_dr33[mm][1],cheb_bases_dr33[mm][2]);
			  		// 	}
			  		// }
			  		
			  		pos_store[0][atom_idx] = dxc[0];
			  		pos_store[1][atom_idx] = dxc[1];
			  		pos_store[2][atom_idx] = dxc[2];
			  		pos_store[3][atom_idx] = dr; 
			  		chem_idx=find_chem_idx(atom_idx, file_param_Obj, pMAIN_Obj);		
			  		for (i_Gauss = 0; i_Gauss < pMAIN_Obj->num_bases_C33; i_Gauss++){
			  			if (pMAIN_Obj->ischeb!=1){
			  				Gauss_index =  i_Gauss / (1+pMAIN_Obj->poly_order_C33);
				  			Poly_order =  i_Gauss % (1+pMAIN_Obj->poly_order_C33);
				  			func_GTO(pMAIN_Obj->Gaussian_Width_C33[Gauss_index][chem_idx], dr, Poly_order, func_out, pMAIN_Obj->ischeb, maxdist);
			  			} else {
			  				func_out[0] = cheb_bases_dr33[i_Gauss][0];
			  				func_out[1] = cheb_bases_dr33[i_Gauss][1];
			  				func_out[2] = cheb_bases_dr33[i_Gauss][2];
			  			}
			  			basis_matrixG1[0][atom_idx*numD3 + i_Gauss] = func_out[0];
			  			basis_matrixG1[1][atom_idx*numD3 + i_Gauss] = func_out[1];
			  			basis_matrixG1[2][atom_idx*numD3 + i_Gauss] = func_out[2];
			  			// printf("%1f ",func_out[0]);
			  		}
				}



				// printf("1. came here i,j,k %d %d %d \n",grid_i, grid_j, grid_k);
				// if (rank==0 && grid_i == 2 && grid_j == 2 && grid_k == 0) printf("came here\n");
				for (i = 0; i < pMAIN_Obj->nelem; i++)
					for (j = 0; j < pMAIN_Obj->nelem; j++)
						for (k = 0; k < pMAIN_Obj->nelem; k++)
							for (m = 0; m < numD3; m++)
								for (n = 0; n < numD3; n++)
									for (p = 0; p < numD3; p++){
										
										corr33_mat[i*iv1 + j*iv2 + k][m*iv3 + n*iv4 + p] = 0.00;
										
										if(pMAIN_Obj->isC34==1) corr34_mat[i*iv1 + j*iv2 + k][m*iv3 + n*iv4 + p] = 0.00;
										
										for (int xx=0; xx<der_terms; xx++){
											der_corr33_mat[xx][i*iv1 + j*iv2 + k][m*iv3 + n*iv4 + p] = 0.00;
											if(pMAIN_Obj->isC34==1) der_corr34_mat[xx][i*iv1 + j*iv2 + k][m*iv3 + n*iv4 + p] = 0.00;
										}					
									}
				// printf("2. came here i,j,k %d %d %d \n",grid_i, grid_j, grid_k);
				// t_c34=0;
				// t_c33=0;

				for (coeff_idx1 = 0; coeff_idx1 < numD3; coeff_idx1++)
					for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD3; coeff_idx2++)
						for (coeff_idx3 = coeff_idx2; coeff_idx3 < numD3; coeff_idx3++){
							for (atom_idx = 0; atom_idx < file_param_Obj->natoms; atom_idx++)
								for (atom_idx1 = (atom_idx + 1); atom_idx1 < file_param_Obj->natoms; atom_idx1++)
									for (atom_idx2 = (atom_idx1 + 1); atom_idx2 < file_param_Obj->natoms; atom_idx2++) {
										i = find_chem_idx(atom_idx,file_param_Obj, pMAIN_Obj);
										j = find_chem_idx(atom_idx1,file_param_Obj, pMAIN_Obj);
										k = find_chem_idx(atom_idx2,file_param_Obj, pMAIN_Obj);
										Gauss_index_1 =  coeff_idx1/(1+pMAIN_Obj->poly_order_C33);
										Gauss_index_2 =  coeff_idx2/(1+pMAIN_Obj->poly_order_C33);
										Gauss_index_3 =  coeff_idx3/(1+pMAIN_Obj->poly_order_C33);

										Poly_order_1 =   coeff_idx1%(1+pMAIN_Obj->poly_order_C33);
										Poly_order_2 =   coeff_idx2%(1+pMAIN_Obj->poly_order_C33);
										Poly_order_3 =   coeff_idx3%(1+pMAIN_Obj->poly_order_C33);


										if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1){
											
									  	dxc[0] = pos_store[0][atom_idx];
											dxc[1] = pos_store[1][atom_idx];
											dxc[2] = pos_store[2][atom_idx];

											dxc1[0] = pos_store[0][atom_idx1];
											dxc1[1] = pos_store[1][atom_idx1];
											dxc1[2] = pos_store[2][atom_idx1];

											dxc2[0] = pos_store[0][atom_idx2];
											dxc2[1] = pos_store[1][atom_idx2];
											dxc2[2] = pos_store[2][atom_idx2];

											dr = pos_store[3][atom_idx];
											dr1 = pos_store[3][atom_idx1];
											dr2 = pos_store[3][atom_idx2];
									  		

										}
										// t1 = MPI_Wtime();
										if (pMAIN_Obj->isC34  == 1)
									    {
									      dr_ij = dist_ij[atom_idx*file_param_Obj->natoms + atom_idx1];
									      dr_ik = dist_ij[atom_idx*file_param_Obj->natoms + atom_idx2];
									      dr_jk = dist_ij[atom_idx1*file_param_Obj->natoms + atom_idx2];
									    }
									    // t2 = MPI_Wtime();
									    // t_c34 = t_c34+t2-t1;
										  var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx1];
				  						var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx2];	
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx3];
				  						
				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx1];
				  						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx2];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx3];
				  						
				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx1];
				  						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx2];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx3];
				  						var4 = var1*var2*var3;


				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
				  							var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[0]*(1/dr);

				  							var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[1]*(1/dr);

				  							var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[2]*(1/dr);


											var4_x = var1x;
											var4_y = var1y;
											var4_z = var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);

												var4_xx = var1xx;
												var4_yy = var1yy;
												var4_zz = var1zz;
											}
											
										}
				  						// t3 = MPI_Wtime();
				  						// t_c33 = t_c33+t3-t2;
				  						if (pMAIN_Obj->isC34  == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j]);
										      Poly_order_ij = Poly_order_1+Poly_order_2;
										      func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	} else {
									    		Poly_order_ij = coeff_idx1+coeff_idx2;
									    		// func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
									    		G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_3;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										       G1_ik = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx1+coeff_idx3;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
									      	G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      // G1_ik = func_out[0];
									      // G1_ik = func_Gaussian(sigma1_ij, dr_ik);
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_2+Poly_order_3;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx2+coeff_idx3;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										      
									      
									      // G1_jk = func_Gaussian(sigma1_ij, dr_jk);
									      var5 = var1*var2*var3*G1_ij*G1_ik*G1_jk;
									      if (pMAIN_Obj->isgradient==1){
									      	var5_x = var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y = var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z = var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx = var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy = var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz = var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    }
									    // t4 = MPI_Wtime();
									    // t_c34=t_c34+(t4-t3);

				  						var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx1];
				  						var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx3];
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx2];

				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx1];
				  						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx3];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx2];

				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx1];
				  						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx3];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx2];
				  						
				  						var4 += var1*var2*var3;

				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
											var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
											+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
											+ dvar1*var2*var3*dxc[0]*(1/dr);

											var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
											+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
											+ dvar1*var2*var3*dxc[1]*(1/dr);

											var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
											+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
											+ dvar1*var2*var3*dxc[2]*(1/dr);

											var4_x += var1x;
											var4_y += var1y;
											var4_z += var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);

												var4_xx += var1xx;
												var4_yy += var1yy;
												var4_zz += var1zz;
											}
											
										}
				  						// t5 = MPI_Wtime();
				  						// t_c33=t_c33+(t5-t4);
				  						if (pMAIN_Obj->isC34  == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j]);
										      Poly_order_ij = Poly_order_1+Poly_order_3;
										      func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	} else {
									    		Poly_order_ij = coeff_idx1+coeff_idx3;
										      // func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_2;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = func_out[0];
									      } else{
									      	Poly_order_ij = coeff_idx1+coeff_idx2;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      

									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_2+Poly_order_3;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx2+coeff_idx3;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										  	
									      var5 += var1*var2*var3*G1_ij*G1_ik*G1_jk;
									      if (pMAIN_Obj->isgradient==1){
									      	var5_x += var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y += var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z += var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx += var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy += var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz += var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    } 


									    // t6 = MPI_Wtime();
									    // t_c34=t_c34+(t6-t5);
				  						var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx2];
				 						  var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx3];
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx1];

				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx2];
				 						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx3];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx1];

				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx2];
				 						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx3];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx1];

				  						var4 += var1*var2*var3;
				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
				  							var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[0]*(1/dr);

				  							var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[1]*(1/dr);

				  							var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[2]*(1/dr);


											var4_x += var1x;
											var4_y += var1y;
											var4_z += var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);


												var4_xx += var1xx;
												var4_yy += var1yy;
												var4_zz += var1zz;
											}
											
										}
				  						// t7 = MPI_Wtime();
				  						// t_c33=t_c33+(t7-t6);
				  						if (pMAIN_Obj->isC34  == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j]);
										      Poly_order_ij = Poly_order_2+Poly_order_3;
										      func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	} else {
									    		Poly_order_ij = coeff_idx2 + coeff_idx3;
										      // func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
										      
									      

									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_2;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx1 + coeff_idx2;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										      
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][k]);
										      Poly_order_ij = Poly_order_3+Poly_order_1;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
									      	G1_jk = func_out[0];
									      } else{
									      	Poly_order_ij = coeff_idx3 + coeff_idx1;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										      
									      

									      var5 += var1*var2*var3*G1_ij*G1_ik*G1_jk;

									      if (pMAIN_Obj->isgradient==1){
									      	var5_x += var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y += var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z += var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx += var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy += var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz += var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    } // end_if (c34 > 0)
									    // t8 = MPI_Wtime();
									    // t_c34=t_c34+(t8-t7);

				  						var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx2];
				  						var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx1];
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx3];

				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx2];
				  						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx1];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx3];

				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx2];
				  						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx1];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx3];

				  						var4 += var1*var2*var3;

				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
				  							var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[0]*(1/dr);

				  							var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[1]*(1/dr);

				  							var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[2]*(1/dr);


											var4_x += var1x;
											var4_y += var1y;
											var4_z += var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);


												var4_xx += var1xx;
												var4_yy += var1yy;
												var4_zz += var1zz;
											}
											
										}
				  						// t9 = MPI_Wtime();
				  						// t_c33=t_c33+(t9-t8);

				  						if (pMAIN_Obj->isC34  == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j]);
										      Poly_order_ij = Poly_order_2+Poly_order_1;
										      func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	}else {
									    		Poly_order_ij = coeff_idx2 + coeff_idx1;
										      // func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
									      
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_2+Poly_order_3;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx2 + coeff_idx3;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										      
									      
									      // G1_ik = func_Gaussian(sigma1_ij, dr_ik);
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_3;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx1 + coeff_idx3;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
										      
									      

									      var5 += var1*var2*var3*G1_ij*G1_ik*G1_jk;

									      if (pMAIN_Obj->isgradient==1){
									      	var5_x += var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y += var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z += var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx += var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy += var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz += var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    } // end_if (c34 > 0)
									    // t10 = MPI_Wtime();
									    // t_c34=t_c34+(t10-t9);
				  						var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx3];
				  						var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx1];
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx2];

				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx3];
				  						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx1];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx2];

				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx3];
				  						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx1];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx2];

				  						var4 += var1*var2*var3;

				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
				  							var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[0]*(1/dr);

				  							var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[1]*(1/dr);

				  							var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[2]*(1/dr);

											var4_x += var1x;
											var4_y += var1y;
											var4_z += var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);


												var4_xx += var1xx;
												var4_yy += var1yy;
												var4_zz += var1zz;
											}
											
										}
				  						// t11 = MPI_Wtime();
				  						// t_c33=t_c33+(t11-t10);
				  						if (pMAIN_Obj->isC34  == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j]);
										      Poly_order_ij = Poly_order_3+Poly_order_1;
										      func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	} else {
									    		Poly_order_ij = coeff_idx3 + coeff_idx1;
										      // func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
									      
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k]);
										      Poly_order_ij = Poly_order_3+Poly_order_2;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx3 + coeff_idx2;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      
									      // G1_ik = func_Gaussian(sigma1_ij, dr_ik);
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_2;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx1 + coeff_idx2;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      

									      var5 += var1*var2*var3*G1_ij*G1_ik*G1_jk;

									      if (pMAIN_Obj->isgradient==1){
									      	var5_x += var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y += var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z += var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx += var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy += var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz += var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    } // end_if (c34 > 0)
									    // t12 = MPI_Wtime();
									    // t_c34=t_c34+(t12-t11);
				  						var1 = basis_matrixG1[0][atom_idx*numD3 + coeff_idx3];
				  						var2 = basis_matrixG1[0][atom_idx1*numD3 + coeff_idx2];
				  						var3 = basis_matrixG1[0][atom_idx2*numD3 + coeff_idx1];

				  						dvar1 = basis_matrixG1[1][atom_idx*numD3 + coeff_idx3];
				  						dvar2 = basis_matrixG1[1][atom_idx1*numD3 + coeff_idx2];
				  						dvar3 = basis_matrixG1[1][atom_idx2*numD3 + coeff_idx1];

				  						ddvar1 = basis_matrixG1[2][atom_idx*numD3 + coeff_idx3];
				  						ddvar2 = basis_matrixG1[2][atom_idx1*numD3 + coeff_idx2];
				  						ddvar3 = basis_matrixG1[2][atom_idx2*numD3 + coeff_idx1];

				  						var4 += var1*var2*var3;


				  						if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1) {
				  							var1x = var1*var2*dvar3*dxc2[0]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[0]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[0]*(1/dr);

				  							var1y = var1*var2*dvar3*dxc2[1]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[1]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[1]*(1/dr);

				  							var1z = var1*var2*dvar3*dxc2[2]*(1/dr2) 
				  							+ var1*dvar2*var3*dxc1[2]*(1/dr1) 
				  							+ dvar1*var2*var3*dxc[2]*(1/dr);


											var4_x += var1x;
											var4_y += var1y;
											var4_z += var1z;
											if (pMAIN_Obj->ishessian==1) {
												var1xx = var1*var2*ddvar3*dxc2[0]*dxc2[0]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[0]*dxc2[0]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[0]*dxc2[0]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[0]*dxc[0]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[0]*dxc1[0]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[0]*dxc1[0]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[0]*dxc2[0]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[0]*dxc[0]*(1/dr)*(-1/dr)*(1/dr);

												var1yy = var1*var2*ddvar3*dxc2[1]*dxc2[1]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[1]*dxc2[1]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[1]*dxc2[1]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[1]*dxc[1]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[1]*dxc1[1]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[1]*dxc1[1]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[1]*dxc2[1]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[1]*dxc[1]*(1/dr)*(-1/dr)*(1/dr);

												var1zz = var1*var2*ddvar3*dxc2[2]*dxc2[2]*(1/dr2)*(1/dr2)+
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												var1*var2*dvar3*(1/dr2)+
												var1*var2*dvar3*dxc2[2]*dxc2[2]*(1/dr2)*(-1/dr2)*(1/dr2) +
												var1*ddvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(1/dr1) +
												var1*dvar2*dvar3*dxc1[2]*dxc2[2]*(1/dr1)*(1/dr2) +
												dvar1*dvar2*var3*dxc1[2]*dxc[2]*(1/dr1)*(1/dr) +
												var1*dvar2*var3*(1/dr1) +
												var1*dvar2*var3*dxc1[2]*dxc1[2]*(1/dr1)*(-1/dr1)*(1/dr1)+
												ddvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(1/dr) +
												dvar1*dvar2*var3*dxc[2]*dxc1[2]*(1/dr)*(1/dr1) +
												dvar1*var2*dvar3*dxc[2]*dxc2[2]*(1/dr)*(1/dr2) +
												dvar1*var2*var3*(1/dr)+
												dvar1*var2*var3*dxc[2]*dxc[2]*(1/dr)*(-1/dr)*(1/dr);
	
												var4_xx += var1xx;
												var4_yy += var1yy;
												var4_zz += var1zz;
											}
											
										}
				  						// t13 = MPI_Wtime();
				  						// t_c33=t_c33+(t13-t12);

				  						if (pMAIN_Obj->isC34 == 1)
									    {
									    	if (pMAIN_Obj->ischeb!=1){
									    		sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j]);
										      Poly_order_ij = Poly_order_3+Poly_order_2;
										      func_GTO(sigma1_ij, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = func_out[0];
									    	} else {
									    		Poly_order_ij = coeff_idx3 + coeff_idx2;
										      // func_GTO(0.0, dr_ij, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ij = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx1][Poly_order_ij];
									    	}
										      
									      
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][i] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_3][k]);
										      Poly_order_ij = Poly_order_1+Poly_order_3;
										      func_GTO(sigma1_ij, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = func_out[0];
									      } else{
									      	Poly_order_ij = coeff_idx1 + coeff_idx3;
										      // func_GTO(0.0, dr_ik, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_ik = bases_ij[atom_idx*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      
									      // G1_ik = func_Gaussian(sigma1_ij, dr_ik);
									      
									      if (pMAIN_Obj->ischeb!=1){
									      	sigma1_ij = pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j]*pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][k];
										      sigma1_ij = sigma1_ij/(pMAIN_Obj->Gaussian_Width_C33[Gauss_index_2][j] + pMAIN_Obj->Gaussian_Width_C33[Gauss_index_1][k]);
										      Poly_order_ij = Poly_order_2+Poly_order_1;
										      func_GTO(sigma1_ij, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = func_out[0];
									      } else {
									      	Poly_order_ij = coeff_idx2+coeff_idx1;
										      // func_GTO(0.0, dr_jk, Poly_order_ij, func_out, pMAIN_Obj->ischeb, maxdist);
										      G1_jk = bases_ij[atom_idx1*file_param_Obj->natoms + atom_idx2][Poly_order_ij];
									      }
									      
									      

									      var5 += var1*var2*var3*G1_ij*G1_ik*G1_jk;

									      if (pMAIN_Obj->isgradient==1){
									      	var5_x += var1x*G1_ij*G1_ik*G1_jk;
									      	var5_y += var1y*G1_ij*G1_ik*G1_jk;
									      	var5_z += var1z*G1_ij*G1_ik*G1_jk;
									      }
									      if (pMAIN_Obj->ishessian==1){
									      	var5_xx += var1xx*G1_ij*G1_ik*G1_jk;
									      	var5_yy += var1yy*G1_ij*G1_ik*G1_jk;
									      	var5_zz += var1zz*G1_ij*G1_ik*G1_jk;
									      }
									    } // end_if (c34 > 0)
									    // t14 = MPI_Wtime();
									    // t_c34=t_c34+(t14-t13);

				  						iv3 = numD3 * numD3 * coeff_idx1;
				  						iv3 += numD3 * coeff_idx2 + coeff_idx3;
				  
				  						iv1 = pMAIN_Obj->nelem;
				  						iv2 = pMAIN_Obj->nelem * pMAIN_Obj->nelem;
				  						// printf("came here 1\n");
				  					    if ((i <= j) && (j <= k)){
									    	corr33_mat[i*iv2 + j*iv1 + k][iv3] += var4/6.0;
									    	
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][i*iv2 + j*iv1 + k][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][i*iv2 + j*iv1 + k][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][i*iv2 + j*iv1 + k][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][i*iv2 + j*iv1 + k][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][i*iv2 + j*iv1 + k][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][i*iv2 + j*iv1 + k][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][i*iv2 + j*iv1 + k][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][i*iv2 + j*iv1 + k][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][i*iv2 + j*iv1 + k][iv3]+= var4_zz/6.0;
									    	}
				  					    }
									  	else if ((i < k) && (k < j)){
									    	corr33_mat[i*iv2 + k*iv1 + j][iv3] += var4/6.0;
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][i*iv2 + k*iv1 + j][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][i*iv2 + k*iv1 + j][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][i*iv2 + k*iv1 + j][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][i*iv2 + k*iv1 + j][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][i*iv2 + k*iv1 + j][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][i*iv2 + k*iv1 + j][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][i*iv2 + k*iv1 + j][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][i*iv2 + k*iv1 + j][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][i*iv2 + k*iv1 + j][iv3]+= var4_zz/6.0;
									    	}
									  	}
									  	else if ((j < k) && (k < i)){
									    	corr33_mat[j*iv2 + k*iv1 + i][iv3] += var4/6.0;
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][j*iv2 + k*iv1 + i][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][j*iv2 + k*iv1 + i][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][j*iv2 + k*iv1 + i][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][j*iv2 + k*iv1 + i][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][j*iv2 + k*iv1 + i][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][j*iv2 + k*iv1 + i][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][j*iv2 + k*iv1 + i][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][j*iv2 + k*iv1 + i][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][j*iv2 + k*iv1 + i][iv3]+= var4_zz/6.0;
									    	}
									  	}
									  	else if ((j < i) && (i < k)){
									    	corr33_mat[j*iv2 + i*iv1 + k][iv3] += var4/6.0;
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][j*iv2 + i*iv1 + k][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][j*iv2 + i*iv1 + k][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][j*iv2 + i*iv1 + k][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][j*iv2 + i*iv1 + k][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][j*iv2 + i*iv1 + k][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][j*iv2 + i*iv1 + k][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][j*iv2 + i*iv1 + k][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][j*iv2 + i*iv1 + k][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][j*iv2 + i*iv1 + k][iv3]+= var4_zz/6.0;
									    	}
									  	}
									  	else if ((k < j) && (j < i)){
									    	corr33_mat[k*iv2 + j*iv1 + i][iv3] += var4/6.0;
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][k*iv2 + j*iv1 + i][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][k*iv2 + j*iv1 + i][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][k*iv2 + j*iv1 + i][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][k*iv2 + j*iv1 + i][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][k*iv2 + j*iv1 + i][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][k*iv2 + j*iv1 + i][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][k*iv2 + j*iv1 + i][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][k*iv2 + j*iv1 + i][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][k*iv2 + j*iv1 + i][iv3]+= var4_zz/6.0;
									    	}
									  	}
									  	else if ((k < i) && (i < j)){
									    	corr33_mat[k*iv2 + i*iv1 + j][iv3] += var4/6.0;
									    	if (pMAIN_Obj->isgradient==1){
									    		der_corr33_mat[0][k*iv2 + i*iv1 + j][iv3]+= var4_x/6.0;
									    		der_corr33_mat[1][k*iv2 + i*iv1 + j][iv3]+= var4_y/6.0;
									    		der_corr33_mat[2][k*iv2 + i*iv1 + j][iv3]+= var4_z/6.0;
									    		if (pMAIN_Obj->ishessian==1){
									    			der_corr33_mat[3][k*iv2 + i*iv1 + j][iv3]+= var4_xx/6.0;
									    			der_corr33_mat[4][k*iv2 + i*iv1 + j][iv3]+= var4_yy/6.0;
									    			der_corr33_mat[5][k*iv2 + i*iv1 + j][iv3]+= var4_zz/6.0;
									    		}
									    	} else if (pMAIN_Obj->ishessian==1){
									    		der_corr33_mat[0][k*iv2 + i*iv1 + j][iv3]+= var4_xx/6.0;
									    		der_corr33_mat[1][k*iv2 + i*iv1 + j][iv3]+= var4_yy/6.0;
									    		der_corr33_mat[2][k*iv2 + i*iv1 + j][iv3]+= var4_zz/6.0;
									    	}
									  	}
									    // t15 = MPI_Wtime();
									    // t_c33=t_c33+(t15-t14);
									    if (pMAIN_Obj->isC34 == 1 )
									    {
									      	if ((i <= j) && (j <= k)){
												corr34_mat[i*iv2 + j*iv1 + k][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][i*iv2 + j*iv1 + k][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][i*iv2 + j*iv1 + k][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][i*iv2 + j*iv1 + k][iv3]+= var4_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][i*iv2 + j*iv1 + k][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][i*iv2 + j*iv1 + k][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][i*iv2 + j*iv1 + k][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][i*iv2 + j*iv1 + k][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][i*iv2 + j*iv1 + k][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][i*iv2 + j*iv1 + k][iv3]+= var5_zz/6.0;
										    	}
									      	}
									      	else if ((i < k) && (k < j)){
												corr34_mat[i*iv2 + k*iv1 + j][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][i*iv2 + k*iv1 + j][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][i*iv2 + k*iv1 + j][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][i*iv2 + k*iv1 + j][iv3]+= var5_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][i*iv2 + k*iv1 + j][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][i*iv2 + k*iv1 + j][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][i*iv2 + k*iv1 + j][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][i*iv2 + k*iv1 + j][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][i*iv2 + k*iv1 + j][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][i*iv2 + k*iv1 + j][iv3]+= var5_zz/6.0;
										    	}
									      	}
									      	else if ((j < k) && (k < i)){
												corr34_mat[j*iv2 + k*iv1 + i][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][j*iv2 + k*iv1 + i][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][j*iv2 + k*iv1 + i][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][j*iv2 + k*iv1 + i][iv3]+= var5_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][j*iv2 + k*iv1 + i][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][j*iv2 + k*iv1 + i][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][j*iv2 + k*iv1 + i][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][j*iv2 + k*iv1 + i][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][j*iv2 + k*iv1 + i][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][j*iv2 + k*iv1 + i][iv3]+= var5_zz/6.0;
										    	}
									      	}
									      	else if ((j < i) && (i < k)){
												corr34_mat[j*iv2 + i*iv1 + k][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][j*iv2 + i*iv1 + k][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][j*iv2 + i*iv1 + k][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][j*iv2 + i*iv1 + k][iv3]+= var5_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][j*iv2 + i*iv1 + k][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][j*iv2 + i*iv1 + k][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][j*iv2 + i*iv1 + k][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][j*iv2 + i*iv1 + k][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][j*iv2 + i*iv1 + k][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][j*iv2 + i*iv1 + k][iv3]+= var5_zz/6.0;
										    	}
									      	}
									      	else if ((k < j) && (j < i)){
												corr34_mat[k*iv2 + j*iv1 + i][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][k*iv2 + j*iv1 + i][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][k*iv2 + j*iv1 + i][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][k*iv2 + j*iv1 + i][iv3]+= var5_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][k*iv2 + j*iv1 + i][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][k*iv2 + j*iv1 + i][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][k*iv2 + j*iv1 + i][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][k*iv2 + j*iv1 + i][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][k*iv2 + j*iv1 + i][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][k*iv2 + j*iv1 + i][iv3]+= var5_zz/6.0;
										    	}
									      	}
									      	else if ((k < i) && (i < j)){
												corr34_mat[k*iv2 + i*iv1 + j][iv3] += var5/6.0;
												if (pMAIN_Obj->isgradient==1){
										    		der_corr34_mat[0][k*iv2 + i*iv1 + j][iv3]+= var5_x/6.0;
										    		der_corr34_mat[1][k*iv2 + i*iv1 + j][iv3]+= var5_y/6.0;
										    		der_corr34_mat[2][k*iv2 + i*iv1 + j][iv3]+= var5_z/6.0;
										    		if (pMAIN_Obj->ishessian==1){
										    			der_corr34_mat[3][k*iv2 + i*iv1 + j][iv3]+= var5_xx/6.0;
										    			der_corr34_mat[4][k*iv2 + i*iv1 + j][iv3]+= var5_yy/6.0;
										    			der_corr34_mat[5][k*iv2 + i*iv1 + j][iv3]+= var5_zz/6.0;
										    		}
										    	} else if (pMAIN_Obj->ishessian==1){
										    		der_corr34_mat[0][k*iv2 + i*iv1 + j][iv3]+= var5_xx/6.0;
										    		der_corr34_mat[1][k*iv2 + i*iv1 + j][iv3]+= var5_yy/6.0;
										    		der_corr34_mat[2][k*iv2 + i*iv1 + j][iv3]+= var5_zz/6.0;
										    	}
									      	}
									    }
									    // t16 = MPI_Wtime();
									    // t_c34=t_c34+(t16-t15);
									    // 
									}
						}

				
				for (int dd =0; dd<der_terms+1; dd++){
					coeff_track = 1 + pMAIN_Obj->numC11_coeffs + pMAIN_Obj->numC22_coeffs
								+ pMAIN_Obj->numC23_coeffs;
					for (i = 0; i < pMAIN_Obj->nelem; i++)
						for (j = i; j < pMAIN_Obj->nelem; j++)
					 		for (k = j; k < pMAIN_Obj->nelem; k++) {
					 			iv2 = i*pMAIN_Obj->nelem*pMAIN_Obj->nelem;
					 			iv2 += j*pMAIN_Obj->nelem + k;
					 			for (coeff_idx1 = 0; coeff_idx1 < numD3; coeff_idx1++)
					 				for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD3; coeff_idx2++)
					 					for (coeff_idx3 = coeff_idx2; coeff_idx3 < numD3; coeff_idx3++){
					 						iv1 = coeff_idx1*numD3*numD3;
					 						iv1 += coeff_idx2*numD3 + coeff_idx3;
					 						if (dd==0)
					 							var1 = corr33_mat[iv2][iv1];
					 						if (dd>0)
					 							var1 = der_corr33_mat[dd-1][iv2][iv1];
					 						pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] = var1;

											coeff_track=coeff_track+1;
					 					}
					 		}

					 if (pMAIN_Obj->isC34 == 1 ) {
					 	for (i = 0; i < pMAIN_Obj->nelem; i++)
							for (j = i; j < pMAIN_Obj->nelem; j++)
					 			for (k = j; k < pMAIN_Obj->nelem; k++) {
					 				iv2 = i*pMAIN_Obj->nelem*pMAIN_Obj->nelem;
					 				iv2 += j*pMAIN_Obj->nelem + k;
					 				for (coeff_idx1 = 0; coeff_idx1 < numD3; coeff_idx1++)
					 					for (coeff_idx2 = coeff_idx1; coeff_idx2 < numD3; coeff_idx2++)
					 						for (coeff_idx3 = coeff_idx2; coeff_idx3 < numD3; coeff_idx3++){
					 							iv1 = coeff_idx1*numD3*numD3;
					 							iv1 += coeff_idx2*numD3 + coeff_idx3;
					 							if (dd==0)
					 								var1 = corr34_mat[iv2][iv1];
					 							if (dd>0)
					 								var1 = der_corr34_mat[dd-1][iv2][iv1];
					 							pMAIN_Obj->Amat_domain[pMAIN_Obj->count2][coeff_track] = var1;
												coeff_track=coeff_track+1;
					 						}
					 			}
					 }
					 pMAIN_Obj->count2 = pMAIN_Obj->count2+1;
					 // printf("grid_i %d/%d,%d/%d,%d/%d\n",grid_i,NDx,grid_j,NDy,grid_k,NDz);
				}
			}
		}
		
	}

	for (i = 0; i < pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem; i++){
		free(corr33_mat[i]);
		if(pMAIN_Obj->isC34==1) free(corr34_mat[i]);
	}
	free(corr33_mat);
	if(pMAIN_Obj->isC34==1) free(corr34_mat);

	free(dist_ij);

	for(i=0; i<3 ; i++){
		free(basis_matrixG1[i]);
	}
	free(basis_matrixG1);

	for(i=0;i<4;i++) {
		free(pos_store[i]);
	}
	free(pos_store);


	if (pMAIN_Obj->isgradient==1 || pMAIN_Obj->ishessian==1){
		for(i = 0; i<der_terms; i++){
			for(j = 0; j<pMAIN_Obj->nelem * pMAIN_Obj->nelem * pMAIN_Obj->nelem; j++) {
				free(der_corr33_mat[i][j]);
				if(pMAIN_Obj->isC34==1) free(der_corr34_mat[i][j]);
			}
		}

		for(i = 0; i<der_terms; i++){
			free(der_corr33_mat[i]);
			if(pMAIN_Obj->isC34==1) free(der_corr34_mat[i]);
		}

		free(der_corr33_mat);
		if(pMAIN_Obj->isC34==1) free(der_corr34_mat);	
	}

	for (int bas =0; bas < 2*(pMAIN_Obj->num_bases_C33-1); bas++){
		free(cheb_bases_dr34[bas]);
	}
	for (int bas =0; bas < pMAIN_Obj->num_bases_C33; bas++){
		free(cheb_bases_dr33[bas]);
	}
	free(cheb_bases_dr33);
	free(cheb_bases_dr34);

	
	for (int bas=0; bas <file_param_Obj->natoms * file_param_Obj->natoms; bas++){
		free(bases_ij[bas]);
	}
	free(bases_ij);


}


int find_chem_idx(int atom_idx, DenFile_param_Obj *file_param_Obj, main_Obj *pMAIN_Obj) {
	int chem_idx_f;
	int i, j;
	int start_idx, end_idx;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (atom_idx >= 0 && atom_idx < file_param_Obj->atom_typ[0]){
		chem_idx_f = 0;
	}


			
	if (pMAIN_Obj->nelem > 1) {
		start_idx = file_param_Obj->atom_typ[0];
		end_idx = file_param_Obj->atom_typ[0] + file_param_Obj->atom_typ[1];
		for (i=1; i< pMAIN_Obj->nelem; i++){
			if (atom_idx >= start_idx && atom_idx< end_idx){
				chem_idx_f = i;
				break;
		}
		start_idx = start_idx + file_param_Obj->atom_typ[i];
		end_idx = end_idx + file_param_Obj->atom_typ[i+1];
	}
} 
return chem_idx_f;
 }

void print_Amat_distributed(main_Obj *pMAIN_Obj) {
	int i,j, temp1, der_temp;
	int rank, nprocs, rows_Amat;



	int der_terms; 

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0){
		rows_Amat = pMAIN_Obj->grid_points_procs;
		der_terms = 0;
	}
		
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0){
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
		der_terms = 3;
	}
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1){
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
		der_terms = 3;
	}
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1){
		rows_Amat = 7*pMAIN_Obj->grid_points_procs;
		der_terms = 6;
	}

	double *Amat_1D, *Amat_root_1D, **Amat_root,*bvec_root,*bvec_rootx,*bvec_rooty,*bvec_rootz,*bvec_root_rho_only;
	double *bvec_rootxx,*bvec_rootyy,*bvec_rootzz;
  Amat_1D = (double *) malloc(rows_Amat*pMAIN_Obj->num_coeff * sizeof(double));

  if (rank==0){
		Amat_root_1D = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *pMAIN_Obj->num_coeff* sizeof(double));
		Amat_root = (double **) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) * sizeof(double*));
		bvec_root = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) * sizeof(double));
		bvec_root_rho_only = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		bvec_rootx = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		bvec_rooty = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		bvec_rootz = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		bvec_rootxx = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		bvec_rootyy = (double *) malloc(pMAIN_Obj->total_grid_points_train* sizeof(double));
		bvec_rootzz = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));
		// I_root = (int *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(int));
		// J_root = (int *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(int));
		// K_root = (int *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(int));
		// F_root = (int *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(int));
		for (int i = 0; i < pMAIN_Obj->total_grid_points_train*(1+der_terms); i++){
			Amat_root[i] = (double *) malloc(pMAIN_Obj->num_coeff * sizeof(double));
		}
		for (int i = 0; i < pMAIN_Obj->total_grid_points_train*(1+der_terms); i++){
			for (int j=0;j<pMAIN_Obj->num_coeff;j++){
				Amat_root[i][j] = 0.0;
			}
		} 
	}


	int idx=0;
	for (int i=0;i<rows_Amat;i++){
		for(int j =0;j<pMAIN_Obj->num_coeff;j++){
			Amat_1D[idx] = pMAIN_Obj->Amat_domain[i][j];
			idx++;
		}
	}


	int *recvcounts, *displs, *recvcounts_bvec, *displs_bvec, *recvcounts_bvec_der, *displs_bvec_der;
	if (rank==0) recvcounts = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) displs = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) recvcounts_bvec = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) displs_bvec = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) recvcounts_bvec_der = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) displs_bvec_der = (int *)malloc(nprocs*sizeof(int)); 
	int send_count1 =rows_Amat *pMAIN_Obj->num_coeff;
	int send_count2 = rows_Amat;
	int send_count3 = pMAIN_Obj->grid_points_procs;

	if (rank==0) {
		int track = 0, track_bvec=0, track_bvec_der=0;
		for (int j =0; j<nprocs; j++){
			displs[j] = track;
			displs_bvec[j] = track_bvec;
			displs_bvec_der[j] = track_bvec_der;
			track = track+pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms)*pMAIN_Obj->num_coeff;
			track_bvec = track_bvec+pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms);
			track_bvec_der = track_bvec_der+pMAIN_Obj->grid_trained_per_procs[j];
			recvcounts[j] = pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms) * pMAIN_Obj->num_coeff;
			recvcounts_bvec[j] = pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms) ;
			recvcounts_bvec_der[j] = pMAIN_Obj->grid_trained_per_procs[j] ;
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gatherv( Amat_1D, send_count1, MPI_DOUBLE, 
							Amat_root_1D, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	MPI_Gatherv( pMAIN_Obj->bvec_domain, send_count2, MPI_DOUBLE, 
							bvec_root, recvcounts_bvec, displs_bvec, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(pMAIN_Obj->bvec_domain_rho_only, send_count3, MPI_DOUBLE, 
							bvec_root_rho_only, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Gatherv( pMAIN_Obj->bvec_domainx, send_count3, MPI_DOUBLE, 
							bvec_rootx, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domainy, send_count3, MPI_DOUBLE, 
							bvec_rooty, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domainz, send_count3, MPI_DOUBLE, 
							bvec_rootz, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domainxx, send_count3, MPI_DOUBLE, 
							bvec_rootxx, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domainyy, send_count3, MPI_DOUBLE, 
							bvec_rootyy, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domainzz, send_count3, MPI_DOUBLE, 
							bvec_rootzz, recvcounts_bvec_der, displs_bvec_der, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Gatherv( pMAIN_Obj->I_idx, send_count2, MPI_INT, 
	// 						I_root, recvcounts_bvec, displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Gatherv( pMAIN_Obj->J_idx, send_count2, MPI_INT, 
	// 						J_root, recvcounts_bvec, displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Gatherv( pMAIN_Obj->K_idx, send_count2, MPI_INT, 
	// 						K_root, recvcounts_bvec, displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	// MPI_Gatherv( pMAIN_Obj->file_idx, send_count2, MPI_INT, 
	// 						F_root, recvcounts_bvec, displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);

	if (rank==0){
		idx=0;
		for (int i=0;i<pMAIN_Obj->total_grid_points_train*(1+der_terms);i++){
			for(int j =0;j<pMAIN_Obj->num_coeff;j++){
				Amat_root[i][j] = Amat_root_1D[idx];
				idx++;
			}
		}
	}
	// printf("came here\n");
	if (rank==0) {
		FILE *fptr1,*fptr2,*fptr3,*fptr4,*fptr5;
		fptr1 = fopen("Amat.txt","w");
		fptr2 = fopen("bvec.txt","w");
		fptr3 = fopen("bvec_dx.txt","w");
		fptr4 = fopen("bvec_dxx.txt","w");
		fptr5 = fopen("bvec_rho_only.txt","w");
		// fptr3 = fopen("index_train.txt","w");
		for (int i=0;i<pMAIN_Obj->total_grid_points_train*(1+der_terms);i++){
			int temp_rho_index;
			temp_rho_index = i/(1+der_terms);
			if (bvec_root_rho_only[temp_rho_index] > 0.00001){
				fprintf(fptr2,"%.15f\n",bvec_root[i]);
				// fprintf(fptr3,"%.15f %.15f %.15f\n",bvec_rootx[i],bvec_rooty[i],bvec_rootz[i]);		
				// fprintf(fptr4,"%.15f %.15f %.15f\n",bvec_rootxx[i],bvec_rootyy[i],bvec_rootzz[i]);		
			// fprintf(fptr3,"%d %d %d %d\n",F_root[i],I_root[i],J_root[i],K_root[i]);
			for(int j =0;j<pMAIN_Obj->num_coeff;j++){			
					fprintf(fptr1,"%.15f ",Amat_root[i][j]);					
			}
		
			fprintf(fptr1,"\n");
			}
		}	

		for (int i=0;i<pMAIN_Obj->total_grid_points_train;i++){
			if (bvec_root_rho_only[i] > 0.00001){
				fprintf(fptr3,"%.15f %.15f %.15f\n",bvec_rootx[i],bvec_rooty[i],bvec_rootz[i]);		
				fprintf(fptr4,"%.15f %.15f %.15f\n",bvec_rootxx[i],bvec_rootyy[i],bvec_rootzz[i]);	
				fprintf(fptr5,"%.15f\n",bvec_root_rho_only[i]);	
			// fprintf(fptr3,"%d %d %d %d\n",F_root[i],I_root[i],J_root[i],K_root[i]);
			}
		}	
		fclose(fptr1);
		fclose(fptr2);
		fclose(fptr3);
		fclose(fptr4);
		fclose(fptr5);
	}



	free(Amat_1D);
	if (rank==0){
		free(recvcounts);
		free(displs);
		free(recvcounts_bvec);
		free(displs_bvec);
		free(recvcounts_bvec_der);
		free(displs_bvec_der);

		for (int i = 0; i < pMAIN_Obj->total_grid_points_train*(1+der_terms); i++){
			free(Amat_root[i]);
		}
		free(Amat_root);
		free(bvec_root);
		free(bvec_root_rho_only);
		free(bvec_rootx);
		free(bvec_rooty);
		free(bvec_rootz);
		free(bvec_rootxx);
		free(bvec_rootyy);
		free(bvec_rootzz);
		// free(I_root);
		// free(J_root);
		// free(K_root);
		// free(F_root);
		free(Amat_root_1D);
	}
	

}


void Gather_Amat_distributed(main_Obj *pMAIN_Obj) {

	int i,j, mat_pos_local,carryover_index1,file_temp,I_temp,J_temp,K_temp, temp1, der_temp;
	int rank, nprocs, rows_Amat;
	int *ipiv;
	double *linear_system_Atb, *linear_system_AtA, *fitted_coeffs;

	double *linear_system_AtA_root, *linear_system_Atb_root, *rho_predict_root;

	double *rho_predict;

	double error, error_total, RMSE;

	int *recvcounts_bvec, *displs_bvec, der_terms; 

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 0){
		rows_Amat = pMAIN_Obj->grid_points_procs;
		der_terms = 0;
	}
		
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 0){
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
		der_terms = 3;
	}
	if (pMAIN_Obj->isgradient == 0 && pMAIN_Obj->ishessian == 1){
		rows_Amat = 4*pMAIN_Obj->grid_points_procs;
		der_terms = 3;
	}
	if (pMAIN_Obj->isgradient == 1 && pMAIN_Obj->ishessian == 1){
		rows_Amat = 7*pMAIN_Obj->grid_points_procs;
		der_terms = 6;
	}


	displs_bvec = (int *)malloc(nprocs * sizeof(int));
	recvcounts_bvec = (int *)malloc(nprocs * sizeof(int));

	linear_system_Atb = (double *)malloc(pMAIN_Obj->num_coeff * sizeof(double));
	linear_system_AtA = (double *)malloc(pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff * sizeof(double));
	fitted_coeffs = (double *)malloc(pMAIN_Obj->num_coeff * sizeof(double));
	rho_predict = (double *)malloc(pMAIN_Obj->grid_points_procs*(1+der_terms) * sizeof(double));


	// for (i=0; i<pMAIN_Obj->num_coeff; i++) linear_system_Atb[i]=0.0;
	// for (i=0; i<pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff; i++) linear_system_AtA[i]=0.0;

	linear_system_Atb_root = (double *)malloc(pMAIN_Obj->num_coeff * sizeof(double));
	linear_system_AtA_root = (double *)malloc(pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff * sizeof(double));
	// for (i=0; i<pMAIN_Obj->num_coeff; i++) linear_system_Atb_root[i]=0.0;
	// for (i=0; i<pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff; i++) linear_system_AtA_root[i]=0.0;

	

	

	for (i = 0; i <rows_Amat;i++ ) {
		for (j=0; j<pMAIN_Obj->num_coeff; j++) {
			mat_pos_local = j + i*pMAIN_Obj->num_coeff;
			pMAIN_Obj->Amat_1D[mat_pos_local] = pMAIN_Obj->Amat_domain[i][j];
		}
	}

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pMAIN_Obj->num_coeff, 1,
	      rows_Amat, 1, pMAIN_Obj->Amat_1D, pMAIN_Obj->num_coeff, 
	      pMAIN_Obj->bvec_domain, 1, 0, linear_system_Atb, 1);

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pMAIN_Obj->num_coeff, pMAIN_Obj->num_coeff,
	      rows_Amat, 1, pMAIN_Obj->Amat_1D, pMAIN_Obj->num_coeff, 
	      pMAIN_Obj->Amat_1D, pMAIN_Obj->num_coeff, 0, linear_system_AtA, pMAIN_Obj->num_coeff);

	MPI_Reduce(linear_system_Atb, linear_system_Atb_root, pMAIN_Obj->num_coeff, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(linear_system_AtA, linear_system_AtA_root, pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff, MPI_DOUBLE, MPI_SUM,
			   0, MPI_COMM_WORLD);

	if (rank==0){
		FILE *fp;
		fp = fopen("Atb_d.txt", "w");
		for(i=0; i<pMAIN_Obj->num_coeff; i++){
			fprintf(fp,"%.15f\n",linear_system_Atb_root[i]);
		}
		fclose(fp);

		fp = fopen("AtA_d.txt", "w");
    	int newline=0;
    	for (i=0; i<pMAIN_Obj->num_coeff*pMAIN_Obj->num_coeff; i++){
    		fprintf(fp, "%.15f ",linear_system_AtA_root[i]);
    		newline = newline+1;
			if (newline==pMAIN_Obj->num_coeff){
				fprintf(fp,"\n");	
				newline=0;
			} 
    	}
    	fclose(fp);

	}

	ipiv = (int *) malloc (pMAIN_Obj->num_coeff * sizeof(int));

	if (rank==0) {
		
		for (i=0; i<pMAIN_Obj->num_coeff; i++){
			linear_system_AtA_root[i*pMAIN_Obj->num_coeff+i]=linear_system_AtA_root[i*pMAIN_Obj->num_coeff+i] + pMAIN_Obj->lambda;
		}
		int return_info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, pMAIN_Obj->num_coeff, 1,
			      linear_system_AtA_root, pMAIN_Obj->num_coeff, ipiv,
			      linear_system_Atb_root, 1 );
		if (pMAIN_Obj->ifwrite==1){
    		FILE *fp;
    		fp = fopen("solved_coeffs_d.txt", "w");
    		for (j = 0; j < pMAIN_Obj->num_coeff; j++){
    			fitted_coeffs[j] = linear_system_Atb_root[j];
    			fprintf(fp, "%.15f\n", linear_system_Atb_root[j]);
    		}
    		fclose(fp);
    	}
	}

	MPI_Bcast(fitted_coeffs, pMAIN_Obj->num_coeff, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_Amat, 1,
	      pMAIN_Obj->num_coeff, 1, pMAIN_Obj->Amat_1D, pMAIN_Obj->num_coeff, 
	      fitted_coeffs, 1, 0, rho_predict, 1);

	for (i = 0; i <rows_Amat;i++ ) {
		file_temp = pMAIN_Obj->file_idx[i];
		I_temp = pMAIN_Obj->node_begin_x[file_temp] + pMAIN_Obj->I_idx[i];
		J_temp = pMAIN_Obj->node_begin_y[file_temp] + pMAIN_Obj->J_idx[i];
		K_temp = pMAIN_Obj->node_begin_z[file_temp] + pMAIN_Obj->K_idx[i];
		der_temp = pMAIN_Obj->der_idx[i];
		temp1 = (1+der_terms)*(I_temp + J_temp* pMAIN_Obj->Nx_used[file_temp] + K_temp* pMAIN_Obj->Nx_used[file_temp]* pMAIN_Obj->Ny_used[file_temp])+der_temp;
		if (file_temp==0) {
			carryover_index1 = 0;
		} else {
			carryover_index1 = 0;
			for (int ii = 0; ii<file_temp; ii++) {
				carryover_index1 = carryover_index1 + pMAIN_Obj->Nx_used[ii] *pMAIN_Obj->Ny_used[ii] *pMAIN_Obj->Nz_used[ii] *(1+der_terms);
			}
		}
		pMAIN_Obj->bvec_pos[i] = carryover_index1 + temp1;
	}

	if (rank==0) {
		int track_bvec=0;
		for (int j =0; j<nprocs; j++){
			displs_bvec[j] = track_bvec;
			track_bvec = track_bvec+pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms);
			recvcounts_bvec[j] = pMAIN_Obj->grid_trained_per_procs[j]*(1+der_terms) ;
		}
	}

	pMAIN_Obj->bvec_pos_root = (int *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *  sizeof(int));
	pMAIN_Obj->bvec_root = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *  sizeof(double));
	rho_predict_root = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *  sizeof(double));

	MPI_Gatherv( pMAIN_Obj->bvec_pos, rows_Amat, MPI_INT, pMAIN_Obj->bvec_pos_root, recvcounts_bvec,
			     displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Gatherv(rho_predict, rows_Amat, MPI_DOUBLE, rho_predict_root, recvcounts_bvec,
			     displs_bvec, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Gatherv(pMAIN_Obj->bvec_domain, rows_Amat, MPI_DOUBLE, pMAIN_Obj->bvec_root, recvcounts_bvec,
			     displs_bvec, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank==0){
		double *bvec_sorted, *rho_predict_sorted;
		bvec_sorted = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *  sizeof(double));
		rho_predict_sorted = (double *) malloc(pMAIN_Obj->total_grid_points_train*(1+der_terms) *  sizeof(double));

		for (i=0; i<pMAIN_Obj->total_grid_points_train*(1+der_terms); i++){
			bvec_sorted[pMAIN_Obj->bvec_pos_root[i]] = pMAIN_Obj->bvec_root[i];
			rho_predict_sorted[pMAIN_Obj->bvec_pos_root[i]] = rho_predict_root[i];
		}

		double *bvec_sorted_onlyRho, *rho_predict_sorted_onlyRho;
		bvec_sorted_onlyRho = (double *) malloc(pMAIN_Obj->total_grid_points_train *  sizeof(double));
		rho_predict_sorted_onlyRho = (double *) malloc(pMAIN_Obj->total_grid_points_train *  sizeof(double));
		int idx_x =0;
		for (i=0; i<pMAIN_Obj->total_grid_points_train*(1+der_terms); i=i+der_terms+1){
			bvec_sorted_onlyRho[idx_x] = bvec_sorted[i];
			rho_predict_sorted_onlyRho[idx_x] = rho_predict_sorted[i];
			idx_x=idx_x+1;
		}


		FILE *fp2;
		fp2 = fopen("prediction_exact_d.txt","w");
		fprintf(fp2,"DFT    Prediction\n");
		for (i=0; i<pMAIN_Obj->total_grid_points_train; i++){
			fprintf(fp2,"%1f %1f\n",bvec_sorted_onlyRho[i], rho_predict_sorted_onlyRho[i]);
		}
		fclose(fp2);
		free(bvec_sorted);
		free(rho_predict_sorted);
		free(bvec_sorted_onlyRho);
		free(rho_predict_sorted_onlyRho);
		
	}


	error = 0;
	for (i=0; i<pMAIN_Obj->grid_points_procs*(1+der_terms); i=i+1+der_terms){
		error = error + (rho_predict[i] - pMAIN_Obj->bvec_domain[i])*(rho_predict[i] - pMAIN_Obj->bvec_domain[i]);
	}

	MPI_Reduce(&error, &error_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank==0) RMSE = sqrt(error_total/(pMAIN_Obj->total_grid_points_train));

	if (rank==0){
		printf("RMSE is: %1f\n",RMSE);
		FILE *fprmse;
		fprmse = fopen("final_RMSE.txt","w");
		fprintf(fprmse,"%1f\n",RMSE);
		fclose(fprmse);

	} 
// printf("came here\n");

	free(pMAIN_Obj->bvec_pos_root);
	free(pMAIN_Obj->bvec_root);
	free(rho_predict_root);
	free(linear_system_Atb);
	free(linear_system_AtA);
	for (i=0; i<pMAIN_Obj->grid_points_procs*(1+der_terms); i++) free(pMAIN_Obj->Amat_domain[i]);
	free(pMAIN_Obj->Amat_domain);
	free(pMAIN_Obj->bvec_domain);
	free(pMAIN_Obj->bvec_domainx);
	free(pMAIN_Obj->bvec_domainy);
	free(pMAIN_Obj->bvec_domainz);
	free(pMAIN_Obj->bvec_domainxx);
	free(pMAIN_Obj->bvec_domainyy);
	free(pMAIN_Obj->bvec_domainzz);

	free(pMAIN_Obj->Amat_1D);

	free(linear_system_Atb_root);
	free(linear_system_AtA_root);
	free(rho_predict);
	free(fitted_coeffs);
	free(displs_bvec);
	free(recvcounts_bvec);


	free(pMAIN_Obj->alpha_C11);
	free(pMAIN_Obj->beta_C11);
	free(pMAIN_Obj->alpha_C22);
	free(pMAIN_Obj->beta_C22);
	free(pMAIN_Obj->alpha_C33);
	free(pMAIN_Obj->beta_C33);
	free(pMAIN_Obj->chg_jump_x);
	free(pMAIN_Obj->chg_jump_y);
	free(pMAIN_Obj->chg_jump_z);
	free(pMAIN_Obj->node_begin_x);
	free(pMAIN_Obj->node_end_x);
	free(pMAIN_Obj->node_begin_y);
	free(pMAIN_Obj->node_end_y);
	free(pMAIN_Obj->node_begin_z);
	free(pMAIN_Obj->node_end_z);
	free(pMAIN_Obj->node_num_x);
	free(pMAIN_Obj->node_num_y);
	free(pMAIN_Obj->node_num_z);
	if (pMAIN_Obj->ischeb !=1) {
		for (int i=0; i<pMAIN_Obj->numG_C11; i++)
			free(pMAIN_Obj->Gaussian_Width_C11[i]);

		free(pMAIN_Obj->Gaussian_Width_C11);

		for (int i=0; i<pMAIN_Obj->numG_C22; i++)
			free(pMAIN_Obj->Gaussian_Width_C22[i]);

		free(pMAIN_Obj->Gaussian_Width_C22);

		for (int i=0; i<pMAIN_Obj->numG_C33; i++)
			free(pMAIN_Obj->Gaussian_Width_C33[i]);

		free(pMAIN_Obj->Gaussian_Width_C33);
	}
	
	
	free(pMAIN_Obj->Nx_used);
	free(pMAIN_Obj->Ny_used);
	free(pMAIN_Obj->Nz_used);
	free(pMAIN_Obj->natoms);
	free(pMAIN_Obj->Amat_pos);
	free(pMAIN_Obj->bvec_pos);
	free(pMAIN_Obj->I_idx);
	free(pMAIN_Obj->J_idx);
	free(pMAIN_Obj->K_idx);
	free(pMAIN_Obj->file_idx);
	free(pMAIN_Obj->der_idx);
	if (rank==0) free(pMAIN_Obj->grid_trained_per_procs);
   

}

void Gather_Amat(main_Obj *pMAIN_Obj) {

	int i, j;
	int I_temp, J_temp, K_temp, file_temp;
	long int carryover_index,carryover_index1, temp1;
	int node_begin_x, node_end_x, node_begin_y, node_end_y, node_begin_z, node_end_z;
	int mat_pos_local, mat_pos_global;
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	// FILE *fptr1;
	// fptr1 = fopen("Amat.txt","w");
	// for (i=0;i<pMAIN_Obj->total_grid_points_train;i++){
	// 	for (j=0;j<pMAIN_Obj->num_coeff;j++){
	// 		fprintf(fptr1,"%1f ",pMAIN_Obj->Amat_domain[i][j]);
	// 	}
	// 	fprintf(fptr1,"\n");
	// }
	// fclose(fptr1);

	int *recvcounts, *displs, *recvcounts_bvec, *displs_bvec; 

	if (rank==0) recvcounts = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) displs = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) displs_bvec = (int *)malloc(nprocs*sizeof(int)); 
	if (rank==0) recvcounts_bvec = (int *)malloc(nprocs*sizeof(int)); 
	// printf("numcoeff: %d\n",pMAIN_Obj->num_coeff);

	if (rank==0) pMAIN_Obj->Amat_1D_root = (double *) malloc(pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff * sizeof(double));
	if (rank==0) pMAIN_Obj->Amat_pos_root = (int *) malloc(pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff * sizeof(int));
	if (rank==0) pMAIN_Obj->bvec_root = (double *) malloc(pMAIN_Obj->total_grid_points_train *  sizeof(double));
	if (rank==0) pMAIN_Obj->bvec_pos_root = (int *) malloc(pMAIN_Obj->total_grid_points_train *  sizeof(int));

	for (i = 0; i <pMAIN_Obj->grid_points_procs;i++ ) {
		file_temp = pMAIN_Obj->file_idx[i];
		I_temp = pMAIN_Obj->node_begin_x[file_temp] + pMAIN_Obj->I_idx[i];
		J_temp = pMAIN_Obj->node_begin_y[file_temp] + pMAIN_Obj->J_idx[i];
		K_temp = pMAIN_Obj->node_begin_z[file_temp] + pMAIN_Obj->K_idx[i];
		temp1 = I_temp + J_temp* pMAIN_Obj->Nx_used[file_temp] + K_temp* pMAIN_Obj->Nx_used[file_temp]* pMAIN_Obj->Ny_used[file_temp];
		if (file_temp==0) {
			carryover_index = 0;
			carryover_index1 = 0;
		} else {
			carryover_index = 0;
			carryover_index1 = 0;
			for (int ii = 0; ii<file_temp; ii++) {
				carryover_index = carryover_index + pMAIN_Obj->Nx_used[ii] *pMAIN_Obj->Ny_used[ii] *pMAIN_Obj->Nz_used[ii] *pMAIN_Obj->num_coeff;
				carryover_index1 = carryover_index1 + pMAIN_Obj->Nx_used[ii] *pMAIN_Obj->Ny_used[ii] *pMAIN_Obj->Nz_used[ii];
			}
		}
		pMAIN_Obj->bvec_pos[i] = carryover_index1 + temp1;
		for (j=0; j<pMAIN_Obj->num_coeff; j++) {
			mat_pos_local = j + i*pMAIN_Obj->num_coeff;
			
			mat_pos_global = carryover_index + temp1*pMAIN_Obj->num_coeff + j;
			pMAIN_Obj->Amat_1D[mat_pos_local] = pMAIN_Obj->Amat_domain[i][j];
			pMAIN_Obj->Amat_pos[mat_pos_local] = mat_pos_global;
		}
	}

	if (rank==0) {
		int track = 0, track_bvec=0;
		for (int j =0; j<nprocs; j++){
			displs[j] = track;
			displs_bvec[j] = track_bvec;

			// recvcounts[j] = pMAIN_Obj->node_num_x[j]*pMAIN_Obj->node_num_y[j]*pMAIN_Obj->node_num_z[j]*pMAIN_Obj->num_coeff;
			// track = track+pMAIN_Obj->node_num_x[j]*pMAIN_Obj->node_num_y[j]*pMAIN_Obj->node_num_z[j]*pMAIN_Obj->num_coeff;
			
			track = track+pMAIN_Obj->grid_trained_per_procs[j]*pMAIN_Obj->num_coeff;
			track_bvec = track_bvec+pMAIN_Obj->grid_trained_per_procs[j];
			recvcounts[j] = pMAIN_Obj->grid_trained_per_procs[j] * pMAIN_Obj->num_coeff;
			recvcounts_bvec[j] = pMAIN_Obj->grid_trained_per_procs[j] ;
		}
	}
	

	int send_count1 = pMAIN_Obj->grid_points_procs *pMAIN_Obj->num_coeff;
	int send_count1_bvec = pMAIN_Obj->grid_points_procs ;



	MPI_Gatherv( pMAIN_Obj->Amat_1D, send_count1, MPI_DOUBLE, pMAIN_Obj->Amat_1D_root, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->Amat_pos, send_count1, MPI_INT, pMAIN_Obj->Amat_pos_root, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_pos, send_count1_bvec, MPI_INT, pMAIN_Obj->bvec_pos_root, recvcounts_bvec, displs_bvec, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gatherv( pMAIN_Obj->bvec_domain, send_count1_bvec, MPI_DOUBLE, pMAIN_Obj->bvec_root, recvcounts_bvec, displs_bvec, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	if (rank==0) sort_Amat(pMAIN_Obj);



	if (rank==0) free(pMAIN_Obj->Amat_1D_root);

	if (rank==0) free(pMAIN_Obj->Amat_pos_root);

	if (rank==0) free(pMAIN_Obj->bvec_root);

	if (rank==0) free(pMAIN_Obj->bvec_pos_root);
	// printf("6.came here rank %d\n",rank);
	

	for (int j = 0; j<pMAIN_Obj->grid_points_procs; j++){
		free(pMAIN_Obj->Amat_domain[j]);
	}
	free(pMAIN_Obj->Amat_domain);
	free(pMAIN_Obj->bvec_domain);
	free(pMAIN_Obj->Amat_1D);
	free(pMAIN_Obj->Amat_pos);
	free(pMAIN_Obj->bvec_pos);
	free(pMAIN_Obj->I_idx);
	free(pMAIN_Obj->J_idx);
	free(pMAIN_Obj->K_idx);
	free(pMAIN_Obj->file_idx);

	

	if (rank==0 && pMAIN_Obj->ifwrite==1) {
		free(recvcounts);
		free(displs);
		free(displs_bvec);
		free(recvcounts_bvec);
		FILE *fptr1,*fptr2; 
		fptr1 = fopen("Amat.txt","w");
		fptr2 = fopen("bvec.txt","w");
		int newline=0;
		printf("Started printing Amat\n");
		for (i=0; i<pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff; i++){
			
			fprintf(fptr1, "%1f ",pMAIN_Obj->Amat_1D_root_sorted[i]);
			newline = newline+1;
			if (newline==pMAIN_Obj->num_coeff){
				fprintf(fptr1,"\n");	
				newline=0;
			} 
		}
		printf("Started printing bvec\n");
		for (i=0; i<pMAIN_Obj->total_grid_points_train; i++){
			fprintf(fptr2, "%1f\n",pMAIN_Obj->bvec_root_sorted[i]);
		}
		fclose(fptr1);
		fclose(fptr2);
	}
}



void sort_Amat(main_Obj *pMAIN_Obj) {
	int i;

	pMAIN_Obj->Amat_1D_root_sorted = (double *) malloc(pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff * sizeof(double));
	pMAIN_Obj->bvec_root_sorted = (double *) malloc(pMAIN_Obj->total_grid_points_train * sizeof(double));

	for (i = 0; i <pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff; i++){
		pMAIN_Obj->Amat_1D_root_sorted[i] = 0;
	}


	for (i = 0; i <pMAIN_Obj->total_grid_points_train ; i++){
		pMAIN_Obj->bvec_root_sorted[i] = 0;
	}

	
	for (i = 0; i <pMAIN_Obj->total_grid_points_train * pMAIN_Obj->num_coeff; i++){
		// printf("%d %1f\n",i,pMAIN_Obj->Amat_pos_root[i]);
		pMAIN_Obj->Amat_1D_root_sorted[pMAIN_Obj->Amat_pos_root[i]] = pMAIN_Obj->Amat_1D_root[i];
	}


	for (i = 0; i <pMAIN_Obj->total_grid_points_train ; i++){
		pMAIN_Obj->bvec_root_sorted[pMAIN_Obj->bvec_pos_root[i]] = pMAIN_Obj->bvec_root[i];
	}

}

void solve_linear_system(main_Obj *pMAIN_Obj){
	int rank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank==0){
		int i, j, *ipiv, return_info;
		double *linear_system_Atb, *linear_system_AtA;
		linear_system_Atb = (double *) malloc (pMAIN_Obj->num_coeff*sizeof(double));
		for (j = 0; j < pMAIN_Obj->num_coeff; j++)
    		linear_system_Atb[j] = 0.00;

    	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pMAIN_Obj->num_coeff, 1,
	      pMAIN_Obj->total_grid_points_train, 1, pMAIN_Obj->Amat_1D_root_sorted, pMAIN_Obj->num_coeff, 
	      pMAIN_Obj->bvec_root_sorted, 1, 0, linear_system_Atb, 1);

    	FILE *fp;
    	fp = fopen("Atb.txt", "w");
    	for (i=0; i<pMAIN_Obj->num_coeff; i++){
    		fprintf(fp,"%1f\n",linear_system_Atb[i]);
    		
    	}
    	fclose(fp);

    	
    	linear_system_AtA = (double *) malloc (pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff * sizeof(double));
    	for (j = 0; j < pMAIN_Obj->num_coeff * pMAIN_Obj->num_coeff; j++)
    		linear_system_AtA[j] = 0.00;

    	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, pMAIN_Obj->num_coeff, pMAIN_Obj->num_coeff,
	      pMAIN_Obj->total_grid_points_train, 1, pMAIN_Obj->Amat_1D_root_sorted, pMAIN_Obj->num_coeff, 
	      pMAIN_Obj->Amat_1D_root_sorted, pMAIN_Obj->num_coeff, 0, linear_system_AtA, pMAIN_Obj->num_coeff);

    	fp = fopen("AtA.txt", "w");
    	int newline=0;
    	for (i=0; i<pMAIN_Obj->num_coeff*pMAIN_Obj->num_coeff; i++){
    		fprintf(fp, "%1f ",linear_system_AtA[i]);
    		newline = newline+1;
			if (newline==pMAIN_Obj->num_coeff){
				fprintf(fp,"\n");	
				newline=0;
			} 
    	}
    	fclose(fp);

    	for (j = 0; j < pMAIN_Obj->num_coeff; j++)
    		linear_system_AtA[j*pMAIN_Obj->num_coeff+j] += pMAIN_Obj->lambda;

    	ipiv = (int *) malloc (pMAIN_Obj->num_coeff * sizeof(int));

    	return_info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, pMAIN_Obj->num_coeff, 1,
			       linear_system_AtA, pMAIN_Obj->num_coeff, ipiv,
			       linear_system_Atb, 1 );
    	// free((void *) linear_system_AtA);

    	if (pMAIN_Obj->ifwrite==1){
    		FILE *fp;
    		fp = fopen("solved_coeffs.txt", "w");
    		for (j = 0; j < pMAIN_Obj->num_coeff; j++)
    			fprintf(fp, "%.10f\n", linear_system_Atb[j]);
    		fclose(fp);
    	}

    	double *rho_predict, *rho_diff, sq_error=0.0,RMSE;
    	rho_predict = (double *) malloc (pMAIN_Obj->total_grid_points_train * sizeof(double));
    	rho_diff = (double *) malloc (pMAIN_Obj->total_grid_points_train * sizeof(double));
    	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, pMAIN_Obj->total_grid_points_train, 1,
	      pMAIN_Obj->num_coeff, 1, pMAIN_Obj->Amat_1D_root_sorted, pMAIN_Obj->num_coeff, 
	      linear_system_Atb, 1, 0, rho_predict, 1);

    	for (j=0; j<pMAIN_Obj->total_grid_points_train;j++){
    		rho_diff[j] = rho_predict[j]-pMAIN_Obj->bvec_root_sorted[j];
    		sq_error = sq_error + rho_diff[j] * rho_diff[j];
    	}
    	RMSE = sqrt(sq_error/pMAIN_Obj->total_grid_points_train);

    	printf("RMSE: %.10f\n",RMSE);
    	if (pMAIN_Obj->ifwrite==1){
    		FILE *fp;
    		fp = fopen("prediction_exact.txt", "w");
    		fprintf(fp,"DFT     Prediction\n");
    		for (j = 0; j < pMAIN_Obj->total_grid_points_train; j++)
    			fprintf(fp, "%1f %1f\n", pMAIN_Obj->bvec_root_sorted[j],rho_predict[j]);
    		fclose(fp);
    	}
    	free(pMAIN_Obj->bvec_root_sorted);
    	free(linear_system_Atb);
    	free(linear_system_AtA);
    	free(ipiv);
    	free(rho_predict);
    	free(rho_diff);
    	free(pMAIN_Obj->Amat_1D_root_sorted);
	}
	free(pMAIN_Obj->alpha_C11);
	free(pMAIN_Obj->beta_C11);
	free(pMAIN_Obj->alpha_C22);
	free(pMAIN_Obj->beta_C22);
	free(pMAIN_Obj->alpha_C33);
	free(pMAIN_Obj->beta_C33);
	free(pMAIN_Obj->chg_jump_x);
	free(pMAIN_Obj->chg_jump_y);
	free(pMAIN_Obj->chg_jump_z);
	free(pMAIN_Obj->node_begin_x);
	free(pMAIN_Obj->node_end_x);
	free(pMAIN_Obj->node_begin_y);
	free(pMAIN_Obj->node_end_y);
	free(pMAIN_Obj->node_begin_z);
	free(pMAIN_Obj->node_end_z);
	free(pMAIN_Obj->node_num_x);
	free(pMAIN_Obj->node_num_y);
	free(pMAIN_Obj->node_num_z);
	for (int i=0; i<pMAIN_Obj->numG_C11; i++)
		free(pMAIN_Obj->Gaussian_Width_C11[i]);
	free(pMAIN_Obj->Gaussian_Width_C11);
	free(pMAIN_Obj->Nx_used);
	free(pMAIN_Obj->Ny_used);
	free(pMAIN_Obj->Nz_used);
	free(pMAIN_Obj->natoms);

}