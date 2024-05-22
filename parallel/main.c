/**
 * @file    main.c
 * @brief   This file contains the main function for the charge density training and prediction
 *
 * @authors Shashikant Kumar <kumar17@llnl.gov>/<shashikant@gatech.edu>
 *          Amit Smanata <samanta1@llnl.gov>
 * 
 */

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
// #include "mkl.h"


// #include "blacs.h"
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
// #include "common.c"
// #include "read_chgcar.c"
#include <stdbool.h>


#include "STRUCTURES.h"
#include "fitting.h"
#include "initialization.h"
#include "common.h"


int main(int argc, char *argv[]) {
	// Initialize the MPI environment
	MPI_Init(&argc, &argv);
	// Setting the communicator size and rank of the given processor
	MPI_Comm comm = MPI_COMM_WORLD;

	int nproc, rank;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);
   #ifdef DEBUG
      if (rank==0) printf("Initializing MPI\n");
   #endif

	
	main_Obj MAIN_Obj;

	double t1, t2;

	 MPI_Barrier(MPI_COMM_WORLD);
	// // start timer
    t1 = MPI_Wtime(); MAIN_Obj.time_start = t1;

 //    // MPI_Barrier(MPI_COMM_WORLD);
    #ifdef DEBUG
      if (rank==0) printf("Initializing the MAIN_Obj");
   #endif
    Initialize(&MAIN_Obj, argc, argv);





    if (MAIN_Obj.iftraining == 1) {
    	fitting(&MAIN_Obj);
    }

     MPI_Barrier(MPI_COMM_WORLD);

     t2 = MPI_Wtime();



    if (rank == 0) {
        printf("The programsss took %.3f s.\n", t2 - t1); 
    }


MPI_Finalize();

// printf("Rank %d came here\n",rank);
return 0;

}









