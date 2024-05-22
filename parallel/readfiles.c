// readfiles
#define L_STRING 512

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
#include <errno.h>
#include "mkl.h"
// #include "common.c"
// #include "read_chgcar.c"
#include <stdbool.h>

#include "STRUCTURES.h"
#include "fitting.h"
#include "initialization.h"
#include "readfiles.h"
#include "common.h"

void read_input(InputFile_Obj *fInput_Obj, main_Obj *ffmain_common_Obj){
	char *input_filename = malloc(L_STRING * sizeof(char));
	char *str = malloc(L_STRING * sizeof(char));
	char *temp = malloc(L_STRING * sizeof(char));

	snprintf(input_filename, L_STRING, "%s", fInput_Obj->filename);

	FILE *input_fp = fopen(input_filename,"r");

    #ifdef DEBUG
        printf("Reading the input file: %s\n",input_filename);
        printf("Input file parameters: \n");
        printf("-------------------------------------------\n");
    #endif

	if (input_fp == NULL) {
        printf("\nCannot open file \"%s\"\n",input_filename);
        // print_usage();
        exit(EXIT_FAILURE);
    }

    while (!feof(input_fp)) {
    	fscanf(input_fp,"%s",str);



    	if (str[0] == '#' || str[0] == '\n'|| strcmp(str,"undefined") == 0) {
            fscanf(input_fp, "%*[^\n]\n"); // skip current line
            continue;
        }


        if (strcmp(str,"isC11") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->isC11);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isC11: %d\n",fInput_Obj->isC11);
            #endif
        } else if (strcmp(str,"isC22") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->isC22);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isC22: %d\n",fInput_Obj->isC22);
            #endif
        } else if (strcmp(str,"isC23") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->isC23);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isC23: %d\n",fInput_Obj->isC23);
            #endif
        } else if (strcmp(str,"isC33") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->isC33);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isC33: %d\n",fInput_Obj->isC33);
            #endif
        } else if (strcmp(str,"isC34") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->isC34);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isC34: %d\n",fInput_Obj->isC34);
            #endif
        } else if (strcmp(str,"isgradient") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->isgradient);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("isgradient: %d\n",fInput_Obj->isgradient);
            #endif
        } else if (strcmp(str,"ishessian") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->ishessian);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("ishessian: %d\n",fInput_Obj->ishessian);
            #endif
        } else if (strcmp(str,"numG_C33") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->numG_C33);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("numG_C33: %d\n",fInput_Obj->numG_C33);
            #endif
        } else if (strcmp(str,"numG_C22") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->numG_C22);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("numG_C22: %d\n",fInput_Obj->numG_C22);
            #endif
        } else if (strcmp(str,"ischeb") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->ischeb);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("ischeb: %d\n",fInput_Obj->ischeb);
            #endif
        } else if (strcmp(str,"numG_C11") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->numG_C11);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("numG: %d\n",fInput_Obj->numG_C11);
            #endif
        } else if (strcmp(str,"poly_order_C11") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->poly_order_C11);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("poly_order_C11: %d\n",fInput_Obj->poly_order_C11);
            #endif
        } else if (strcmp(str,"poly_order_C33") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->poly_order_C33);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("poly_order_C33: %d\n",fInput_Obj->poly_order_C33);
            #endif
        } else if (strcmp(str,"poly_order_C22") == 0) {
            fscanf(input_fp,"%d", &fInput_Obj->poly_order_C22);
            fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("poly_order_C22: %d\n",fInput_Obj->poly_order_C22);
            #endif
        } else if (strcmp(str,"chg_stp_jump_dist") == 0) {
            char str1[L_STRING];
            char *ptr;
            double ret;
            fscanf(input_fp,"%s",str1);
            ret = strtod(str1, &ptr);
            fInput_Obj->chg_step_jump_dist = ret;
            #ifdef DEBUG
                printf("chg_stp_jump_dist: %1f\n",fInput_Obj->chg_step_jump_dist);
            #endif
            // printf("in read function string is: %s \n", str);
        	// fscanf(input_fp,"%f", &fInput_Obj->chg_step_jump_dist);
            // printf("read value is %f \n",fInput_Obj->chg_step_jump_dist);
        	fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmp(str,"lambda") == 0) {
            char str1[L_STRING];
            char *ptr;
            double ret;
            fscanf(input_fp,"%s",str1);
            ret = strtod(str1, &ptr);
            fInput_Obj->lambda = ret;
            #ifdef DEBUG
                printf("lambda: %1f\n",fInput_Obj->lambda);
            #endif
        	// fscanf(input_fp,"%1f", &fInput_Obj->lambda);
        	fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmp(str,"den_file_name") == 0) {
        	fscanf(input_fp,"%s", fInput_Obj->den_file_name);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("den_file_name: %s\n",fInput_Obj->den_file_name);
            #endif
        } else if (strcmp(str,"den_file_start_idx") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->start_idx_den_file);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("den_file_start_idx: %d\n",fInput_Obj->start_idx_den_file);
            #endif
        } else if (strcmp(str,"den_file_end_idx") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->end_idx_den_file);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("den_file_end_idx: %d\n",fInput_Obj->end_idx_den_file);
            #endif
        } else if (strcmp(str,"n_element") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->nelem);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("n_element: %d\n",fInput_Obj->nelem);
            #endif
        } else if (strcmp(str,"Rcut") == 0) {
            char str1[L_STRING];
            char *ptr;
            double ret;
            fscanf(input_fp,"%s",str1);
            ret = strtod(str1, &ptr);
            fInput_Obj->Rcut = ret;
            #ifdef DEBUG
                printf("Rcut: %1f\n",fInput_Obj->Rcut);
            #endif
        	// fscanf(input_fp,"%1f", &fInput_Obj->Rcut);
        	fscanf(input_fp, "%*[^\n]\n");
        } else if (strcmp(str,"iftraining") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->iftraining);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("iftraining: %d\n",fInput_Obj->iftraining);
            #endif
        } else if (strcmp(str,"ifpredict") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->ifpredict);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("ifpredict: %d\n",fInput_Obj->ifpredict);
            #endif
        } else if (strcmp(str,"ifwrite") == 0) {
        	fscanf(input_fp,"%d", &fInput_Obj->ifwrite);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("ifwrite: %d\n",fInput_Obj->ifwrite);
            #endif
        } else if (strcmp(str,"coeff_file_name") == 0) {
        	fscanf(input_fp,"%s", fInput_Obj->coeff_file_name);
        	fscanf(input_fp, "%*[^\n]\n");
            #ifdef DEBUG
                printf("coeff_file_name: %s\n",fInput_Obj->coeff_file_name);
            #endif
        } 
    }

    ffmain_common_Obj->alpha_C11 = (double *)malloc(fInput_Obj->nelem * sizeof(double));
    ffmain_common_Obj->beta_C11 = (double *)malloc(fInput_Obj->nelem * sizeof(double));
    ffmain_common_Obj->alpha_C22 = (double *)malloc(fInput_Obj->nelem * sizeof(double));
    ffmain_common_Obj->beta_C22 = (double *)malloc(fInput_Obj->nelem * sizeof(double));
    ffmain_common_Obj->alpha_C33 = (double *)malloc(fInput_Obj->nelem * sizeof(double));
    ffmain_common_Obj->beta_C33 = (double *)malloc(fInput_Obj->nelem * sizeof(double));


    fseek(input_fp, 0L, SEEK_SET);

    while (!feof(input_fp)) {
    	fscanf(input_fp,"%s",str);


    	if (str[0] == '#' || str[0] == '\n'|| strcmp(str,"undefined") == 0) {
            fscanf(input_fp, "%*[^\n]\n"); // skip current line
            continue;
        }
        if (strcmp(str,"alpha_C11") == 0) {
            #ifdef DEBUG
                printf("alpha_C11: ");
            #endif
            char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;
            int errnum;
            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            if (count != fInput_Obj->nelem){
               fprintf(stderr,"Error: alpha_C11 should have same dimension as the number of elements\n");
               exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->alpha_C11[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->alpha_C11[i]);
                #endif
            }
    	
        } else if (strcmp(str,"beta_C11") == 0) {
            #ifdef DEBUG
                printf("\nbeta_C11: ");
            #endif
        	char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;
            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            if (count != fInput_Obj->nelem){
                fprintf(stderr,"Error: beta_C11 should have same dimension as the number of elements\n");
                exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->beta_C11[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->beta_C11[i]);
                #endif
            }
        } else if (strcmp(str,"alpha_C22") == 0) {
            #ifdef DEBUG
                printf("\nalpha_C22: ");
            #endif
            char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;
            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            if (count != fInput_Obj->nelem){
                fprintf(stderr,"Error: alpha_C22 should have same dimension as the number of elements\n");
                exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->alpha_C22[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->alpha_C22[i]);
                #endif
            }  
        } else if (strcmp(str,"beta_C22") == 0) {
            #ifdef DEBUG
                printf("\nbeta_C22: ");
            #endif
            char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;

            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            if (count != fInput_Obj->nelem){
                fprintf(stderr,"Error: beta_C22 should have same dimension as the number of elements\n");
                exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->beta_C22[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->beta_C22[i]);
                #endif
            }
        } else if (strcmp(str,"alpha_C33") == 0) {
            #ifdef DEBUG
                printf("\nalpha_C33: ");
            #endif
            char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;

            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            // printf("count %d\n",count);
            if (count != fInput_Obj->nelem){
                fprintf(stderr,"Error: alpha_C33 should have same dimension as the number of elements\n");
                exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->alpha_C33[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->alpha_C33[i]);
                #endif
            } 
        } else if (strcmp(str,"beta_C33") == 0) {
            #ifdef DEBUG
                printf("\nbeta_C33: ");
            #endif
            char str1[L_STRING];
            fgets(str1,L_STRING,input_fp);
            char *ptr;
            int count=0;
            double temp_store[100], temp4;
            int errnum;
            
            char* piece = strtok(str1," ");
            
            while(piece != NULL) {
                temp4 = strtod(piece, &ptr);
                temp_store[count] = temp4;
                piece = strtok(NULL, " ");
                count++;
            }
            if (count != fInput_Obj->nelem){
                fprintf(stderr,"Error: beta_C33 should have same dimension as the number of elements\n");
                exit(-1);
            }
            for (int i=0; i<fInput_Obj->nelem;i++ ){
                ffmain_common_Obj->beta_C33[i] = temp_store[i];
                #ifdef DEBUG
                    printf("%1f ",ffmain_common_Obj->beta_C33[i]);
                #endif
            }  
        }


    }

    #ifdef DEBUG
        printf("\n");
        printf("-------------------------------------------\n");
        printf("Finished reading the input file\n");
    #endif
    // printf("fInput_Obj->chg_step_jump_dist %1f\n",fInput_Obj->chg_step_jump_dist);
    // printf("ffmain_common_Obj->beta[0] %1f\n",ffmain_common_Obj->beta[0]);
    // printf("ffmain_common_Obj->alpha[0] %1f\n",ffmain_common_Obj->alpha[0]);
    // printf("fInput_Obj->Rcut %1f\n",fInput_Obj->Rcut);
    // printf("fInput_Obj->den_file_name %s\n",fInput_Obj->den_file_name);
    free(input_filename);

}