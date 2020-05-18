/******************************************************************************\
*				  				 Files Manipulation							 *
\******************************************************************************/
 
#include "defs.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<string>
#include<cstring>
#include<fstream>


/***********************************************************************\
* Read the problem instance 											*
\***********************************************************************/
void read_problem(char *prob_name){
	int i, j;
	char line[CHAR_LEN], * keywords,Delimiters[] = " :=\n\t\r\f\v";
	char name[CHAR_LEN];

	sprintf(name,"%s.dat", prob_name);	
	
	ifstream fin(name);

	 if (!fin.is_open()){
			cout<<"file name error"<<endl;
			exit(0);
	}
	
	while((fin.getline(line, CHAR_LEN-1))){
			if(!(keywords = strtok(line, Delimiters)))
	  			continue;
			if(!strcmp(keywords, "N_ATTRIBUTES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &lcrom)){
					cout<<"N_ATTRIBUTES error"<<endl;
					exit(0);
	  			}
			}
			if(!strcmp(keywords, "N_EXAMPLES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &n_examples)){
					cout<<"N_EXAMPLES"<<endl;
					exit(0);
	  			}
				X_dataset=aloc_matrixd (n_examples,lcrom);
				d_dataset=aloc_vectori (n_examples);		
			}
			if(!strcmp(keywords, "N_CLASSES")){			
	  			if(!sscanf(strtok(NULL, Delimiters), "%d", &n_classes)){
					cout<<"N_CLASSES"<<endl;
					exit(0);
	  			}	
			}
			else if(!strcmp(keywords, "DATASET")){
	  			if(n_examples>0){
	  				for(i=0; i<n_examples; i++){	  				
						for(j=0; j<lcrom; j++)						
							fin>>X_dataset[i][j];						
						fin>>d_dataset[i];
					}
	    		}
			}
	}
	fin.close();
}


/******************************************************************************\
* 										Save data : end of the simulation						 *
\******************************************************************************/
void file_output(char *prob_name)
{
	int i, gene, j;
	FILE *Bestfit_file, *Bestind_file, *Time_file, *Gen_file, *sucRateAll_file, *impRateAll_file;
	char *name_p;
	char name[CHAR_LEN];

    name_p = name;

		
  	// Best fitness in each generation for run 0
  	if (save_datagen_flag==1){
  		FILE *Bfg_file;
		sprintf(name,"bfg_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
		if ((Bfg_file = fopen(name_p,"w"))==NULL) {
			puts("The file bfg to be saved cannot be open \n");
			exit(1);
		}
		for (i=0;i<max_gen;i++) {
			fprintf(Bfg_file,"%.3f ",file_best_fitness_gen[i]);
		}
		fclose(Bfg_file);
	}
	
	// Diversity in each generation for run 0
  	if (save_datagen_flag==1){
  		FILE *Div_file;
		sprintf(name,"div_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
		if ((Div_file = fopen(name_p,"w"))==NULL) {
			puts("The file div to be saved cannot be open \n");
			exit(1);
		}
		for (i=0;i<max_gen;i++) {
			fprintf(Div_file,"%.3f ",file_diversity_gen[i]);
		}
		fclose(Div_file);
	}
	
	// Crossover Statistics: successful recombination rate  considering each crossover operator (for types 3 and 4) for generation for run 0
	if (save_datagen_flag==1){
		if (cross_type==3 || cross_type==4 || cross_type==2 ){
			FILE *sucRateCrTypeGens_file;
			sprintf(name,"sucRateCrTypeGens_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
			if ((sucRateCrTypeGens_file = fopen(name_p,"w"))==NULL) {
				puts("The file sucRateCrTypeGens to be saved cannot be open \n");
				exit(1);
			}
			for (i=0;i<max_gen;i++){		
				for (j=0;j<3;j++){
					fprintf(sucRateCrTypeGens_file,"%.5f ",File_sucRate_crType_gens[i][j]);
				}
				fprintf(sucRateCrTypeGens_file,"\n");
			}
			fclose(sucRateCrTypeGens_file);
		}
	}

    // Best fitness 
	sprintf(name,"bfi_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((Bestfit_file = fopen(name_p,"w"))==NULL) {
		puts("The file bfi to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++) {
		fprintf(Bestfit_file,"%.14f ",file_best_fitness[i]);
	}
	fclose(Bestfit_file);
		
	 // Best individuals
	sprintf(name,"bind_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((Bestind_file = fopen(name_p,"w"))==NULL) {
		puts("The file bind to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++) {
		for (gene=0;gene<lcrom;gene++)
			fprintf(Bestind_file,"%d ",File_best_ind[i][gene]);
		fprintf(Bestind_file,"\n");
	}
	fclose(Bestind_file);
	
  	// Time for each run
	sprintf(name,"time_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((Time_file = fopen(name_p,"w"))==NULL) {
		puts("The file time to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++) {
		fprintf(Time_file,"%.2f ",time_run[i]);
	}
	fclose(Time_file);
	
	// Number of generations for each run
	sprintf(name,"gen_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((Gen_file = fopen(name_p,"w"))==NULL) {
		puts("The file gen to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++) {
		fprintf(Gen_file,"%d ",file_gen[i]);
	}
	fclose(Gen_file);		
	
	// Crossover Statistics for runs: successful recombination rate considering each crossover operator (for types 3 and 4)
	if (cross_type==3 || cross_type==4){
		FILE *sucRateCrType_file;
		sprintf(name,"sucRateCrType_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
		if ((sucRateCrType_file = fopen(name_p,"w"))==NULL) {
			puts("The file sucRateCrType to be saved cannot be open \n");
			exit(1);
		}
		for (i=0;i<n_runs_max;i++){		
			for (j=0;j<3;j++){
				fprintf(sucRateCrType_file,"%.8f ",File_sucRate_crType[i][j]);
			}
			fprintf(sucRateCrType_file,"\n");
		}
		fclose(sucRateCrType_file);
	}
	
	// Crossover Statistics for runs: improvement rate considering each crossover operator (for types 3 and 4)
	if (cross_type==3 || cross_type==4){
		FILE *impRateCrType_file;
		sprintf(name,"impRateCrType_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
		if ((impRateCrType_file = fopen(name_p,"w"))==NULL) {
			puts("The file impRateCrType to be saved cannot be open \n");
			exit(1);
		}
		for (i=0;i<n_runs_max;i++){		
			for (j=0;j<3;j++){
				fprintf(impRateCrType_file,"%.8f ",File_impRate_crType[i][j]);
			}
			fprintf(impRateCrType_file,"\n");
		}
		fclose(impRateCrType_file);
	}
	
	// Crossover Statistics for runs: successful recombination rate 
	sprintf(name,"sucRateAll_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((sucRateAll_file = fopen(name_p,"w"))==NULL) {
		puts("The file sucRateAll to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++)		
		fprintf(sucRateAll_file,"%.8f ",File_sucRate_all[i]);		
	fclose(sucRateAll_file);
	
	// Crossover Statistics for runs: improvement rate 
	sprintf(name,"impRateAll_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config);
	if ((impRateAll_file = fopen(name_p,"w"))==NULL) {
		puts("The file impRateAll to be saved cannot be open \n");
		exit(1);
	}
	for (i=0;i<n_runs_max;i++)		
		fprintf(impRateAll_file,"%.8f ",File_impRate_all[i]);		
	fclose(impRateAll_file);
	
}


