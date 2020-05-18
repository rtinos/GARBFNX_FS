#include "defs.h"
 
/******************************************************************************\
*				  	Global Variables								 		   *
\******************************************************************************/
// GA
population popold, popnew, poplopt;				// populations
int maxgen;										// maximum number of generations 
double *file_best_fitness, *time_run;			// data to be stored: best fitness, runtime
double *file_best_fitness_gen;					// data to be stored: best fitness over the generations for run 0
double *file_diversity_gen;						// data to be stored: population diversity over the generations for run 0
int *file_gen;									// data to be stored: number of generations
int **File_best_ind;							// data to be stored: best individual
double **File_sucRate_crType;					// data to be stored: successful crossover rate for each crossover type (only for cross_type=3 and 4)
double **File_sucRate_crType_gens;				// data to be stored: successful crossover rate over the generations (run 0) for each crossover type (only for cross_type=3 and 4)
double **File_impRate_crType;					// data to be stored: successful rate for each crossover type (only for cross_type=3 and 4)
double *File_sucRate_all;						// data to be stored: successful crossover rate 
double *File_impRate_all;						// data to be stored: improvement rate
long int gen;									// generation
long int max_gen;								// maximum number of generations (used only in some experiments)
int lcrom;										// size of the cromosome 
int popsize = 100;								// size of the population (>1)
int n_runs_max = 30;							// runs of the GA
double p_cross=0.60;							// crossover rate
double imig_rate=1.0;							// immigrant rate 
int tournament_size=3;							// size of the pool for tournament selection 
int cross_type;									// crossover type 1: 2X; type 2: UX; type 3: dRBFNX; type 4: sRBFNX; type 5: mRBFNX 
int *temp_v;									// vector of integers (used in different methods)
long int sum_cross_crType[3];					// for the crossover statistics 
long int sum_sucRate_crType[3]; 				// for the crossover statistics 
long int sum_impRate_crType[3];					// for the crossover statistics 
long int sum_cross_all; 						// for the crossover statistics 
long int sum_sucRate_all;						// for the crossover statistics 
long int sum_impRate_all;						// for the crossover statistics 
int save_datagen_flag=0;						// flag for saving data for generation in the first run
int pop_lopt_flag=1;							// flag for keeping a local optima population and using it

// RBFN
int RBF_config;									// configuration of the RBF (see RBF_sup.cpp)

// Data set and classifier
double **X_dataset;								// dataset: inputs
int *d_dataset;									// dataset: desired outputs 
int n_examples;									// number of examples
int n_classes;									// number of classes
int classifier;									// classifier: 1: KNN with K=3; 2: KNN with K=5; 3: KNN with K=7
				
