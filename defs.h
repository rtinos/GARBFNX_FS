/******************************************************************************\
*								 Definitions							 		*
\******************************************************************************/
#include <iostream>
#define EPS1 1.0e-16
#define EPS2 1.0e-12
#define CHAR_LEN 1000
using namespace std; 

/* Data structures */
typedef int allele; 						// data structure alele
typedef struct {
			allele *chromosome;				// chromosome
			double fitness;					// fitness
} individual;
typedef struct {			
			individual *ind;
			double sum_fitness;
			double mean_fitness;
			double max_fitness;
			int best_individual;	
			int popsize;	
} population;

// Global variables
// GA
extern population popold, popnew, poplopt;					// populations
extern double *file_best_fitness, *time_run;				// data to be stored: best fitness, runtime
extern double *file_best_fitness_gen;						// data to be stored: best fitness over the generations for run 0
extern double *file_diversity_gen;							// data to be stored: population diversity over the generations for run 0
extern int *file_gen;										// data to be stored: number of generations
extern int **File_best_ind;									// data to be stored: best individual
extern double **File_sucRate_crType;						// data to be stored: successful crossover rate for each crossover type (only for cross_type=3 and 4)
extern double **File_sucRate_crType_gens;					// data to be stored: successful crossover rate over the generations (run 0) for each crossover type (only for cross_type=3 and 4)
extern double **File_impRate_crType;						// data to be stored: successful rate for each crossover type (only for cross_type=3 and 4)
extern double *File_sucRate_all;							// data to be stored: successful crossover rate 
extern double *File_impRate_all;							// data to be stored: improvement rate
extern long int gen;										// generation
extern long int max_gen;									// maximum number of generations (used only in some experiments)
extern int popsize;											// size of the population 
extern int n_runs_max;										// total of runs 
extern double p_cross;										// crossover rate
extern double imig_rate;									// immigrant rate 
extern int tournament_size;									// size of the pool for tournament selection 
extern int lcrom;											// size of the chromosome 
extern int cross_type;										// crossover type 1: 2X; type 2: UX; type 3: dRBFNX; type 4: sRBFNX; type 5: mRBFNX   
extern int *temp_v;											// vector of integers (used in different methods)
extern long int sum_cross_crType[3];						// for the crossover statistics 
extern long int sum_sucRate_crType[3]; 						// for the crossover statistics 
extern long int sum_impRate_crType[3];						// for the crossover statistics 
extern long int sum_cross_all; 								// for the crossover statistics 
extern long int sum_sucRate_all;							// for the crossover statistics 
extern long int sum_impRate_all;							// for the crossover statistics 
extern int save_datagen_flag;								// flag for saving data for generation in the first run
extern int pop_lopt_flag;									// flag for keeping a local optima population and using it

// RBFN
extern int RBF_config;										// configuration of the RBF (see RBF_sup.cpp)

// Data set and classifier
extern double **X_dataset;							// dataset: inputs
extern int *d_dataset;								// dataset: desired outputs
extern int n_examples;								//  number of examples 
extern int n_classes;								// number of classes
extern int classifier;								// classifier 1: KNN with K=3; 2: KNN with K=5; 3: KNN with K=7

// Declaration of the functions
// statistics.cpp
void statistics( population *pop , int n_run);
double popDiversity( population *pop);
// util_functions.cpp
int *aloc_vectori(int lines);
double *aloc_vectord(int lines);
individual *aloc_vectorind(int lines);
double **aloc_matrixd(int lines , int collums);
void desaloc_matrixd(double **Matrix , int lines);
int **aloc_matrixi(int lines , int collums);
void desaloc_matrixi(int **Matrix , int lines);
int random_int(int L_range, int H_range);
double random_dou(void);
void rand_perm(int *inp, int *out, int size);
int binvec2dec(int *x, int l);
void multMatrix(double **M, double **A, int l_A, int c_A, double **B, int l_B, int c_B);
void multMatrixDI(double **M, double **A, int l_A, int c_A, int **B, int l_B, int c_B);
void transpose(double **Mt, double **M , int l , int c);
double **inverse(double **M_inv, int l, double **M);
int hammingDistance(int *x, int *y, int l);
// selection.cpp
int selection( population *pop );
// transformation.cpp
void mutation (allele *offspring, double p_mut);
void Point2X(allele *parent1, allele *parent2, allele *offspring1 );
void UX(allele *parent1, allele *parent2, allele *offspring1  );
// file_man.cpp
void file_output(char *prob_name);
void read_problem(char *prob_name);
// fitness.cpp
double compFitFS(allele *ind);
