/******************************************************************************\
*								 Statistics						 			   *
\******************************************************************************/
#include "defs.h"
#include <cstdlib>


/******************************************************************************\
*		statistics for the population					  	   	 		       *
\******************************************************************************/
void statistics( population *pop, int n_run )
{	
	int i, j;

	pop->sum_fitness = pop->ind[0].fitness; 			// sum of the fitness in the population
	pop->max_fitness = pop->ind[0].fitness;   			// maximum fitness in the population
	pop->best_individual = 0;							// best individual in the population
	if (pop->ind[0].fitness > file_best_fitness[n_run]){
			file_best_fitness[n_run] = pop->ind[0].fitness;
			for(i=0;i<lcrom;i++) 
				File_best_ind[n_run][i]=pop->ind[0].chromosome[i];
	}			
	for(j=1;j<pop->popsize;j++) {
		pop->sum_fitness = pop->sum_fitness + pop->ind[j].fitness;
		if (pop->ind[j].fitness > pop->max_fitness )	{	
			pop->max_fitness = pop->ind[j].fitness; 
			pop->best_individual = j;			
		}
		if (pop->ind[j].fitness > file_best_fitness[n_run]){		
			file_best_fitness[n_run] = pop->ind[j].fitness;
			for(i=0;i<lcrom;i++)
				File_best_ind[n_run][i]=pop->ind[j].chromosome[i];
		}
	}

	pop->mean_fitness = pop->sum_fitness / popsize; 	// mean fitness in the population
	
}

/******************************************************************************\
*		Compute population diversity using Hamming distance	   	 		       *
\******************************************************************************/
double popDiversity( population *pop){
	int j, k;
	double diversity=0.0;
	
	for(j=0;j<pop->popsize-1;j++) 
		for(k=j+1;k<pop->popsize;k++) 
			diversity=diversity+hammingDistance(pop->ind[j].chromosome, pop->ind[k].chromosome, lcrom);
	diversity=diversity/lcrom;
	
	if (pop->popsize>1)
		diversity = diversity/( ((pop->popsize-1)*pop->popsize)/2.0 );			// sum of the Hamming distance divided by the sum of pairs
	else
		diversity = 0.0;
		
	return (diversity);	
	
}
