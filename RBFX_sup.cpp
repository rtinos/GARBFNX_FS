/* -*- Mode: C; indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4 -*-  */
/*
 * Radial Basis Function Network Crossover - Using Supervised Learning
 * Test problem: Feature Selection
 *
 * Copyright (C) 2019  Renato Tinos <rtinos@ffclrp.usp.br>
 * 
* Reference: Tinos, R. (2020), "Artificial Neural Network Based Crossover for Evolutionary Algorithms", 
*                             Submitted to Applied Soft Computing.                                      
 * 
 * RBFX_fs is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * RBFX_fs  is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#include "defs.h"
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <float.h>
#include "RBFX.h"			// Radial Basis Function Network Crossover class


/******************************************************************************\
*				  	Print data							 			 *
\******************************************************************************/
void print_data(population *pop, int n_run ){

	
	cout <<"Generation:"<< gen << ", run: "<<n_run<<endl;
	cout <<"Best individual:"<< pop->best_individual << endl;
	cout <<"Fitness of the best individual:"<< pop->max_fitness << endl;
	cout <<"Mean fitness: "<< pop->mean_fitness << endl;	
	/*	
	int i, gene;
	for (i=0;i<pop->popsize ;i++) {	
		cout <<"("<< pop->ind[i].fitness<<") " ;
		for (gene=0;gene<lcrom ;gene++) 
			cout << pop->ind[i].chromosome[gene]<<" ";
		cout << endl;
	}*/
}
 
 
/******************************************************************************\
*								Fitness Computation    				   		   *
\******************************************************************************/
double compFitness( allele *ind ){
		double Fitness;

		Fitness = compFitFS( ind );		// fitness function for Feature selection Problem

		return Fitness;
}


/******************************************************************************\
*								 Crossover	 									*
\******************************************************************************/
double crossover( RBFX *RBFXop, allele *parent1, allele *parent2, allele *offspring, double fitness_p1, double fitness_p2, int n_run ){
	int cross_type_aux;				// crossover type 1: 2X; type 2: UX; type 3: dRBFNX; type 4: sRBFNX; type 5: mRBFNX 
	double r_aux, fitness_off;
	
	// Decision about crossover type
	cross_type_aux=cross_type;
	if ( cross_type_aux==3 || cross_type_aux==4 ) {
		// for sRBFNX and dRBFNX (random with same probability)
		r_aux=random_dou();
		if (RBFXop->train_flag==0){
			// if RBFN has not been trained yet, use  UX
			cross_type_aux=2;			
		}
		else{
			// if RBFN has been trained, choose RBFNX (with probability) 0.5 or UX (with probability 0.5)
			if (r_aux<0.5)
				cross_type_aux=2;
		}		
	}	
	else if (cross_type_aux==5){
		// for mRBFNX (random with different probabilities)		
		if (RBFXop->train_flag==0){
			// if RBFN has not been trained yet, use  UX
			cross_type_aux=2;		
		}
		else{
			// if RBFN has been trained, choose mRBFNX (with probability) 0.3 or UX (with probability 0.7)
			r_aux=random_dou();		
			if (r_aux<0.7)
				cross_type_aux=2;			
		}		
	}

	// Generating offspring by crossover
	if (cross_type_aux==1){
		// 2-point Crossover
		Point2X(parent1,parent2,offspring);				
	}
	else if (cross_type_aux==2){
		// Uniform Crossover
		UX(parent1,parent2,offspring);		
	}
	else if (cross_type_aux==3){
		// Deterministic RBFN Crossover (dRBFNX)
		RBFXop->detRBFX(parent1,parent2,offspring);	
	}
	else if ( cross_type_aux==4){
		// Stochastic RBFN Crossover (sRBFNX)
		RBFXop->stoRBFX(parent1,parent2,offspring);		
	}
	else if ( cross_type_aux==5){
		// Mixed RBFNX Crossover (mRBFNX)
		RBFXop->sto2RBFX(parent1,parent2,offspring);		
	}
			
	// fitness computation 
	fitness_off=compFitness(offspring);									// fitness computation	
	
	// Successful recombination rate: Check if resulted in improvement from parents
	if ( (fitness_off-fitness_p1)>EPS2 && (fitness_off-fitness_p2)>EPS2 ){
		if (cross_type==3 || cross_type==4 || cross_type==5){
			RBFXop->addTrainSet( parent1,parent2,offspring,fitness_off);	// add example to the RBFN training set	
		}
		if (cross_type==3 || cross_type==4){
			if (RBFXop->train_flag==1){			
				// record succesfull recombination rate for each type only after RBFN is trained
				if (cross_type_aux<4)
					sum_sucRate_crType[cross_type_aux-1]=sum_sucRate_crType[cross_type_aux-1]+1;
				else
					sum_sucRate_crType[2]=sum_sucRate_crType[2]+1;				
			}
		}
		sum_sucRate_all=sum_sucRate_all+1;	// Record successful recombination rate
	}
	// Improvement rate: check if resulted in improvement from best found individual 
	if ( (fitness_off-file_best_fitness[n_run])>EPS2 ){
		if (cross_type==3 || cross_type==4 ){
			if (RBFXop->train_flag==1){
				// record improvement rate for each type only after RBF is trained
				if (cross_type_aux<4)
					sum_impRate_crType[cross_type_aux-1]=sum_impRate_crType[cross_type_aux-1]+1;
				else
					sum_impRate_crType[2]=sum_impRate_crType[2]+1;
			}
		}
		sum_impRate_all=sum_impRate_all+1; // Record improvement rate		
	}
	// Total of recombinations
	if (cross_type==3 || cross_type==4){
		if (RBFXop->train_flag==1){
			// number of recombinations for each crossover type; record only after training the RBF
			if (cross_type_aux<4)
				sum_cross_crType[cross_type_aux-1]=sum_cross_crType[cross_type_aux-1]+1;
			else
				sum_cross_crType[2]=sum_cross_crType[2]+1;
		}
	}
	sum_cross_all=sum_cross_all+1;	// total number of recombinations
	
	return fitness_off;
	
}


/******************************************************************************\
*								 Generation of the GA							 *
\******************************************************************************/
void generation(double p_mut, int n_run, RBFX *RBFXop){
	int gene, i, j=0 , parent1, parent2;
		
	do {	
		// Reproduction
		if ( random_dou () > p_cross ){		
			// Selection of one parent (using redundant fitness)
			parent1=selection( &popold );
			// Mutation
			for (gene=0;gene<lcrom;gene++)
				popnew.ind[j].chromosome[gene]=popold.ind[parent1].chromosome[gene];
			mutation( popnew.ind[j].chromosome, p_mut );	
			popnew.ind[j].fitness=compFitness(popnew.ind[j].chromosome);										// fitness computation		
		}
		else {
			// Selection of two parents (using redundant fitness)
			parent1=selection( &popold );
			parent2=selection( &popold );
			// Crossover
			popnew.ind[j].fitness=crossover(  RBFXop, popold.ind[parent1].chromosome , popold.ind[parent2].chromosome ,  popnew.ind[j].chromosome, popold.ind[parent1].fitness, popold.ind[parent2].fitness, n_run);													// fitness computation			
		}	
		// check if improved best individual
		if (popnew.ind[j].fitness > file_best_fitness[n_run]){		
			file_best_fitness[n_run] = popnew.ind[j].fitness;
			for(i=0;i<lcrom;i++)
				File_best_ind[n_run][i]=popnew.ind[j].chromosome[i];
		}				
		
		j = j + 1;	

	} while ( j < popsize-1);

	// Elitism (j=popsize-1)
	for (gene=0;gene<lcrom;gene++)
		popnew.ind[j].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
	popnew.ind[j].fitness=popold.ind[popold.best_individual].fitness;	
	
}				


/******************************************************************************\
*				  	Random individual					 				 	   *
\******************************************************************************/
void randomInd(int num_ind){
	int gene;
	
	for (gene=0;gene<lcrom;gene++){
		//popold.ind[num_ind].chromosome[gene] = random_int(0,1);
		if (random_dou()<0.1){
			popold.ind[num_ind].chromosome[gene] = 1;
		}
		else{
			popold.ind[num_ind].chromosome[gene] = 0;
		}
	}
     	

    popold.ind[num_ind].fitness = compFitness( popold.ind[num_ind].chromosome );											
	
}


/******************************************************************************\
*				  	Initiate Population 					 				 *
\******************************************************************************/
void initiatePop(int n_run){
	int num_ind, i;
	
	// Size of the populations
	popold.popsize=popsize;
	popnew.popsize=popsize;
	poplopt.popsize=0;
			
	// Dynamic allocation: populations
	popold.ind = aloc_vectorind(popsize);
	popnew.ind = aloc_vectorind(popsize);
	poplopt.ind = aloc_vectorind(popsize);
	
	for (num_ind=0;num_ind<popsize;num_ind++){
		// Dynamic allocation: chromosomes	
		popold.ind[num_ind].chromosome = aloc_vectori(lcrom);
		popnew.ind[num_ind].chromosome = aloc_vectori(lcrom);
		poplopt.ind[num_ind].chromosome = aloc_vectori(lcrom);

		// Random Initialization
		randomInd(num_ind);	 	
      	
	 }
	 file_best_fitness[n_run]=popold.ind[0].fitness;
	for(i=0;i<lcrom;i++)
		File_best_ind[n_run][i]=popold.ind[0].chromosome[i];
	 statistics( &popold, n_run);
	 //print_data(&popold, n_run);
}


/******************************************************************************\
*				   Population: desallocate memory 					 				 *
\******************************************************************************/
void endPop(void){
	int num_ind; 
	
	for (num_ind=0;num_ind<popsize;num_ind++){
		delete [] popold.ind[num_ind].chromosome;
		delete [] popnew.ind[num_ind].chromosome;
		delete [] poplopt.ind[num_ind].chromosome;
	}
	delete [] popold.ind;
	delete [] popnew.ind;
	delete [] poplopt.ind;
}


/******************************************************************************\
*				  	Copy Population							 			 *
\******************************************************************************/
void copy_pop( void ){
	int i, gene;
		
	for (i=0;i<popnew.popsize;i++) {	
		popold.ind[i].fitness=popnew.ind[i].fitness;
		for (gene=0;gene<lcrom;gene++) 
			popold.ind[i].chromosome[gene]=popnew.ind[i].chromosome[gene];
	}
	
}


/******************************************************************************\
*				  	Run of the GA 			 *
\******************************************************************************/
void ga(int n_run, double p_mut, char *prob_name){	
	int num_ind, tau, last_change_gen=0, gene, i, num_ind_fin, min_trainSet, max_trainSet, tau_train, count_train=0;
	clock_t time_start;
	// RBFN Parameters
	int n_hid_neurons;
						
	// Initializing
	gen = 0;
	tau=0.08*max_gen;
	tau_train=0.1*max_gen;
	
	// RBFX initialization
	n_hid_neurons=RBF_config*10; 		// number of neurons in the hidden layer
	min_trainSet=100;				// minimum size of the training set	
	max_trainSet=500;			// maximum size of the training set			
	RBFX *RBFXop = new RBFX(lcrom , n_hid_neurons, lcrom, max_trainSet);			//  initiate operator from class RBFX (RBFX.h): 																							
																					//   	parameters: int n_in_par, int n_hid_par, int n_out_par,	int n_train_par_max

	for (i=0;i<3;i++){	
		sum_sucRate_crType[i]=0;
		sum_impRate_crType[i]=0;
		sum_cross_crType[i]=0;
	}
	sum_sucRate_all=0;
	sum_impRate_all=0;
	sum_cross_all=0;

	time_start=clock();	
	initiatePop(n_run);									// initiate population	
		
	if (save_datagen_flag==1 && n_run==0){
		file_best_fitness_gen[gen]=popold.max_fitness;	
		file_diversity_gen[gen]=popDiversity(&popold);
	}		 
			
	// Genetic Algorithm
	do {
		gen = gen + 1; 								// generation index		
		//cout<<"gen: "<<gen<<endl;		
		
		// Check convergence (tau generations without improving best solution)
		if (  (gen-last_change_gen) > tau ){
			last_change_gen=gen;
									
			// Storing local optima in local optima population
			if (pop_lopt_flag==1){
				if (poplopt.popsize<popsize)
					num_ind=poplopt.popsize;			// add individual to position poplopt.popsize
				else
					num_ind=popsize-1;					// replace last individual	
				for (gene=0;gene<lcrom;gene++)
					poplopt.ind[num_ind].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
				poplopt.ind[num_ind].fitness=popold.ind[popold.best_individual].fitness;
				poplopt.popsize=num_ind+1;
			}				
			// Elitism and Random Immigrants
			//cout<<"Introducing random immigrants....generation "<<gen<<endl;
			if (imig_rate<1.0){					
				for (gene=0;gene<lcrom;gene++)
					popold.ind[0].chromosome[gene]=popold.ind[popold.best_individual].chromosome[gene];
				popold.ind[0].fitness=popold.ind[popold.best_individual].fitness;
				// Introducing Random Immigrants (100*imig_rate % of the population)
				num_ind_fin=imig_rate*popold.popsize+1;
				if (num_ind_fin>popsize)
					num_ind_fin=popsize;
				for (num_ind=1;num_ind<num_ind_fin;num_ind++)
					randomInd(num_ind);
			}
			else {
				// add past local optima to current generation only in the final 1/4 of evolution
				if (pop_lopt_flag==1 && gen>(0.75*max_gen) && poplopt.popsize>=1){
					for (num_ind=0;num_ind<0.2*popold.popsize;num_ind++){		// up to 20% from lopt population
						i=random_int(0,poplopt.popsize-1);
						for (gene=0;gene<lcrom;gene++)
							popold.ind[num_ind].chromosome[gene]=poplopt.ind[i].chromosome[gene];
						popold.ind[num_ind].fitness=poplopt.ind[i].fitness;
					}
					num_ind_fin=num_ind;
				}				
				else{
					num_ind_fin=0;
				}
				// random immigrants	
				for (num_ind=num_ind_fin;num_ind<popsize;num_ind++)
					randomInd(num_ind);
			}
				
			statistics( &popold, n_run );										
		}
		
		generation( p_mut, n_run, RBFXop);
		
		// Train RBFN
		if (RBFXop->train_flag==0 || count_train>=tau_train){
			count_train=0;
			if (cross_type==3 || cross_type==4 || cross_type==5){
				//cout<<"cross: "<<RBFXop->i_train<<", "<<RBFXop->n_train<<endl;
				if (RBFXop->i_train>=min_trainSet){
					//cout<<"Training RBFN....generation "<<gen<<"...i_train "<<RBFXop->i_train<<"...n_train_max "<<RBFXop->n_train_max<<endl;					
					if (RBFXop->i_train>=RBFXop->n_train_max)
						RBFXop->RBFtrain(RBFXop->n_train_max);
					else
						RBFXop->RBFtrain(RBFXop->i_train);	
					//RBFXop->RBFprintTrainSet();
					//RBFXop->RBFprint();	
					//RBFXop->RBFsave(n_run,gen,cross_type,prob_type,lcrom,K,RBF_config);				 
				}
			}
		}
		count_train++;
						
		copy_pop();		// popold=popnew
		
		statistics( &popold , n_run);			
		//print_data(&popold, n_run);
			
		// Save data for generation: only for the first run
		if (save_datagen_flag==1 && n_run==0){			
			// Statistics across the generations
			if (gen<max_gen){			
				file_best_fitness_gen[gen]=popold.max_fitness;	
				file_diversity_gen[gen]=popDiversity(&popold);	
				if (cross_type==3 || cross_type==4 || cross_type==2){
					for (i=0;i<3;i++){
						if (sum_cross_crType[i]>0)
							File_sucRate_crType_gens[gen][i]=((double) sum_sucRate_crType[i])/sum_cross_crType[i];
						else 
							File_sucRate_crType_gens[gen][i]=-1.0;
					}
				}										
			}		
		}			
			
	} while (  gen < max_gen);
	
	//print_data(&popold, n_run);
	
	// Data to be saved
	time_run[n_run] = double( clock() - time_start ) / (double)CLOCKS_PER_SEC;
	file_gen[n_run] = gen;
	if ( (cross_type==3 || cross_type==4 || cross_type==7 || cross_type==8 || cross_type==9 || cross_type==10) && n_run==0 )
		RBFXop->RBFsave(n_run,gen,cross_type,prob_name,RBF_config);
	if ( cross_type==3 || cross_type==4 ){
		for (i=0;i<3;i++){
			if (sum_cross_crType[i]>0){
				File_sucRate_crType[n_run][i]= ((double) sum_sucRate_crType[i])/sum_cross_crType[i];
				File_impRate_crType[n_run][i]= ((double) sum_impRate_crType[i])/sum_cross_crType[i];
			}						
			else {
				File_sucRate_crType[n_run][i]=-1.0;
				File_impRate_crType[n_run][i]=-1.0;
			}
		}
	}
	if (sum_cross_all>0){
		File_sucRate_all[n_run]=((double) sum_sucRate_all)/sum_cross_all;
		File_impRate_all[n_run]=((double) sum_impRate_all)/sum_cross_all;
	}
	else{
		File_sucRate_all[n_run]=-1.0;
		File_impRate_all[n_run]=-1.0;		
	}
				
	endPop();		// population: desallocation 
	delete RBFXop;	
}


/******************************************************************************\
*				  	Main													 *
\******************************************************************************/
int main(int argc , char *argv[])
{
	int i, n_run=0;
	double p_mut;
	char *prob_name;

	// Arguments
	if( argc < 5) {
		cout<<"Insufficient number of arguments!"<<endl;
		cout<<"Call: RBFX_fs <problem name - without extension> <classifier> <cross_type> <RBF_config>"<<endl;
		exit(1);
	}
	else{
		prob_name=argv[1];
		classifier=atoi(argv[2]);
		cross_type=atoi(argv[3]);
		RBF_config=atoi(argv[4]);
		if ( classifier<1 || classifier>3 || cross_type<1 || cross_type>5 || RBF_config<1 || RBF_config>12) {
			cout<<"Incorrect arguments!"<<endl;
			cout<<"Call: RBFX_fs  <problem name - without extension> <classifier> (>=1 && <=3) <cross_type> (>=1 && <=5) <RBF_config> (>=1 && <=12)"<<endl;
			exit(1);
		}
	}	
	
	// load dataset
	read_problem(prob_name);
	
	// Parameters 
	p_mut=2.0/lcrom;						// mutation rate  
	max_gen=lcrom*10;						// maximum number of generations (when stop criterion is time, it is used for saving data per gen.)	
		
	// vector of integers (used in different methods)
	temp_v=aloc_vectori(lcrom);
	for (i=0;i<lcrom;i++)
		temp_v[i]=i;
	
	// Allocation of vectors for the data to be stored
	file_best_fitness_gen=aloc_vectord(max_gen);
	file_diversity_gen=aloc_vectord(max_gen);
	file_best_fitness=aloc_vectord(n_runs_max);
	file_gen=aloc_vectori(n_runs_max);
	time_run=aloc_vectord(n_runs_max);
	File_sucRate_all=aloc_vectord(n_runs_max);
	File_impRate_all=aloc_vectord(n_runs_max);
	File_best_ind=aloc_matrixi (n_runs_max,lcrom);
	File_sucRate_crType=aloc_matrixd (n_runs_max,3);
	File_sucRate_crType_gens=aloc_matrixd (max_gen,3);
	File_impRate_crType=aloc_matrixd (n_runs_max,3);

	cout << "\n ***** Genetic Algorithm ****" << endl;
	cout << "Feature Selection, Dataset: "<<prob_name << endl;
	cout << "cross_type="<<cross_type<< endl;
	
	for (n_run=0;n_run<n_runs_max;n_run++) {	
		srand(n_run+1);	// random seed   		
		cout <<"Run:"<< n_run << endl;
		ga(n_run,p_mut,prob_name);										// run of the ga
	}

	file_output(prob_name);												// save data

	desaloc_matrixi (File_best_ind,n_runs_max);
	desaloc_matrixd (File_sucRate_crType,n_runs_max);
	desaloc_matrixd (File_sucRate_crType_gens,max_gen);
	desaloc_matrixd (File_impRate_crType,n_runs_max);
	desaloc_matrixd (X_dataset,lcrom);
	delete [] d_dataset;
	delete [] time_run;
	delete [] file_gen;
	delete [] file_best_fitness;
	delete [] file_diversity_gen;
	delete [] file_best_fitness_gen;
	delete [] temp_v;
	delete [] File_sucRate_all;
	delete [] File_impRate_all;
		
	return 0;
}
