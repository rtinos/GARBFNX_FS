*** Genetic Algorithm with RBFN Crossover for Feature Selection ***

Description: This is the source code for the the Genetic Algorithm with RBFN Crossover for Feature Selection. 

Reference:  Tinos, R. (2020), "Artificial Neural Network Based Crossover for Evolutionary Algorithms", Submitted to Applied Soft Computing.

Contact: Renato Tinos <rtinos@ffclrp.usp.br>


Running the code: ./GARBFNX_FS <problem name> <classifier> <cross_type> <RBF_config>

<problem name>: name of the instance (dataset), without extension. An example of the dataset format is given in file ionosphere.dat. In the dataset, the inputs were normalized between 0 and 1 and the labels of the classes are integers (starting at 1).

<classifier>: classifier type (here, only KNN is used). 1: KNN with K=3; 2: KNN with K=5; 3: KNN with K=7.

<cross_type>: crossover type 1: 2X (2-point crossover); type 2: UX (uniform crossover); type 3: dRBFNX (deterministic RBFN crossover); type 4: sRBFNX (stochastic RBFN crossover); type 5: mRBFNX (mixed RBFN crossover).

<RBF_config>: controls the number of hidden neurons (radial units). The number of radial units is RBF_config*10. The maximum value is 120 radial units. It is not used for 2X and UX.


Example for running the code for: ionosphere (dataset ionosphere) 1 (KNN with K=3) 3 (dRBFNX) 8 (RBFN with 80 radial units) 

make

./GARBFNX_FS ionosphere 1 3 8


Observation 1: Class RBFNX is given in RBFX.h 

- Function void RBFX::RBFoutput(double *x, double *y) : Output (vector y) of the neural netwok for input vector x.
	
- Function void RBFX::RBFtrain(int n_train_par): train RBFNX with n_train_par training examples.

- Function void RBFX::addTrainSet(int *parent1, int *parent2, int *offspring, double fit_off): add example to the trainig set.
		
Observation 2: file global.cpp contains the parameters of the GA (examples: number of runs, population size, and crossover rate).

Observation 3: GARBFNX_FS generates 4 main files
 
- bfi_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config): best fitness found in each run
	
- bind_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config): best individuals found in each run

- time_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config): time for each run

- gen_%s_c%d_cg%d.dat",prob_name,cross_type,RBF_config): number of generations for each run

Other files are generate for statistics, e.g., successful recombination and improvement rates, and diversity of the population. 
The final RBFN is also saved ("NNt_%s_c%d_cg%d_run%d_gen%d.dat",prob_name,cross_type,RBF_config_par,n_run,n_gen).