/*******************************************************************************************************\
*					Class: Radial Basis Function Network Crossover    				 		  			*
* 	Copyright (C) 2019  Renato Tinos <rtinos@ffclrp.usp.br>									  			*
* 																							  			*
* Reference: Tinos, R. (2020), "Artificial Neural Network Based Crossover for Evolutionary Algorithms", *
*                             Submitted to Applied Soft Computing.                                      *
*																						  			    *
* for RBFN, see reference HAYKIN, S. (1998). "Neural Networks: A Comprehensive Foundation", 2nd Edition *
*																							  			*
* RBFX.h is free software: you can redistribute it and/or modify it						  	  			*
* under the terms of the GNU General Public License as published by the					  				*
* Free Software Foundation, either version 3 of the License, or						  	  				*
* (at your option) any later version.						  								  			*
* 						  																	 			*
* RBFX.h  is distributed in the hope that it will be useful, but						  	 			*
* WITHOUT ANY WARRANTY; without even the implied warranty of						  		  			*
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.						  				  			*
* See the GNU General Public License for more details.						  				  			*
* 						  																	  			*
* You should have received a copy of the GNU General Public License along					  			*
* with this program.  If not, see <http://www.gnu.org/licenses/>.						  	  			*
\*******************************************************************************************************/

#include <cmath>
#include <cstdlib>
#include <stdio.h>

class RBFX {
	private:	
		double **W_out;													// weight matrix between hidden layer and output layer	
		double **C;														// Centers of the hidden units		
		double **X;														// input (train.) set   
		double **D;														// desired (train.) set 
		double *fit_TS;													// fitness of the offspring used to add an example of the training set
		double *beta; 													// kernel width: a symmetric width for each unit is assumed                     
		int n_in;														// number of units in the input layer
		int n_hid;														// number of neurons in the hidden layer
		int n_out;														// number of neurons in the output layer
		int n_train;													// size of the current training set  
		double qgaussian(double x, double q);							// q-Gaussian Function			
		double gaussian(double x);										// Gaussian Function						
		double invmultquad(double x);									// Inverse Multiquadratic Function	
		double cauchy(double x);										// Cauchy Function	
		double radialBasisFunction(double x, int j);					// RBF (of x) used in the RBF network
		void RBFpreTrain(double **H);									// Pre-trainnning RBF: generate the activations of the neurons of hidden layer for all training examples
		void RBFoutput(double *x, double *y);							// compute the output (y) of the RBF network for input x
		void RBFupdateCenters(int c_op);								// update centers
		void RBFoperationPrint(double *x, double *h, double *y);		// Print NNet operation information	
		int closestExample(double fit_off);
	public:		
		int n_train_max;												// maximum size of the training set  
		int train_flag;													// flag: 0 if not trained; 1 if trained	
		long int i_train;												// number of sucesssful crossovers		
		RBFX(int n_in_par, int n_hid_par, int n_out_par, int n_train_max_par);
	    ~RBFX(void);
		void RBFtrain(int n_train_par);               					// RBF trainning
		void addTrainSet(int *parent1, int *parent2, int *offspring, double fit_off);	// add example to dataset
		void detRBFX(int *parent1, int *parent2, int *offspring);		// Deterministic RBFN Crossover (dRBFNX)
		void stoRBFX(int *parent1, int *parent2, int *offspring);		// Stochastic RBFN Crossover (sRBFNX)
		void sto2RBFX(int *parent1, int *parent2, int *offspring);		// Mixed RBFN Crossover (mRBFNX)
	    void RBFsave(int n_run, int n_gen, int cr_type, char *prob_name, int RBF_config_par);
		void RBFprint(void);
	    void RBFprintTrainSet(void);
};


/******************************************************************************\
*								Constructor									   *
\******************************************************************************/
RBFX::RBFX(int n_in_par, int n_hid_par, int n_out_par, int n_train_max_par){
	
	// Parameters of the architecture
	n_in=n_in_par;
	n_hid=n_hid_par;					// number of hidden neurons
	n_out=n_out_par;

	// Parameters of the datasets
	n_train_max=n_train_max_par;
	i_train=0;
	train_flag=0;						// flag: 0 if not trained; 1 if trained
	
    // Memory Allocation
    beta=aloc_vectord(n_hid+1);			// kernel width: for simplicity, a symmetric and constant width is assumed
	fit_TS=aloc_vectord(n_train_max);
	W_out=aloc_matrixd(n_hid+1,n_out);
	C=aloc_matrixd(n_hid+1,n_in);		// first line is not used
	X=aloc_matrixd(n_train_max,n_in);	
	D=aloc_matrixd(n_train_max,n_out);
				   	
}


/******************************************************************************\
*								 Destructor													   *
\******************************************************************************/
RBFX::~RBFX(void){	

	// Memory Desallocation
	delete [] beta;
	delete [] fit_TS;
	desaloc_matrixd(W_out,n_hid+1);
	desaloc_matrixd(C,n_hid+1);
	desaloc_matrixd(X,n_train_max);
	desaloc_matrixd(D,n_train_max);
}  


/************************************************************************************\
*								Radial Basis Funcions  								 *
\************************************************************************************/

double RBFX::radialBasisFunction(double x, int j)
{
	x=beta[j]*x;
	
	return (gaussian(x));			
}


/************************************************************************************\
*								 Gaussian Function 								 	 *
\************************************************************************************/
double RBFX::gaussian(double x)
{	
	return ( exp(-x) );
}


/******************************************************************************\
*					 Output (y) of the neural netwok for input x 		       *
\******************************************************************************/
void RBFX::RBFoutput(double *x, double *y){
	int i, j;
	double  u, *h, dist, var_u=1.0;
	
	h=aloc_vectord(n_hid+1);
		
	// Activation of the neurons in the hidden layer for input x
	h[0]=1.0;							// bias for output neurons
	for (j=1;j<=n_hid;j++) {
		// Mahalanobis distance with diagonal covariance matrix (or standardized Euclidean distance) raised to power 2
		u=0.0;
		for (i=0;i<n_in;i++){
			dist=x[i]-C[j][i];				
			u+= dist*dist/var_u;	
		}		
		// Activation: Radial Basis Function
		h[j]=radialBasisFunction(u,j);							
	}	

	// Activation of the neurons in the output layer for input x
	for (j=0;j<n_out;j++) {
		u =0.0;											
		for (i=0;i<=n_hid;i++)			
			u+= h[i]*W_out[i][j];			// internal activation
		y[j]=u;								// linear function					
	}	
	//RBFoperationPrint(x, h, y);
	
    delete [] h;  

}


/*****************************************************************************************************************************\
*	RBF: generate centers for the radial units	          																		  	  *
\*****************************************************************************************************************************/
void RBFX::RBFupdateCenters(int c_op){      
   int i, j, k, aux, n, t, j_chosen, n_min, *ex_index, nn, nneig=5;
   double u, dist, dist_min, u_min, neta=0.005, *dist_min_c; 
	
	ex_index=aloc_vectori(n_train);
	dist_min_c=aloc_vectord(nneig);
	
	// Adding centers
	if (c_op==0){
		// Initial: random selection of centers		
		for (j=1;j<=n_hid;j++){
			for (i=0;i<n_in;i++){
				C[j][i]=(random_dou()*2.0)-1.0; // random between -1.0 and 1.0
			}
		}
	}
	else{
		// After initial: Replace center of random unit 			
		// Replace: center with smaller distance to closest center
		for (j=1;j<n_hid;j++){
			for (k=j+1;k<=n_hid;k++){
				dist=0.0;
				for (i=0;i<n_in;i++)
					dist+=pow(C[j][i]-C[k][i],2);
				if ( (j==1 && k==(j+1)) || dist<dist_min){
					dist_min=dist;
					j_chosen=j;
				}
			}
		}
		// New center: random
		for (i=0;i<n_in;i++)
			C[j_chosen][i]=(random_dou()*2.0)-1.0; // random between -1.0 and 1.0			
				
	}
	
    // Update centers: adaptation of the self-organizing rule presented in [Haykin, 1998]
    // Update each center with the closest example of the training set    
    for (t=1;t<=n_hid;t++){    	 
	              
    	// finding nneig closest examples
    	for (n=0;n<n_train;n++)
			ex_index[n]=n;
    	
    	for (nn=0;nn<nneig;nn++){
    		n=ex_index[nn];
    		u=0.0;
		    for (i=0;i<n_in;i++){
				dist=X[n][i]-C[t][i];
				u+= dist*dist;
			}
			u_min=u;
			n_min=nn;
	    	for (k=nn+1;k<n_train;k++){
		    	n=ex_index[k];
				u=0.0;
		    	i=0;
		    	while (i<n_in && (u_min==0.0 || u<u_min) ){
		    		dist=X[n][i]-C[t][i];
					u+= dist*dist;	  
					i++;  		
				}
		    	if ( (u>0.0 && u<u_min) || u_min==0.0 ){
					u_min=u;
					n_min=k;
				}				
			}
			aux=ex_index[n_min];
			ex_index[n_min]=ex_index[nn];
			ex_index[nn]=aux;
			dist_min_c[nn]=sqrt(u_min);    		    		// Euclidean distance (used to compute width of the radial unit)
		}
		
		// computing width of the radial unit
		// similar to k-means, but only for the nearest neighbors: See https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
		u=0.0;
		for (nn=0;nn<nneig;nn++){
			u+=dist_min_c[nn];	
		}
		if (u>0.0){
			u=u/nneig;
			beta[t]=1.0/(2*u*u);
		}
		else{
			beta[t]=0.00001;
		}
		
    	// updating center
    	for (nn=0;nn<nneig;nn++){	
			n_min=ex_index[nn];
			for (i=0;i<n_in;i++)
				C[t][i]=C[t][i]+neta*(X[n_min][i]-C[t][i]);	
		}
						
	}	 
	
	delete [] ex_index;  
	delete [] dist_min_c;   	             						  						  		              	      
}


/*****************************************************************************************************************************\
*	RBF Pre-trainnning: compute the activations of the neurons of hidden layer for all training examples			          *
\*****************************************************************************************************************************/
void RBFX::RBFpreTrain(double **H){      
   	int i, j, n;
   	double u, dist, var_u=1.0;
              	              	
   	// Activation of the neurons in hidden layer for all examples of the training set
	for (n=0;n<n_train;n++) {
		H[n][0]=1.0;		// bias for the output neurons
		for (j=1;j<=n_hid;j++) {
			// Mahalanobis distance with diagonal covariance matrix (or standardized Euclidean distance) raised to power 2				
			u = 0.0;			
			for (i=0;i<n_in;i++){
				dist=X[n][i]-C[j][i];
				u+= dist*dist/var_u;
			}				
			// Activation Function
			H[n][j]=radialBasisFunction(u,j);		
		}
	}	      
}


/*****************************************************************************************************************************\
*	RBFN Training 			   																							  	  *
\*****************************************************************************************************************************/
void RBFX::RBFtrain(int n_train_par){
	int n_hid1;
	double **H, **Ht, **HtH, **HtH_i, **H_pinv;
  	   
  	n_train=n_train_par;		// size of the training set
	// Update the number of hidden neurons and update centers
	if (train_flag==0){
		RBFupdateCenters(0);
		train_flag=1; 
	}
	else{
		RBFupdateCenters(1);	
	}
  	n_hid1=n_hid+1;			// number of hidden neurons + 1  	
  
  	// Memory Allocation  	   
  	H=aloc_matrixd(n_train,n_hid1);				// matrix with the activations of the n_hid neurons of hidden layer during the trainning
	Ht=aloc_matrixd(n_hid1,n_train); 
	HtH=aloc_matrixd(n_hid1,n_hid1); 
    HtH_i=aloc_matrixd(n_hid1,n_hid1); 
    H_pinv=aloc_matrixd(n_hid1,n_train); 
		
	// Updating the centers of the radial units and computing H  
	RBFpreTrain(H);		
    		 		 		 	   
    // Regularized pseudoinverse of H    
	transpose(Ht, H, n_train, n_hid1);   								// Ht: transpose of H
	multMatrix(HtH, Ht, n_hid1, n_train, H, n_train, n_hid1);			// HtH: Ht*H
    inverse(HtH_i, n_hid1, HtH);										// HtH_i: inverse of HtH 
    multMatrix(H_pinv, HtH_i, n_hid1, n_hid1, Ht, n_hid1, n_train);		// H_pinv: Pseudo-inverse
    
	// Computing W_out 
    multMatrix(W_out, H_pinv, n_hid1, n_train, D, n_train, n_out);
	    
	// Memory Desallocation
    desaloc_matrixd(Ht,n_hid1); 
    desaloc_matrixd(HtH,n_hid1); 
    desaloc_matrixd(HtH_i,n_hid1);
    desaloc_matrixd(H_pinv,n_hid1);
	desaloc_matrixd (H,n_train);
	   
}


/******************************************************************\
*		find example with closest value of fit_TS		           *
\******************************************************************/
int RBFX::closestExample(double fit_off){
	int i, i_minf, i_ex;
	double minf, auxf;
	
	i_minf=random_int(0,n_train_max-1);;
	minf=fabs(fit_TS[i_minf]-fit_off);
	for (i=1;i<10;i++){
		i_ex=random_int(0,n_train_max-1);	
		auxf=fabs(fit_TS[i_ex]-fit_off);
		if(auxf<minf){
			i_minf=i_ex;
			minf=auxf;
		}		
	}
	
	return (i_minf);
}


/******************************************************************\
*		Add example to the training dataset 			           *
\******************************************************************/
void RBFX::addTrainSet(int *parent1, int *parent2, int *offspring, double fit_off) {
  int j, i_ex;
  double xmax=1.0, xmin=0.0, xmax_sc, xmin_sc, x_aux, x_aux2;
	
	xmax_sc=2.0*xmax-xmin;		// remember that the eq. is x=2*x1-x2
	xmin_sc=2.0*xmin-xmax;		// remember that the eq. is x=2*x1-x2
	x_aux=2.0/(xmax_sc-xmin_sc);
  
  	  
	if (i_train<n_train_max ){
	  	i_ex=i_train;
	}	
	else {
	  	//i_ex=random_int(0,n_train_max-1);		// replace random example
	  	i_ex=closestExample(fit_off);
	} 
	  
	// Input (one value for each combination of parents)
	for (j=0;j<n_in;j++){
		x_aux2=2.0*parent1[j]-parent2[j];
		X[i_ex][j]=x_aux*(x_aux2-xmin_sc)-1.0;			// scaling (-1,+1): x[i]=2.0*(x[i]-xmin_sc)/(xmax_sc-xmin_sc)-1.0;
	}
	  
	// Desired output (crossover mask)
	for (j=0;j<n_out;j++){
	  	if (parent1[j]==parent2[j]) 
	  		D[i_ex][j]=0.5;
		else if (parent1[j]==offspring[j])
			D[i_ex][j]=0.0;    	
		else
			D[i_ex][j]=1.0; 
	}
			  	
	i_train++;
}


/******************************************************************************\
*				Deterministic RBFN Crossover (dRBFNX)			   			   *
\******************************************************************************/
void RBFX::detRBFX(int *parent1, int *parent2, int *offspring){
	int i;
	double *mask, *x, xmax=1.0, xmin=0.0, xmax_sc, xmin_sc, x_aux, x_aux2;
	
	xmax_sc=2.0*xmax-xmin;		// remember that the eq. is x=2*x1-x2
	xmin_sc=2.0*xmin-xmax;		// remember that the eq. is x=2*x1-x2
	x_aux=2.0/(xmax_sc-xmin_sc);
		
	mask=aloc_vectord(n_out);
	x=aloc_vectord(n_in);
	
	// Input 
	for (i=0;i<n_in;i++){
		x_aux2=2.0*parent1[i]-parent2[i];
		x[i]=x_aux*(x_aux2-xmin_sc)-1.0;	// scaling (-1,+1): x[i]=2.0*(x[i]-xmin_sc)/(xmax_sc-xmin_sc)-1.0;
	}
	
	RBFoutput(x,mask);
		
	for (i=0;i<n_out;i++){		
		if (mask[i]<0.5)
			offspring[i]=parent1[i];
		else
			offspring[i]=parent2[i];
	}
		
	delete [] x;	
	delete [] mask;		
}


/******************************************************************************\
*				Stochastic RBFN Crossover (sRBFNX)	 					   	   *
\******************************************************************************/
void RBFX::stoRBFX(int *parent1, int *parent2, int *offspring){
	int  i;
	double *mask, *x, rand_aux, xmax=1.0, xmin=0.0, xmax_sc, xmin_sc, x_aux, x_aux2;
	
	xmax_sc=2.0*xmax-xmin;		// remember that the eq. is x=2*x1-x2
	xmin_sc=2.0*xmin-xmax;		// remember that the eq. is x=2*x1-x2
	x_aux=2.0/(xmax_sc-xmin_sc);
	
	mask=aloc_vectord(n_out);
	x=aloc_vectord(n_in);
	
	// Input
	for (i=0;i<n_in;i++){
		x_aux2=2.0*parent1[i]-parent2[i];
		x[i]=x_aux*(x_aux2-xmin_sc)-1.0;	// scaling (-1,+1): x[i]=2.0*(x[i]-xmin_sc)/(xmax_sc-xmin_sc)-1.0;
	}
	
	RBFoutput(x,mask);
		
	for (i=0;i<n_out;i++){
		// Stochastic with bias given by mask[i]
		rand_aux=random_dou();			// random double in [0.0,1.0]
		if ( rand_aux > mask[i])
			offspring[i]=parent1[i];
		else
			offspring[i]=parent2[i];		
	}

	delete [] x;	
	delete [] mask;		
}


/******************************************************************************\
*			Mixed RBFN Crossover (mRBFNX)  	   			       				   *
\******************************************************************************/
void RBFX::sto2RBFX(int *parent1, int *parent2, int *offspring){
	int  i;
	double *mask, *x, rand_aux, xmax=1.0, xmin=0.0, xmax_sc, xmin_sc, x_aux, x_aux2;
	
	xmax_sc=2.0*xmax-xmin;		// remember that the eq. is x=2*x1-x2
	xmin_sc=2.0*xmin-xmax;		// remember that the eq. is x=2*x1-x2
	x_aux=2.0/(xmax_sc-xmin_sc);
		
	mask=aloc_vectord(n_out);
	x=aloc_vectord(n_in);
	
	// Input 
	for (i=0;i<n_in;i++){
		x_aux2=2.0*parent1[i]-parent2[i];
		x[i]=x_aux*(x_aux2-xmin_sc)-1.0;	// scaling (-1,+1): x[i]=2.0*(x[i]-xmin_sc)/(xmax_sc-xmin_sc)-1.0;
	}
	
	RBFoutput(x,mask);
		
	for (i=0;i<n_out;i++){	
		if (mask[i]>0.2 && mask[i]<0.8){
			// Stochastic with bias given by mask[i]
			rand_aux=random_dou();			// random double in [0.0,1.0]		
			if ( rand_aux > mask[i])
				offspring[i]=parent1[i];
			else
				offspring[i]=parent2[i];
		}
		else{
			// Deterministic	
			if ( mask[i]<0.5 )
				offspring[i]=parent1[i];
			else
				offspring[i]=parent2[i];
		}
	}

	delete [] x;	
	delete [] mask;		
}


/******************************************************************************\
*								Save NNet information														   *
\******************************************************************************/
void RBFX::RBFsave(int n_run, int n_gen, int cr_type, char *prob_name, int RBF_config_par){
	int i, j;
	FILE *NN_file;
	char *name_p;
	char name[CHAR_LEN];

    name_p = name;
	sprintf(name,"NNt_%s_c%d_cg%d_run%d_gen%d.dat",prob_name,cross_type,RBF_config_par,n_run,n_gen);
	if ((NN_file = fopen(name_p,"w"))==NULL) {
		puts("The file NN to be saved cannot be open \n");
		exit(1);
	}
	// Save NN information
	fprintf(NN_file,"Parameters:\n"); 
	fprintf(NN_file,"%d %d %d \n",n_in, n_hid, n_out);      
	fprintf(NN_file,"RBF_type:\n");  
	// Save Centers of Hidden Neurons
	fprintf(NN_file,"beta:\n");
		for (j=1;j<=n_hid;j++) 
			fprintf(NN_file,"%1.12f ",beta[j]); 	 
	fprintf(NN_file,"\n");
	fprintf(NN_file,"C:\n"); 
	for (i=0;i<n_in;i++){
	      for (j=1;j<=n_hid;j++)
		        fprintf(NN_file,"%1.12f ",C[j][i]);
          fprintf(NN_file,"\n");         
	}   
	// Save W_out
	fprintf(NN_file,"W_out:\n");
	for (i=0;i<n_hid;i++){
	      for (j=0;j<n_out;j++)
		        fprintf(NN_file,"%1.12f ",W_out[i][j]);
          fprintf(NN_file,"\n");         
	}
	fclose(NN_file);
		
}

/******************************************************************************\
*								Print NNet information						   *
\******************************************************************************/
void RBFX::RBFprint(void){
	int i, j;
	
	cout<< "Radial Basis Funcion Network: "<<endl;
	cout<<" Number of Inputs: "<<n_in<<endl;
	cout<<" Number of Neurons in Hidden Layer: "<<n_hid<<endl;
	cout<<" Number of Neurons in Output Layer: "<<n_out<<endl;
	cout<<" Hidden Layer: "<<endl;
	for (j=1;j<=n_hid;j++){
			cout<<" Hidden Neuron: " <<j<<", beta="<<beta[j]<<endl;
			cout<<endl;	
			cout<<"  Center: ";	
			for (i=0;i<n_in;i++)
					cout<< C[j][i] << ", ";
			cout<<endl;		
	}

	cout<<" Output Layer: "<<endl;
	for (i=0;i<n_out;i++){
			cout<<" Output Neuron: " <<i<<endl;
			cout<<"  Weights: ";	
			for (j=0;j<=n_hid;j++)
					cout<< W_out[j][i] << ", ";
			cout<<endl;
	}
	
}


/******************************************************************************\
*								Print NNet operation information			   *
\******************************************************************************/
void RBFX::RBFoperationPrint(double *x, double *h, double *y){
	int i, j;
	
	cout<< "Radial Basis Funcion Network: "<<endl;
	cout<<" Inputs: "<<endl;
	for (i=0;i<n_in;i++)		
		cout<< x[i] << ", ";
	cout<<endl;
	cout<<" Activation of the Hidden Neurons: "<<endl;
	for (j=1;j<=n_hid;j++)
		cout<< h[j] << ", ";
	cout<<endl;
	cout<<" Outputs: "<<endl;
	for (i=0;i<n_out;i++)
		cout<< y[i] << ", ";
	cout<<endl;
	system("pause");
}


/******************************************************************************\
*								Print Training Set						   		*
\******************************************************************************/
void RBFX::RBFprintTrainSet(void){
	int i, j;
	
	
	cout<< "Training Set: "<<endl;
	for (i=0;i<n_train;i++){
			cout<<" Example: " <<i<<endl;
			cout<<"  Input: ";	
			for (j=0;j<n_in;j++)
					cout<<" "<< X[i][j];
			cout<<endl;
			cout<<"  Desired Output: ";	
			for (j=0;j<n_out;j++)
					cout<<" "<< D[i][j];
			cout<<endl;
			cout<<endl;
	}
	//system("pause");

}
