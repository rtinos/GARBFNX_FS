/******************************************************************************\
*								 Feature Selection Problem					   *
\******************************************************************************/
#include "defs.h"
#include <cstdlib>
#include <cmath>

// type: object (for classifier KNN)
typedef struct{
	int inde;
	double dist;	
} x_KNN;

x_KNN *aloc_vectorx_KNN(int lines)
{
	x_KNN *vector;

	vector = new x_KNN[lines];
	if (!vector) {
		cout<<"Allocation Error!"<<endl;
		exit(1);
	}
	return vector;
}

/******************************************************************************\
*				Euclidean Distance raised to 2						   		   *
\******************************************************************************/
double eucDist2(int i, int j, int *at_list, int at_list_size){
	int k;
	double sum=0.0;

	for (k=0;k<at_list_size;k++)
		sum=sum+pow( (X_dataset[i][at_list[k]]-X_dataset[j][at_list[k]]), 2 );

	return sum;
}


/******************************************************************************\
*			K-Nearest Neighbors Algorithm							           *
\******************************************************************************/
int KNN(int i, int *ex_list, int n_ex_list_size, int *at_list, int at_list_size, int K, double **Dist){
	int j, k, *freq_class, ind_min, max_c;
	x_KNN *VKNN, aux_d;
		
	freq_class=aloc_vectori(n_classes+1);
	VKNN=aloc_vectorx_KNN(n_ex_list_size);
	
	// finding the distances to the i-th example
	for (j=0;j<n_ex_list_size;j++){
		VKNN[j].inde=ex_list[j];
		VKNN[j].dist=Dist[i][VKNN[j].inde];
	}
	
	// sorting K elements
	for (k=0;k<K;k++){	
		ind_min=k;		
		for (j=k+1;j<n_ex_list_size;j++){	
			if (VKNN[j].dist<VKNN[ind_min].dist)
				ind_min=j;				
		}
		aux_d=VKNN[k];
		VKNN[k]=VKNN[ind_min];
		VKNN[ind_min]=aux_d;
	}
	
	// finding most frequent class
	for (j=1;j<=n_classes;j++)
		freq_class[j]=0;			
	for (k=0;k<K;k++)
		freq_class[ d_dataset[VKNN[k].inde]  ] +=1;
	max_c=1;
	for (j=2;j<=n_classes;j++)
		if (freq_class[j]>freq_class[max_c])
			max_c=j;

	delete [] VKNN;
	delete [] freq_class;
	
	return max_c;
}


/******************************************************************************\
*			k-Fold Cross Validation									           *
\******************************************************************************/
double kfold_eval(int *at_list, int at_list_size, int n_folds, double **Dist){
	int i, j, size_fold_max, i_beg, i_end=-1, n_hits=0, K; 
	int n_ex_train, n_ex_test, *ex_list_train, *ex_list_test, class_predict;
	
	size_fold_max = ceil(n_examples/n_folds);
	
	ex_list_train=aloc_vectori(n_examples);				
	ex_list_test=aloc_vectori(size_fold_max);		
	
	// Classifiers
	// 1: KNN with K=3; 2: KNN with K=5; 3: KNN with K=7
	if (classifier==1)
		K=3;
	else if (classifier==2)
		K=5;
	else if (classifier==3)
		K=7;
			
	for (j=0;j<n_folds;j++){		
		// Building training and test subsets
		i_beg=i_end+1;
		i_end=i_beg+size_fold_max-1;
		if (i_end>n_examples-1)
			i_end=n_examples-1;		
		n_ex_train=0;
		n_ex_test=0;
		for (i=0;i<n_examples;i++){
			if (i<i_beg || i>i_end){
				ex_list_train[n_ex_train]=i;
				n_ex_train++;				
			}
			else {
				ex_list_test[n_ex_test]=i;
				n_ex_test++;					
			}			
		}
		
		// Classification and number of hits
		for (i=0;i<n_ex_test;i++){			
			if (classifier<=3){
				class_predict=KNN(ex_list_test[i], ex_list_train, n_ex_train, at_list, at_list_size, K, Dist);
				//cout<<"class_predict: "<<class_predict<<", d_dataset[ex_list_test[i]]: "<<d_dataset[ex_list_test[i]]<<endl;					
				if (class_predict==d_dataset[ex_list_test[i]])
					n_hits++;
			}
		}		
					
	}	
	//cout<<"nhits: "<<n_hits<<endl;
	
	delete [] ex_list_train;
	delete [] ex_list_test;
	
	return ( (double) n_hits/n_examples);	
}


/******************************************************************************\
*			Fitness for the Feature Selection Problem				           *
\******************************************************************************/
double compFitFS( allele *ind ){
	int i, j=0, *at_list, at_list_size=0,  n_folds=10;
	double accuracy, f, **Dist;
	
	// finding number of attributes
	for (i=0;i<lcrom;i++)
		at_list_size+=ind[i];
		
	//cout<<"at_list_size: "<<at_list_size<<endl;	
	if (at_list_size==0)
		return (0.0);

	// list of attributes
	at_list=aloc_vectori(at_list_size);
	for (i=0;i<lcrom;i++){
		if (ind[i]==1){
			at_list[j]=i;
			//cout<<at_list[j]<<", ";
			j++;
		}
	}
	//	cout<<endl;
	
	// distance matrix
	Dist=aloc_matrixd(n_examples,n_examples);
	for (i=0;i<n_examples;i++){
	 	Dist[i][i]=0.0;
	 	for (j=i+1;j<n_examples;j++){
	 		Dist[i][j]=eucDist2(i, j, at_list, at_list_size);
	 		Dist[j][i]=Dist[i][j];
	 	}
	}
							
	// call k-fold cross-validation
	accuracy=kfold_eval(at_list, at_list_size, n_folds, Dist);
	//	cout<<"accuracy: "<<accuracy<<", lcrom: "<<lcrom<<", natrib: "<<(at_list_size)<<", stern: "<<( (double) (lcrom-at_list_size) )/lcrom<<endl;
	
	
	// compute fitness
	f = 0.9*accuracy + 0.1*( (double) (lcrom-at_list_size)) /lcrom;				
	
	desaloc_matrixd(Dist,n_examples);
	delete [] at_list;
	
	return f;
}

