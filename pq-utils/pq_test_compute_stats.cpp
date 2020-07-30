#include "pq_test_compute_stats.h"

void pq_test_compute_stats (int *ids, matI ids_gnd, int k){
	int *nn_ranks_pqc,
		max;

	float 	r_at_i;

	nn_ranks_pqc= (int*) malloc(sizeof(int)*ids_gnd.n);

	

	for (int i=0; i<ids_gnd.n; i++){
		nn_ranks_pqc[i]=k+1;
		for(int j=0; j<k; j++){
			if(ids[i*k+j]==ids_gnd.mat[i]){
				nn_ranks_pqc[i]=j;
			}
		}
	}

	sort(nn_ranks_pqc, nn_ranks_pqc+ids_gnd.n);

	for(int i=0; i<13; i++){

		switch (i){
			case 0:
				max=1;
				break;
			case 1:
				max=2;
				break;
			case 2:
				max=5;
				break;
			case 3:
				max=10;
				break;
			case 4:
				max=20;
				break;
			case 5:
				max=50;
				break;
			case 6:
				max=100;
				break;
			case 7:
				max=200;
				break;
			case 8:
				max=500;
				break;
			case 9:
				max=1000;
				break;
			case 10:
				max=2000;
				break;
			case 11:
				max=5000;
				break;
			case 12:
				max=10000;
				break;
		}
		float length=0;
		if(max<=k){
			for(int j=0; j<ids_gnd.n; j++){
				if(nn_ranks_pqc[j]<max && nn_ranks_pqc[j]<k)length++;
			}
			r_at_i = (length/ids_gnd.n)*100;
			printf("r@%d = %.3f\n",max, r_at_i);
		}
	}
}
