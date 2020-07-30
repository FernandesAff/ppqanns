#include "pq_search.h"

// pq : estrutura do quantizador
// codebook : estrutura que armazena o codebook
// vquery : estrutura que contem os vetores da query
// k : numero de vetores proximos a serem retornados
// dis : vetor que vai receber as distancias entre os vetores
// ids : vetor que vai receber os indices

void pq_search(pqtipo pq, matI codebook, mat vquery, int k, float *dis, int *ids){

	// ids1 : vetor que armazena temporariamente o indice dos vetores mais proximos
	// dis1 : vetor que armazena temporariamente a distancia entre os vetores mais proximos e o vetor da fila
	// distab_temp : vetor que recebe temporariamente a tabela de distancias de um subespaco
	// vsub : estrutura que recebe um subvetor temporariamente
	// distab : estrutura que recebe todas as distancias tabeladas
	// disquerybase : estrutura que recebe as distancias entre os vetores da fila e os vetores mais proximos

	int i,
		j,
		*ids1;

	float 	*dis1,
			*distab_temp;

	mat vsub,
		distab,
		disquerybase;

	//definicao de variaveis

	vsub.mat= (float*)malloc(sizeof(float)*pq.ds);
	vsub.n=1;
	vsub.d=pq.ds;

	distab.mat = (float*)malloc(sizeof(float)*pq.ks*pq.nsq);
	distab.n = pq.nsq;
	distab.d = pq.ks;

	distab_temp = (float*)malloc(sizeof(float)*pq.ks);

	disquerybase.mat= (float*)malloc(sizeof(float)*codebook.n);
	disquerybase.n=codebook.n;
	disquerybase.d=1;

	dis1= (float*)malloc(sizeof(float)*k);
	ids1= (int*)malloc(sizeof(int)*k);

	for (i=0;i<vquery.n;i++){
		for (j=0;j<pq.nsq;j++){
			copySubVectors(vsub.mat,vquery ,pq.ds,j,i,i);
			compute_cross_distances (vsub.d, vsub.n, pq.ks, vsub.mat, &pq.centroids[j*pq.centroidsn], distab_temp);
			memcpy(distab.mat+j*pq.ks*vsub.n, distab_temp, sizeof(float)*pq.ks*vsub.n);
		}
		disquerybase.mat = sumidxtab(distab, codebook);
		k_min(disquerybase, k, dis1, ids1);

		for(int l=0; l<k; l++){
			memcpy(dis + vquery.n*l+i, dis1+l, sizeof(float));
		}
		for(int l=0; l<k; l++){
			memcpy(ids + vquery.n*l+i, ids1+l, sizeof(int));
		}
	}

	free(distab_temp);
	free(disquerybase.mat);
	free(dis1);
	free(ids1);
}

float* sumidxtab(mat distab, matI codebook){
	//aloca o vetor a ser retornado
	float *dis = (float*)malloc(sizeof(float)*codebook.n);
	int i, j;

	//soma as distancias para cada vetor
	for (i = 0; i < codebook.n ; i++) {
		float dis_tmp = 0;
		for(j=0; j<distab.n; j++){
			dis_tmp += distab.mat[codebook.mat[j+i*distab.n] + j*distab.d];
		}
		dis[i]=dis_tmp;
	}

	return dis;
}

void k_min (mat disquerybase, int k, float *dis, int *ids){
	int i,
		j,
		d,
		n=1;

	if(disquerybase.d==1 && disquerybase.n>1){
		d=disquerybase.n;
		n=1;
	}

	for (i=0; i<n; i++){
		fvec_k_min(disquerybase.mat, d, ids, k);
		for(j=0; j<k; j++){
			dis[j] = disquerybase.mat[ids[j]];
			ids[j]++;
		}
		ids += k;
		dis += k;
		disquerybase.mat += d;
	}
}
