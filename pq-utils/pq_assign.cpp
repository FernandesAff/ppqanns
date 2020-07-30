#include "pq_assign.h"

using namespace std;

void check_assign(){
    cout << ":::PQ_ASSIGN OK:::" << endl;
}

// pq : estrutura do quantizador
// v : base de vetores a ser usada

matI pq_assign (pqtipo pq, mat v){

    // assigns : indice dos k elementos mais proximos
    // dis : distancia entre os k elementos mais proximos e os definitivos
    // vsub : estrutura que contém temporariamente os subvetores
    // code : estrutura que armazena o codebook

    int i,
        *assigns;

    float *dis;

    mat vsub;

    matI code;

    // definicao de variaveis

    vsub.mat = (float*)malloc(sizeof(float)*(pq.ds)*(v.n));
    vsub.d=pq.ds;
    vsub.n=v.n;
    assigns = (int*)malloc(sizeof(int)*v.n);
    dis = (float*)malloc(sizeof(float)*v.n);
    code.mat = (int*)malloc(sizeof(int)*v.n*pq.nsq);
    code.n = v.n;
    code.d = 0;

    // criacao do codebook

    for (i = 0; i < pq.nsq ; i++) {
	
	copySubVectors(vsub.mat, v, pq.ds,i, 0, (v.n)-1);
        
	knn_full(2, vsub.n, pq.ks, pq.ds, 1 , pq.centroids+i*pq.centroidsn, vsub.mat, NULL , assigns, dis);
        
	ivec_concat_transp(code,assigns,pq.nsq);
        
	code.d++;
    }


    free(vsub.mat);
    free(assigns);
    free(dis);

    return code;
}
