/*
Copyright © INRIA 2009-2014.
Authors: Matthijs Douze & Herve Jegou
Contact: matthijs.douze@inria.fr  herve.jegou@inria.fr

This file is part of Yael.

    Yael is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Yael is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Yael.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "vector.h"


#ifdef __linux__
#include <malloc.h>
#else
static void *memalign(size_t ignored,size_t nbytes) {
  return malloc(nbytes);
}
#endif


/*-------------------------------------------------------------*/
/* Allocation                                                  */
/*-------------------------------------------------------------*/



float *fvec_new (long n)
{
  float *ret = (float *) memalign (16, sizeof (*ret) * n);
  if (!ret) {
    fprintf (stderr, "fvec_new %ld : out of memory\n", n);
    abort();
  }
  return ret;
}

int *ivec_new (long n)
{
  int *ret = (int *) malloc (sizeof (*ret) * n);
  if (!ret) {
    fprintf (stderr, "ivec_new %ld : out of memory\n", n);
    abort();
  }
  return ret;
}


float *fvec_new_0 (long n)
{
  float *ret = (float *) calloc (sizeof (*ret), n);
  if (!ret) {
    fprintf (stderr, "fvec_new_0 %ld : out of memory\n", n);
    abort();
  }
  return ret;
}


/*-------------------------------------------------------------*/
/* Random                                                      */
/*-------------------------------------------------------------*/

/* Generate Gaussian random value, mean 0, variance 1 (from Python source) */


static double drand_r(unsigned int *seed) {
  return rand_r(seed)/((double)RAND_MAX + 1.0);
}

#define NV_MAGICCONST  1.71552776992141

static double gaussrand_r (unsigned int *seed)
{
  double z;
  while (1) {
    float u1, u2, zz;
    u1 = drand_r (seed);
    u2 = drand_r (seed);
    z = NV_MAGICCONST * (u1 - .5) / u2;
    zz = z * z / 4.0;
    if (zz < -log (u2))
      break;
  }
  return z;
}

void fvec_rand_r (float * v, long n, unsigned int seed)
{
  long i;
  for (i = 0 ; i < n ; i++)
    v[i] = drand_r(&seed);
}

void fvec_randn_r (float * v, long n, unsigned int seed)
{
  long i;
  for (i = 0 ; i < n ; i++)
    v[i] = gaussrand_r(&seed);
}


int * ivec_new_random_idx (int n, int k)
{
  return ivec_new_random_idx_r (n, k, lrand48());
}

int * ivec_new_random_idx_r (int n, int k, unsigned int seed)
{
  int *idx = ivec_new (n);
  int i;

  for (i = 0; i < n; i++)
    idx[i] = i;

  for (i = 0; i < k ; i++) {
    int j = i +  rand_r(&seed) % (n - i);
    /* swap i and j */
    int p = idx[i];
    idx[i] = idx[j];
    idx[j] = p;
  }

  return idx;
}

int *ivec_new_random_perm_r (int n, unsigned int seed)
{
  return ivec_new_random_idx_r (n, n - 1, seed);
}


float *fvec_new_set (long n, float val)
{
  int i;
  float *ret = (float *) calloc (sizeof (*ret), n);
  if (!ret) {
    fprintf (stderr, "fvec_new_set %ld : out of memory\n", n);
    abort();
  }

  for (i = 0 ; i < n ; i++)
    ret[i] = val;

  return ret;
}


/*-------------------------------------------------------------*/
/* Allocate & initialize                                       */
/*-------------------------------------------------------------*/

int *ivec_new_0 (long n)
{
  int *ret = (int *) calloc (sizeof (*ret), n);
  if (!ret) {
    fprintf (stderr, "ivec_new_0 %ld : out of memory\n", n);
    abort();
  }
  return ret;
}

int *ivec_new_set (long n, int val)
{
  int i;
  int *ret = ivec_new(n);

  for (i = 0 ; i < n ; i++)
    ret[i] = val;

  return ret;
}

float * fvec_new_cpy (const float * v, long n) {
  float *ret = fvec_new(n);
  memcpy (ret, v, n * sizeof (*ret));
  return ret;
}


/*-------------------------------------------------------------*/
/* resize                                                      */
/*-------------------------------------------------------------*/


float * fvec_resize (float * v, long n)
{
  float * v2 = realloc (v, n * sizeof (*v));
  return v2;
}


int * ivec_resize (int * v, long n)
{
  int * v2 = realloc (v, n * sizeof (*v));
  return v2;
}


/*-------------------------------------------------------------*/
/* statistics                                                  */
/*-------------------------------------------------------------*/


int *ivec_new_histogram (int k, const int *v, long n)
{
  long i;
  int *h = ivec_new_0 (k);

  for (i = 0; i < n; i++) {
    assert (v[i] >= 0 && v[i] < k);
    h[v[i]]++;
  }

  return h;
}

long ivec_count_occurrences(const int * v, long n, int val) {
  long count=0;
  while(n--) if(v[n]==val) count++;
  return count;
}

long fvec_count_occurrences(const float * v, long n, float val) {
  long count=0;
  while(n--) if(v[n]==val) count++;
  return count;
}


/*---------------------------------------------------------------------------*/
/* Input/Output functions                                                    */
/*                                                                           */
/* To avoid repeating too much code, many functions that are the same        */
/* for ivec, fvec, bvec, etc. are implemented with prefix xvec_ and          */
/* take as 1st argument unitsize, ther size of one element of the type       */
/* at hand                                                                   */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/* Input functions                                                           */
/*---------------------------------------------------------------------------*/

static long xvecs_fsize(long unitsize, const char * fname, int *d_out, int *n_out)
{
  int d, ret; 
  long nbytes;

  *d_out = -1;
  *n_out = -1;

  FILE * f = fopen (fname, "r");
  
  if(!f) {
    fprintf(stderr, "xvecs_fsize %s: %s\n", fname, strerror(errno));
    return -1;
  }
  /* read the dimension from the first vector */
  ret = fread (&d, sizeof (d), 1, f);
  if (ret == 0) { /* empty file */
    *n_out = 0;
    return ret;
  }
  
  fseek (f, 0, SEEK_END);
  nbytes = ftell (f);
  fclose (f);
  
  if(nbytes % (unitsize * d + 4) != 0) {
    fprintf(stderr, "xvecs_size %s: weird file size %ld for vectors of dimension %d\n", fname, nbytes, d);
    return -1;
  }

  *d_out = d;
  *n_out = nbytes / (unitsize * d + 4);
  return nbytes;
}

long bvecs_fsize (const char * fname, int *d_out, int *n_out)
{
  return xvecs_fsize (sizeof(unsigned char), fname, d_out, n_out);
}

int fvecs_read (const char *fname, int d, int n, float *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}

int b2fvecs_read (const char *fname, int d, int n, float *v)
{
  int n_new; 
  int d_new;
  bvecs_fsize (fname, &d_new, &n_new);	

  assert (d_new == d);
  assert (n <= n_new);

  FILE * f = fopen (fname, "r");
  assert (f || "b2fvecs_read: Unable to open the file");
  b2fvecs_fread (f, v, n);
  fclose (f);
  return n;
}

static int xvec_fread (long unit_size, FILE * f, void * v, int d_alloc)
{
  int d;
  int ret = fread (&d, sizeof (int), 1, f);

  if (feof (f))
    return 0;

  if (ret != 1) {
    perror ("# xvec_fread error 1");
    return -1;
  }

  if (d < 0 || d > d_alloc) {
    fprintf(stderr, "xvec_fread: weird vector size (expect %d found %d)\n", d_alloc, d);
    return -1;
  }

  ret = fread (v, unit_size, d, f);
  if (ret != d) {
    perror ("# xvec_fread error 2");
    return -1;
  }

  return d;
}

int fvec_fread (FILE * f, float * v, int d_alloc)
{
  return xvec_fread(sizeof(float), f, v, d_alloc);
}

int ivec_fread (FILE * f, int * v, int d_alloc)
{
  return xvec_fread(sizeof(int), f, v, d_alloc);
}

long b2fvecs_fread (FILE * f, float * v, long n)
{
  long i = 0, d = -1, ret;
  for (i = 0 ; i < n ; i++) {
    if (feof (f))
      break;

    ret = b2fvec_fread (f, v + i * d);
    if (ret == 0)  /* eof */
      break;

    if (ret == -1)
      return 0;

    if (i == 0)
      d = ret;

    if (d != ret) {
      perror ("# b2fvecs_fread: dimension of the vectors is not consistent\n");
      return 0;
    }
  }
  return i;
}

int b2fvec_fread (FILE * f, float * v)
{
  int d, j;
  int ret = fread (&d, sizeof (int), 1, f);
  if (feof (f))
    return 0;

  if (ret != 1) {
    perror ("# bvec_fread error 1");
    return -1;
  }

  unsigned char * vb = (unsigned char *) malloc (sizeof (*vb) * d);

  ret = fread (vb, sizeof (*vb), d, f);
  if (ret != d) {
    perror ("# bvec_fread error 2");
    return -1;
  }
  for (j = 0 ; j < d ; j++)
    v[j] = vb[j];
  free (vb);
  return d;
}

/*---------------------------------------------------------------------------*/
/* Output functions                                                          */
/*---------------------------------------------------------------------------*/

int fvec_fwrite (FILE *fo, const float *v, int d)
{
  int ret;
  ret = fwrite (&d, sizeof (int), 1, fo);
  if (ret != 1) {
    perror ("fvec_fwrite: write error 1");
    return -1;
  }
  ret = fwrite (v, sizeof (float), d, fo);
  if (ret != d) {
    perror ("fvec_fwrite: write error 2");
    return -1;
  }
  return 0;
}

void fvec_print (const float * v, int n)
{
  int i;
  printf ("[");
  for (i = 0 ; i < n ; i++)
    printf ("%g ", v[i]);
  printf ("]\n");
}

void ivec_print (const int * v, int n)
{
  int i;
  printf ("[");
  for (i = 0 ; i < n ; i++)
    printf ("%d ", v[i]);
  printf ("]\n");
}

int ivec_fwrite (FILE *f, const int *v, int d)
{
  int ret = fwrite (&d, sizeof (d), 1, f);
  if (ret != 1) {
    perror ("ivec_fwrite: write error 1");
    return -1;
  }

  ret = fwrite (v, sizeof (*v), d, f);
  if (ret != d) {
    perror ("ivec_fwrite: write error 2");
    return -2;
  }
  return 0;
}


/*---------------------------------------------------------------------------*/
/* Elementary operations                                                     */
/*---------------------------------------------------------------------------*/


void fvec_0(float * v, long n)
{
  memset (v, 0, n * sizeof (*v));
}

void ivec_0(int * v, long n)
{
  memset (v, 0, n * sizeof (*v));
}

void fvec_set (float * v, long n, float val)
{
  long i;
  for (i = 0 ; i < n ; i++)
    v[i] = val;
}

void ivec_cpy (int * vdest, const int * vsource, long n)
{
  memmove (vdest, vsource, n * sizeof (*vdest));
}

void fvec_cpy (float * vdest, const float * vsource, long n)
{
  memmove (vdest, vsource, n * sizeof (*vdest));
}

void fvec_sqr (float * v, long n)
{
  long i;
  for (i = 0 ; i < n ; i++)
    v[i] =  v[i] * v[i];
}

void fvec_mul_by (float * v, long n, double scal)
{
  long i = 0;
  for (i = 0 ; i < n ; i++)
    v[i] *= scal;
}

void fvec_add (float * v1, const float * v2, long n)
{
  long i = 0;
  for (i = 0 ; i < n ; i++)
    v1[i] += v2[i];
}

void fvec_sub (float * v1, const float * v2, long n)
{
  long i = 0;
  for (i = 0 ; i < n ; i++)
    v1[i] -= v2[i];
}

void fvec_div_by (float * v, long n, double scal)
{
  fvec_mul_by(v, n, 1. / scal);
}

/*---------------------------------------------------------------------------*/
/* Vector measures and statistics                                            */
/*---------------------------------------------------------------------------*/

double fvec_normalize (float * v, long n, double norm)
{
  if(norm==0)
    return 0;

  double nr = fvec_norm (v, n, norm);

  /*  if(nr!=0)*/
  fvec_mul_by (v, n, 1. / nr);
  return nr;
}

double fvec_sum (const float * v, long n)
{
  long i;
  double s = 0;
  for (i = 0 ; i < n ; i++)
    s += v[i];

  return s;
}

void ivec_cumsum(int *v, long n) {
  long i;
  int s = 0;
  for (i = 0 ; i < n ; i++) {
    s += v[i];
    v[i] = s;
  }
}


double fvec_sum_sqr (const float * v, long n)
{
  return fvec_norm2sqr(v, n);
}


double fvec_norm (const float * v, long n, double norm)
{
  if(norm==0) return n;

  long i;
  double s = 0;

  if(norm==1) {
    for (i = 0 ; i < n ; i++)
      s += fabs(v[i]);
    return s;
  }

  if(norm==2) {
    for (i = 0 ; i < n ; i++)
      s += v[i]*v[i];

    return sqrt(s);
  }

  if(norm==-1) {
    for (i = 0 ; i < n ; i++)
      if(fabs(v[i])>s) s=fabs(v[i]);
    return s;
  }

  for (i = 0 ; i < n ; i++) {
    s += pow (v[i], norm);
  }

  return pow (s, 1 / norm);
}


double fvec_norm2sqr (const float * v, long n) {
  double s=0;
  long i;
  for (i = 0 ; i < n ; i++)
    s += v[i] * v[i];
  return s;
}

double ivec_unbalanced_factor(const int *hist, long n) {
  int vw;
  double tot = 0, uf = 0;

  for (vw = 0 ; vw < n ; vw++) {
    tot += hist[vw];
    uf += hist[vw] * (double) hist[vw];
  }

  uf = uf * n / (tot * tot);

  return uf;

}

/*---------------------------------------------------------------------------*/
/* Distances                                                                 */
/*---------------------------------------------------------------------------*/

double fvec_distance_L2sqr (const float * v1, const float * v2, long n)
{
  long i;
  double dis = 0, a;

  for (i = 0 ; i < n ; i++) {
    a = (double) v1[i] - v2[i];
    dis += a * a;
  }

  return dis;
}

/*---------------------------------------------------------------------------*/
/* Sparse vector handling                                                    */
/*---------------------------------------------------------------------------*/


long ivec_index(const int * v, long n,int val) {
  long i;
  for(i=0;i<n;i++) if(v[i]==val) return i;
  return -1;
}

void fvec_cpy_subvectors (const float * v, int * idx, int d, int nout, float * vout)
{
  long i;
  for (i = 0 ; i < nout ; i++)
    fvec_cpy (vout + i * d, v + (long) idx[i] * d, d);
}

/* copy a subset of byte vectors and cast them to float vectors in the same time */
void b2fvec_cpy_subvectors (const unsigned char * v, int * idx, int d, int nout, float * vout)
{
  long i, j;
  for (i = 0 ; i < nout ; i++)
    for (j = 0 ; j < d ; j++)
      vout[i* (long)d+j] = v[idx[i] * (long) d+j];
}
