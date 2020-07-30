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

/*---------------------------------------------------------------------------*/

#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

/*---------------------------------------------------------------------------*/
/*! @addtogroup knearestneighbors
 *  @{
 */


/*! @defgroup knearestneighbors
 * Nearest-neighbor (NN) functions
 *
 * All matrices are stored in column-major order (like Fortran) and
 * indexed from 0 (like C, unlike Fortran). The declaration:
 *
 *     a(m, n)
 *
 * means that element a(i, j) is accessed with a[ i * m + j ] where
 *
 *     0 <= i < m and 0 <= j < n
 *
 */


/*!  Finds nearest neighbors of vectors in a base
 *
 * @param    distance_type  2 = L2 distance (see compute_cross_distances_alt for distance_type's)
 * @param    nq             number of query vectors
 * @param    nb             number of base vectors to assign to
 * @param    k              number of neighbors to return
 * @param    q(d, n)        query vectors
 * @param    b(d, nb)       base vectors
 * @param    assign(k, n)   on output, the NNs of vector i are assign(:, i) (sorted by increasing distances)
 * @param    b_weights(nb)  multiply squared distances by this for each base vector (may be NULL)
 * @param    dis(k, n)      squared distances of i to its NNs are dis(0, i) to dis(k-1, i)
 *
 */

void knn_full (int distance_type,
               int nq, int nb, int d, int k,
               const float *b, const float *q,
               const float *b_weights,
               int *assign, float *dis);

/*! multi-threaded version
 */
void knn_full_thread (int distance_type,
                      int nq, int nb, int d, int k,
                      const float *b, const float *q,
                      const float *b_weights,
                      int *assign, float *dis,
                      int n_thread);

/* next functions are simplified calls of the previous */


/*! single NN, returns sum of squared distances */
double nn (int n, int nb, int d,
         const float *b, const float *v,
         int *assign);

void compute_cross_distances_task (void *arg, int tid, int i);

/*! compute_cross_distances with threads */
void compute_cross_distances_thread (int d, int na, int nb,
                                    const float *a,
                                    const float *b, float *dist2,
                                    int nt);


/* next functions are simplified calls of the previous */

void compute_cross_distances (int d, int na, int nb,
                              const float *a,
                              const float *b, float *dist2);

/*! compute_cross_distances for non-packed matrices
 *
 * @param lda            size in memory of one vector of a
 * @param ldb            size in memory of one vector of b
 * @param ldd            size in memory of one vector of dist2
 */
void compute_cross_distances_nonpacked (int d, int na, int nb,
                                        const float *a, int lda,
                                        const float *b, int ldb,
                                        float *dist2, int ldd);

/*! Like compute_cross_distances with alternative distances.
 *
 * @param distance_type    type of distance to compute:
 *    - 1: L1
 *    - 2: L2 (use 12 for optimized version)
 *    - 3: symmetric chi^2
 *    - 4: symmetric chi^2  with absolute value
 *    - 5: histogram intersection (sum of min of vals)
 *    - 6: dot prod (use 16 for optimized version)
 */
void compute_cross_distances_alt (int distance_type, int d, int na, int nb,
                                  const float *a,
                                  const float *b, float *dist2);


/*! compute_cross_distances_alt with non-packed input and output */
void compute_cross_distances_alt_nonpacked (int distance_type, int d, int na, int nb,
                                            const float *a, int lda,
                                            const float *b, int ldb,
                                            float *dist2, int ldd);

/*! version of compute_cross_distances where na==1 */
void compute_distances_1 (int d, int nb,
                          const float *a,
                          const float *b,
                          float *dist2);

void compute_distances_1_nonpacked (int d, int nb,
                                    const float *a,
                                    const float *b, int ldb,
                                    float *dist2);

void compute_distances_1_thread (int d, int nb,
                                 const float *a,
                                 const float *b,
                                 float *dist2,
                                 int n_thread);

void compute_distances_1_nonpacked_thread (int d, int nb,
                                           const float *a,
                                           const float *b, int ldb,
                                           float *dist2,
                                           int n_thread);

/*! @} */

#endif

/*---------------------------------------------------------------------------*/
