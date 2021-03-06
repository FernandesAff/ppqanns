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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "binheap.h"
#include "sorting.h"


size_t fbinheap_sizeof (int maxk) 
{
  return sizeof (fbinheap_t) + maxk * (sizeof (float) + sizeof (int));
}


void fbinheap_init (fbinheap_t *bh, int maxk) 
{
  bh->k = 0;
  bh->maxk = maxk;

  /* weirdly index the nodes from 1 (for the root) */
  bh->val = (float *) ((char *) bh + sizeof (fbinheap_t)) - 1;
  bh->label = (int *) ((char *) bh + sizeof (fbinheap_t)
		       + maxk * sizeof (*bh->val)) - 1;
}


fbinheap_t * fbinheap_new (int maxk)
{
  int bhsize = fbinheap_sizeof (maxk);
  fbinheap_t * bh = (fbinheap_t *) malloc (bhsize);
  fbinheap_init (bh,maxk);
  return bh;
}


void fbinheap_delete (fbinheap_t * bh)
{
  free (bh);
}


void fbinheap_pop (fbinheap_t * bh)
{
  assert (bh->k > 0);

  float val = bh->val[bh->k];
  int i = 1, i1, i2;

  while (1) {
    i1 = i << 1;
    i2 = i1 + 1;

    if (i1 > bh->k)
      break;

    if (i2 == bh->k + 1 || bh->val[i1] > bh->val[i2]) {
      if (val > bh->val[i1])
        break;
      bh->val[i] = bh->val[i1];
      bh->label[i] = bh->label[i1];
      i = i1;
    } 
    else {
      if (val > bh->val[i2])
        break;
      
      bh->val[i] = bh->val[i2];
      bh->label[i] = bh->label[i2];
      i = i2;
    }
  }
  
  bh->val[i] = bh->val[bh->k];
  bh->label[i] = bh->label[bh->k];
  bh->k--;
}


static void fbinheap_push (fbinheap_t * bh, int label, float val)
{
  assert (bh->k < bh->maxk);

  int i = ++bh->k, i_father;

  while (i > 1) {
    i_father = i >> 1;
    if (bh->val[i_father] >= val)  /* the heap structure is ok */
      break; 

    bh->val[i] = bh->val[i_father];
    bh->label[i] = bh->label[i_father];

    i = i_father;
  }
  bh->val[i] = val;
  bh->label[i] = label;
}


void fbinheap_add (fbinheap_t * bh, int label, float val)
{
  if (bh->k < bh->maxk) {
    fbinheap_push (bh, label, val);
    return;
  }

  if (val < bh->val[1]) {  
    fbinheap_pop (bh);
    fbinheap_push (bh, label, val);
  }
}

void fbinheap_addn_label_range (fbinheap_t * bh, int n, int label0, const float * v)
{
  int i;

  for (i = 0 ; i < n && bh->k < bh->maxk; i++)
    if(!isnan(v[i]))
      fbinheap_push (bh, label0+i, v[i]);

  float lim=bh->val[1];
  for ( ; i < n; i++) { /* optimized loop for common case */
    if(v[i]<lim) {
      fbinheap_pop (bh);
      fbinheap_push (bh, label0+i, v[i]);
      lim=bh->val[1];
    }      
  }
}

void fbinheap_sort_labels (fbinheap_t * bh, int * perm)
{
  int i;
  fvec_sort_index (bh->val + 1, bh->k, perm);
  for (i = 0 ; i < bh->k ; i++)
    perm[i] = bh->label[perm[i]+1];
}

void fbinheap_sort (fbinheap_t * bh, int * perm, float *v)
{
  int i, heappos;
  /* TODO use binheap structure to get in the correct order */
  fvec_sort_index (bh->val + 1, bh->k, perm);
  for (i = 0 ; i < bh->k ; i++) {
    heappos = perm[i] + 1;
    perm[i] = bh->label[heappos];
    v[i] = bh->val[heappos];
  }
}
