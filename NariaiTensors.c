/* Components of the tensors used to numerically solve the transport equation 
 * for the Van Vleck determinant along a geodesic evaluated for the Nariai
 * spacetime.
 *
 * Copyright (C) 2009 Barry Wardell
 *  
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 */
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include "SpacetimeTensors.h"

/* The contravariant metric components for Nariai (with theta=Pi/2) */
int metric_up_up(const double *y, gsl_matrix *metric, void *params)
{
  (void)params;
  double r = y[0];

  gsl_matrix_set_zero(metric);
  
  gsl_matrix_set(metric,0,0,1-gsl_pow_2(r));
  gsl_matrix_set(metric,1,1,1);
  gsl_matrix_set(metric,2,2,1);
  gsl_matrix_set(metric,3,3,-1/(1-gsl_pow_2(r)));
  
  return GSL_SUCCESS;
}

/* Calculates the matrix S^a_b = R^a_{ c b d} u^c u^d and fill the values into s.
 * Note that we have already set theat=Pi/2 
 */
int S (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *s, void *params)
{
  (void)params;
  double ur = gsl_vector_get(yp,0);
  double uph = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  /* Initialize all elements to 0 */
  gsl_matrix_set_zero(s);

  /* Set non-zero elements */
  gsl_matrix_set(s,0,0,(-1 + r * r) * ut * ut);
  gsl_matrix_set(s,0,3,-(-1 + r * r) * ut * ur);
  gsl_matrix_set(s,1,1, uph * uph);
  gsl_matrix_set(s,3,0,1 / (-1 + r * r) * ur * ut);
  gsl_matrix_set(s,3,3,-1 / (-1 + r * r) * ur * ur);

  return GSL_SUCCESS;
}

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ b d c} u^d and fill the values into r_sigma, which is an array of matrices.
   We use the convention that c is the index of the array and a and b are the indices of the matrices. Note that we have already
   set theat=Pi/2 and uth=0. 
   FIXME: create this function */
int R_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *r_sigma[], void *params)
{
  (void)params;
  double ur = gsl_vector_get(yp,0);
  double uph = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  int i;
  
  /* Initialize all elements to 0 */
  for(i=0; i<4; i++)
  {
    gsl_matrix_set_zero(r_sigma[i]);
  }
  
  /* Now, set the non-zero elements */
/*  r_sigma[0][3][0] = -(-1 + r * r) * ut;
  r_sigma[0][3][3] = (-1 + r * r) * ur;
  r_sigma[1][2][1] = -uph;
  r_sigma[2][1][1] = uph;
  r_sigma[3][0][0] = -1 / (-1 + r * r) * ut;
  r_sigma[3][0][3] = 1 / (-1 + r * r) * ur;
  */
  
  return GSL_SUCCESS;
}

/* Calculates the matrix Gu^a_b = \Gamma^a_{b c} u^c and fill the values into gu */
int Gu (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *gu, void *params)
{
  (void)params;
  double ur = gsl_vector_get(yp,0);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  gsl_matrix_set_zero(gu);
  
  gsl_matrix_set(gu,0,0,-r / (-1 + r * r) * ur);
  gsl_matrix_set(gu,0,3,(-1 + r * r) * r * ut);
  gsl_matrix_set(gu,3,0,r / (-1 + r * r) * ut);
  gsl_matrix_set(gu,3,3,r / (-1 + r * r) * ur);

  return GSL_SUCCESS;
}

/* RHS of geodesic equations */
int geodesicRHS (double tau, const gsl_vector * y, gsl_vector * f, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  
  double r = gsl_vector_get(y,0);
  double rp = gsl_vector_get(y,1);
  
  gsl_vector_set(f,0,rp);
  gsl_vector_set(f,1,(gsl_pow_2(p.l)-p.type)*r);
  gsl_vector_set(f,2,0.0);
  gsl_vector_set(f,3,p.l);
  gsl_vector_set(f,4,p.e/(1-gsl_pow_2(r)));
  
  return GSL_SUCCESS;
}
