/* Numerically solve the transport equation for the Van Vleck determinant
 * along a geodesic.
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
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "SpacetimeTensors.h"

/* Right hand side of ODE's for q^a_b = \sigm^a_b - \delta^a_b */
int qRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, gsl_matrix * f, void * params)
{
  /* Gu */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  /* Q * Gu */
  gsl_matrix * q_gu = gsl_matrix_calloc(4,4);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, q, gu, 0, q_gu);
  
  /* Gu * Q */
  gsl_matrix * gu_q = gsl_matrix_calloc(4,4);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gu, q, 0, gu_q);
  
  /* Q^2 */
  gsl_matrix * q2 = gsl_matrix_calloc(4,4);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, q, q, 0, q2);
  
  if(tau!=0.0)
  {
      /* (Q^2 + Q)/tau */
      gsl_matrix_add(q2, q);
      gsl_matrix_scale(q2, 1.0/tau);
  }
  
  /* tau * S */
  gsl_matrix * tau_S = gsl_matrix_calloc(4,4);
  S(y, yp, tau_S, params);
  gsl_matrix_scale(tau_S, tau);
  
  /* Add them all together to get the RHS */
  gsl_matrix_set_zero(f);
  gsl_matrix_add(f,q_gu);
  gsl_matrix_sub(f,gu_q);
  if(tau!=0.0)
      gsl_matrix_sub(f, q2);
  gsl_matrix_sub(f,tau_S);
  
  /* The theta,theta component blows up as theta*cot(theta) and makes the numerical scheme break down.
     Since we know the analytic form, don't compute it numerically */
  //gsl_matrix_set(f,1,1,0.0);
  
  gsl_matrix_free(gu);
  gsl_matrix_free(q_gu);
  gsl_matrix_free(gu_q);
  gsl_matrix_free(q2);
  gsl_matrix_free(tau_S);

  return GSL_SUCCESS;
}

int sqrtDeltaRHS (double tau, const gsl_matrix * q, const double * sqrt_delta, double * f, void * params)
{
  int i;
  double rhs = 0.0;
  
  if(tau!=0.0)
  {
      for(i=0; i<4; i++)
      {
          rhs -= gsl_matrix_get(q, i, i);
      }

      rhs = rhs * (*sqrt_delta) / 2.0 / tau;
  }

  *f = rhs;
  
  return GSL_SUCCESS;
}
