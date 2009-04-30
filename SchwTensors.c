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

/* The contravariant metric components for Schwarzschild (with theta=Pi/2) */
int metric_up_up(const double *y, gsl_matrix *metric, void *params)
{
  gsl_matrix_set_zero(metric);
  
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double r = y[0];
  
  gsl_matrix_set(metric,0,0,(r-2*m)/r);
  gsl_matrix_set(metric,1,1,1/gsl_pow_2(r));
  gsl_matrix_set(metric,2,2,1/gsl_pow_2(r));
  gsl_matrix_set(metric,3,3,-r/(r-2*m));
  
  return GSL_SUCCESS;
}

/* Calculates the matrix S^a_b = R^a_{ c b d} u^c u^d and fill the values into s.  Note that we have already
   set theat=Pi/2 */
int S (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *s, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double ur = gsl_vector_get(yp,0);
  double uth = gsl_vector_get(yp,2);
  double up = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  gsl_matrix_set(s,0,0,(m*(4*m*gsl_pow_int(ut,2) - 2*r*gsl_pow_int(ut,2) - gsl_pow_int(r,3)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/gsl_pow_int(r,4));
  gsl_matrix_set(s,0,1,(m*ur*uth)/r);
  gsl_matrix_set(s,0,2,(m*up*ur)/r);
  gsl_matrix_set(s,0,3,(2*m*(-2*m + r)*ur*ut)/gsl_pow_int(r,4));
  gsl_matrix_set(s,1,0,-((m*ur*uth)/((2*m - r)*gsl_pow_int(r,2))));
  gsl_matrix_set(s,1,1,(m*gsl_pow_int(r,2)*(2*(2*m - r)*r*gsl_pow_int(up,2) + gsl_pow_int(ur,2)) - m*gsl_pow_int(-2*m + r,2)*gsl_pow_int(ut,2))/((2*m - r)*gsl_pow_int(r,4)));
  gsl_matrix_set(s,1,2,(-2*m*up*uth)/r);
  gsl_matrix_set(s,1,3,(m*(2*m - r)*ut*uth)/gsl_pow_int(r,4));
  gsl_matrix_set(s,2,0,-((m*up*ur)/((2*m - r)*gsl_pow_int(r,2))));
  gsl_matrix_set(s,2,1,(-2*m*up*uth)/r);
  gsl_matrix_set(s,2,2,(m*(-4*gsl_pow_int(m,2)*gsl_pow_int(ut,2) + r*(4*m*gsl_pow_int(ut,2) + r*(gsl_pow_int(ur,2) - gsl_pow_int(ut,2) - 2*r*(-2*m + r)*gsl_pow_int(uth,2)))))/((2*m - r)*gsl_pow_int(r,4)));
  gsl_matrix_set(s,2,3,(m*(2*m - r)*up*ut)/gsl_pow_int(r,4));
  gsl_matrix_set(s,3,0,(2*m*ur*ut)/((2*m - r)*gsl_pow_int(r,2)));
  gsl_matrix_set(s,3,1,(m*ut*uth)/r);
  gsl_matrix_set(s,3,2,(m*up*ut)/r);
  gsl_matrix_set(s,3,3,(m*(-2*gsl_pow_int(ur,2) + r*(-2*m + r)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/ ((2*m - r)*gsl_pow_int(r,2)));
  
  return GSL_SUCCESS;
}

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ b d c} u^d and fill the values into r_sigma, which is an array of matrices.
   We use the convention that c is the index of the array and a and b are the indices of the matrices. Note that we have already
   set theat=Pi/2 and uth=0. */
int R_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_vector * r_sigma, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double ur = gsl_vector_get(yp,0);
  double uph = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  /* Initialize all elements to 0 */
  gsl_vector_set_zero(r_sigma);
  
  /* Now, set the non-zero elements */
  gsl_vector_set(r_sigma, 16*1 + 4*0 + 1, -m*ur/r);
  gsl_vector_set(r_sigma, 16*0 + 4*0 + 2, m*uph/r);
  gsl_vector_set(r_sigma, 16*2 + 4*0 + 2, -m*ur/r);
  gsl_vector_set(r_sigma, 16*0 + 4*0 + 3, -(2*(-r+2*m))*m*ut/gsl_pow_4(r));
  gsl_vector_set(r_sigma, 16*3 + 4*0 + 3, (2*(-r+2*m))*m*ur/gsl_pow_4(r));
  gsl_vector_set(r_sigma, 16*1 + 4*1 + 0, -m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_vector_set(r_sigma, 16*1 + 4*1 + 2, -2*m*uph/r);
  gsl_vector_set(r_sigma, 16*1 + 4*1 + 3, (-r+2*m)*m*ut/gsl_pow_4(r));
  gsl_vector_set(r_sigma, 16*0 + 4*2 + 0, m*uph/(gsl_pow_2(r)*(-r+2*m)));
  gsl_vector_set(r_sigma, 16*2 + 4*2 + 0, -m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_vector_set(r_sigma, 16*1 + 4*2 + 1, 2*m*uph/r);
  gsl_vector_set(r_sigma, 16*2 + 4*2 + 3, (-r+2*m)*m*ut/gsl_pow_4(r));
  gsl_vector_set(r_sigma, 16*3 + 4*2 + 3, -(-r+2*m)*m*uph/gsl_pow_4(r));
  gsl_vector_set(r_sigma, 16*0 + 4*3 + 0, -2*m*ut/(gsl_pow_2(r)*(-r+2*m)));
  gsl_vector_set(r_sigma, 16*3 + 4*3 + 0, 2*m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_vector_set(r_sigma, 16*1 + 4*3 + 1, -m*ut/r);
  gsl_vector_set(r_sigma, 16*2 + 4*3 + 2, -m*ut/r);
  gsl_vector_set(r_sigma, 16*3 + 4*3 + 2, m*uph/r);
  
  return GSL_SUCCESS;
}

/* Calculates the matrix Gu^a_b = \Gamma^a_{b c} u^c and fill the values into gu */
int Gu (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *gu, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double ur = gsl_vector_get(yp,0);
  double up = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  gsl_matrix_set(gu,0,0,m*ur/r/(2*m-r));
  gsl_matrix_set(gu,0,1,0);
  gsl_matrix_set(gu,0,2,(2*m-r)*up);
  gsl_matrix_set(gu,0,3,m*ut*(r-2*m)/gsl_pow_3(r));
  gsl_matrix_set(gu,1,0,0);
  gsl_matrix_set(gu,1,1,ur/r);
  gsl_matrix_set(gu,1,2,0);
  gsl_matrix_set(gu,1,3,0);
  gsl_matrix_set(gu,2,0,up/r);
  gsl_matrix_set(gu,2,1,0);
  gsl_matrix_set(gu,2,2,ur/r);
  gsl_matrix_set(gu,2,3,0);
  gsl_matrix_set(gu,3,0,m*ut/r/(r-2*m));
  gsl_matrix_set(gu,3,1,0);
  gsl_matrix_set(gu,3,2,0);
  gsl_matrix_set(gu,3,3,m*ur/r/(r-2*m));
  
  return GSL_SUCCESS;
}

/* RHS of geodesic equations */
int geodesicRHS (double tau, const gsl_vector * y, gsl_vector * f, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  
  double r = gsl_vector_get(y,0);
  double rp = gsl_vector_get(y,1);
  
  gsl_vector_set(f,0,rp);
  gsl_vector_set(f,1,(gsl_pow_2(p.l)*(r-3*p.m)+p.m*gsl_pow_2(r)*p.type)/gsl_pow_4(r));
  gsl_vector_set(f,2,0.0);
  gsl_vector_set(f,3,p.l/gsl_pow_2(r));
  gsl_vector_set(f,4,r/(r-2*p.m)*p.e);
  
  return GSL_SUCCESS;
}
