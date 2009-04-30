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
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "SpacetimeTensors.h"
#include "VanVleckEquations.h"

#define NUM_EQS	(5+16+1)

/* RHS of our system of ODEs */
int func (double tau, const double y[], double f[], void *params)
{  
  /* Geodesic equations: 5 coupled equations for r,r',theta,phi,t */
  gsl_vector_view geodesic_eqs = gsl_vector_view_array(f,5);
  gsl_vector_const_view geodesic_coords = gsl_vector_const_view_array(y,5);
  geodesicRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, params);
  
  /* Equations for Q^a_b */
  gsl_matrix_view q_eqs = gsl_matrix_view_array(f+5,4,4);
  gsl_matrix_const_view q_vals = gsl_matrix_const_view_array(y+5,4,4);
  qRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, &q_eqs.matrix, params);
  
  /* Equation for Delta^1/2 */
  sqrtDeltaRHS(tau, &q_vals.matrix, &y[21], &f[21], params);
  
  return GSL_SUCCESS;
}

int main (int argc, char * argv[])
{
  int i;
  
  /* Use a Runge-Kutta integrator with adaptive step-size */
  const gsl_odeiv_step_type * T = gsl_odeiv_step_rkf45;
  gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, NUM_EQS);
  gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-10, 1e-10);
  gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (NUM_EQS);

  if (argc != 6)
  {
    printf("5 parameters required, %d given\n", argc);
    return 0;
  }

  /* Time-like geodesic starting at r=10M and going in to r=4M */
  //struct geodesic_params params = {1,0.950382,3.59211,-1};
  struct geodesic_params params = {atof(argv[1]), atof(argv[2]), atof(argv[3]), atoi(argv[4])};

  gsl_odeiv_system sys = {func, NULL, NUM_EQS, &params};

  double tau = 0.0, tau1 = 1000.0;
  double h = 1e-6;
  double r0 = atof(argv[5]);
  
  /* Initial Condidions */
  double y[NUM_EQS] = { 
    r0, 0.0, 0.0, 0.0, 0.0, /* r, r', theta, phi, t */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* Q^a'_b' */
    1.0 /* Delta^1/2 */
    }; 

  while (tau < tau1)
  {
    int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &tau, tau1, &h, y);

    if (status != GSL_SUCCESS)
      break;
    
    /* Output the results */
    printf ("%.5f", tau);
    for(i=0; i<NUM_EQS; i++)
    {
      printf (", %.5f", y[i]);
    }
    printf("\n");
    
    /* Don't let the step size get bigger than 1 */
    /*if (h > .10)
    {
      fprintf(stderr,"Warning: step size %e greater than 1 is not allowed. Using step size of 1.0.\n",h);
      h=.10;
    }*/
      
    /* Exit if step size get smaller than 10^-12 */
    if (h < 1e-13 || tau > 73.0)
    {
      fprintf(stderr,"Error: step size %e less than 1e-13 is not allowed.\n",h);
      break;
    }
  }

  gsl_odeiv_evolve_free (e);
  gsl_odeiv_control_free (c);
  gsl_odeiv_step_free (s);
  
  return 0;
}
