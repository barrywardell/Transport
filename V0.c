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
#include "VanVleckEquations.h"
#include "V0Equations.h"

#define NUM_EQS (5+16+1+16+16+64+64+64+256+256+256+1)

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
  sqrtDeltaRHS(tau, &q_vals.matrix, &y[5+16], &f[5+16], params);
  
  /* Equation for I */
  gsl_matrix_view I_eqs = gsl_matrix_view_array(f+5+16+1,4,4);
  gsl_matrix_const_view I_vals = gsl_matrix_const_view_array(y+5+16+1,4,4);
  IRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &I_vals.matrix, &I_eqs.matrix, params);
  
  /* Equation for eta */
  gsl_matrix_view eta_eqs = gsl_matrix_view_array(f+5+16+1+16,4,4);
  gsl_matrix_const_view eta_vals = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
  etaRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, &eta_vals.matrix, &eta_eqs.matrix, params);
  
  /* Equation for dI */
  gsl_vector_view dI_eqs_view = gsl_vector_view_array(f+5+16+1+16+16,4*4*4);
  gsl_vector_const_view dI_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16,4*4*4);
  gsl_vector * dI_eqs = &dI_eqs_view.vector;
  const gsl_vector * dI_vals = &dI_vals_view.vector;

  dIinvRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &I_vals.matrix, &q_vals.matrix, dI_vals, dI_eqs, params);
  
  /* Equation for dxi */
  gsl_vector_view dxi_eqs_view = gsl_vector_view_array(f+5+16+1+16+16+64,4*4*4);
  gsl_vector_const_view dxi_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16+64,4*4*4);
  gsl_vector * dxi_eqs = &dxi_eqs_view.vector;
  const gsl_vector * dxi_vals = &dxi_vals_view.vector;

  dxiRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, dxi_eqs, params);

  /* Equation for deta */
  gsl_vector_view deta_eqs_view = gsl_vector_view_array(f+5+16+1+16+16+64+64,4*4*4);
  gsl_vector_const_view deta_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16+64+64,4*4*4);
  gsl_vector * deta_eqs = &deta_eqs_view.vector;
  const gsl_vector * deta_vals = &deta_vals_view.vector;

  detaRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &eta_vals.matrix, deta_vals, deta_eqs, params);

  /* Equation for d2Iinv */
  gsl_vector_view d2Iinv_eqs_view = gsl_vector_view_array(f+5+16+1+16+16+64+64+64,4*4*4*4);
  gsl_vector_const_view d2Iinv_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16+64+64+64,4*4*4*4);
  gsl_vector * d2Iinv_eqs = &d2Iinv_eqs_view.vector;
  const gsl_vector * d2Iinv_vals = &d2Iinv_vals_view.vector;

  d2IinvRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &I_vals.matrix, dI_vals, d2Iinv_vals, d2Iinv_eqs, params);

  /* Equation for d2xi */
  gsl_vector_view d2xi_eqs_view = gsl_vector_view_array(f+5+16+1+16+16+64+64+64+256,4*4*4*4);
  gsl_vector_const_view d2xi_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16+64+64+64+256,4*4*4*4);
  gsl_vector * d2xi_eqs = &d2xi_eqs_view.vector;
  const gsl_vector * d2xi_vals = &d2xi_vals_view.vector;

  d2xiRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &I_vals.matrix, dI_vals, d2xi_vals, d2xi_eqs, params);

  /* Equation for d2eta */
  gsl_vector_view d2eta_eqs_view = gsl_vector_view_array(f+5+16+1+16+16+64+64+64+256+256,4*4*4*4);
  gsl_vector_const_view d2eta_vals_view = gsl_vector_const_view_array(y+5+16+1+16+16+64+64+64+256+256,4*4*4*4);
  gsl_vector * d2eta_eqs = &d2eta_eqs_view.vector;
  const gsl_vector * d2eta_vals = &d2eta_vals_view.vector;

  d2etaRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &eta_vals.matrix, &q_vals.matrix, dxi_vals, d2xi_vals, deta_vals, d2eta_vals, d2eta_eqs, params);

  /* Calculate Box SqrtDelta */
  double box_sqrt_delta = 0;
  boxSqrtDelta (tau, y, &box_sqrt_delta, &params);

  /* Calculate V0 */
  V0RHS (tau, &q_vals.matrix, &box_sqrt_delta, y+5+16+1+16+16+64+64+64+256+256+256, f+5+16+1+16+16+64+64+64+256+256+256, params);
  
  return GSL_SUCCESS;
}

int main (void)
{
  int i;
  
  /* Use a Runge-Kutta integrator with adaptive step-size */
  const gsl_odeiv_step_type * T = gsl_odeiv_step_rkf45;
  gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, NUM_EQS);
  gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-8, 1e-8);
  gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (NUM_EQS);

  /* Time-like geodesic starting at r=10M and going in to r=4M */
  struct geodesic_params params = {1,0.950382,3.59211,-1};
  
  gsl_odeiv_system sys = {func, NULL, NUM_EQS, &params};

  double tau = 0.0, tau1 = 1000.0;
  double h = 1e-2;
  double r0 = 10;
  double m = 1.0;
  
  /* Initial Conditions */
  double y[NUM_EQS] = { 
    /* r, r', theta, phi, t */
    r0, 0.0, 0.0, 0.0, 0.0,

    /* Q^a'_b' */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* Delta^1/2 */
    1.0,

    /* I_a'^b */
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,

    /* eta^a_b' */
    -1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0,

   /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* dxi */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* deta */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* d2I - this will get filled in later */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* d2xi - this will get filled in later*/
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    
    /* d2eta - this will get filled in later */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

    /* V_0 */
    0
  };

  d2IinvInit(y+5+16+1+16+16+64+64+64, r0, &params);
  d2xiInit(y+5+16+1+16+16+64+64+64+256, r0, &params);
  d2etaInit(y+5+16+1+16+16+64+64+64+256+256, r0, &params);

  /* Solve system of ODEs */
  while (tau < tau1)
  {
    int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &tau, tau1, &h, y);

    if (status != GSL_SUCCESS)
      break;
    
    /* Gamma is the matrix inverse of eta */
    int signum;
    gsl_permutation * p = gsl_permutation_alloc (4);
    gsl_matrix_const_view eta = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
    gsl_matrix * gamma = gsl_matrix_calloc(4,4);
    gsl_matrix * lu = gsl_matrix_calloc(4,4);
    gsl_matrix_memcpy(lu, &eta.matrix);
    gsl_linalg_LU_decomp (lu, p, &signum);
    gsl_linalg_LU_invert (lu, p, gamma);
    
    /* Calculate Box SqrtDelta */
    double box_sqrt_delta = 0;
    boxSqrtDelta (tau, y, &box_sqrt_delta, &params);
    
    /* Output the results */
    printf ("%.5f", tau);
    for(i=0; i<NUM_EQS; i++)
    {
      printf (", %.5f", y[i]);
    }
    printf (", %.5f, %.5f, %.5f, %.5f", gsl_matrix_get(gamma,0,0), gsl_matrix_get(gamma,0,1), gsl_matrix_get(gamma,0,2), gsl_matrix_get(gamma,0,3)); 
    printf (", %.5f, %.5f, %.5f, %.5f", gsl_matrix_get(gamma,1,0), gsl_matrix_get(gamma,1,1), gsl_matrix_get(gamma,1,2), gsl_matrix_get(gamma,1,3));
    printf (", %.5f, %.5f, %.5f, %.5f", gsl_matrix_get(gamma,2,0), gsl_matrix_get(gamma,2,1), gsl_matrix_get(gamma,2,2), gsl_matrix_get(gamma,2,3));
    printf (", %.5f, %.5f, %.5f, %.5f", gsl_matrix_get(gamma,3,0), gsl_matrix_get(gamma,3,1), gsl_matrix_get(gamma,3,2), gsl_matrix_get(gamma,3,3));
    printf(", %.5f", box_sqrt_delta);
    //printf(", %.5f", y[NUM_EQS-1]);
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
      fprintf(stderr,"Error: step size %e less than 1e-8 is not allowed.\n",h);
      break;
    }
  }

  //gsl_odeiv_evolve_free (e);
  //gsl_odeiv_control_free (c);
  //gsl_odeiv_step_free (s);
  return 0;
}
