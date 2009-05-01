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

#define NUM_EQS (5+16+1+16+16+64+64+64+256+256+1)

/* RHS of our system of ODEs */
int func (double tau, const double y[], double f[], void *params)
{
  int i;
  
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
  
  /* Equation for I */
  gsl_matrix_view I_eqs = gsl_matrix_view_array(f+5+16+1,4,4);
  gsl_matrix_const_view I_vals = gsl_matrix_const_view_array(y+5+16+1,4,4);
  IRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &I_vals.matrix, &I_eqs.matrix, params);
  
  /* Equation for eta */
  gsl_matrix_view eta_eqs = gsl_matrix_view_array(f+5+16+1+16,4,4);
  gsl_matrix_const_view eta_vals = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
  etaRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, &eta_vals.matrix, &eta_eqs.matrix, params);
  
  /* Equation for dI */
  gsl_matrix_view dI_eqs_views[4] = {gsl_matrix_view_array(f+5+16+1+16+16+0,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+16,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+32,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+48,4,4)};
  gsl_matrix_const_view dI_vals_views[4] = {gsl_matrix_const_view_array(y+5+16+1+16+16+0,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+16,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+32,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+48,4,4)};
  gsl_matrix * dI_eqs[4];
  const gsl_matrix * dI_vals[4];
  
  for (i=0; i<4; i++)
  {
    dI_eqs[i] = &dI_eqs_views[i].matrix;
    dI_vals[i] = &dI_vals_views[i].matrix;
  }
  
  dIinvRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &I_vals.matrix, &q_vals.matrix, dI_vals, dI_eqs, params);
  
  /* Equation for dxi */
  gsl_matrix_view dxi_eqs_views[4] = {gsl_matrix_view_array(f+5+16+1+16+16+64+0,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+16,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+32,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+48,4,4)};
  gsl_matrix_const_view dxi_vals_views[4] = {gsl_matrix_const_view_array(y+5+16+1+16+16+64+0,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+16,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+32,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+48,4,4)};
  gsl_matrix * dxi_eqs[4];
  const gsl_matrix * dxi_vals[4];
  
  for (i=0; i<4; i++)
  {
    dxi_eqs[i] = &dxi_eqs_views[i].matrix;
    dxi_vals[i] = &dxi_vals_views[i].matrix;
  }
  
  dxiRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, dxi_eqs, params);

  /* Equation for deta */
  gsl_matrix_view deta_eqs_views[4] = {gsl_matrix_view_array(f+5+16+1+16+16+64+64+0,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+64+16,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+64+32,4,4),
                               gsl_matrix_view_array(f+5+16+1+16+16+64+64+48,4,4)};
  gsl_matrix_const_view deta_vals_views[4] = {gsl_matrix_const_view_array(y+5+16+1+16+16+64+64+0,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+64+16,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+64+32,4,4),
                                      gsl_matrix_const_view_array(y+5+16+1+16+16+64+64+48,4,4)};
  gsl_matrix * deta_eqs[4];
  const gsl_matrix * deta_vals[4];
  
  for (i=0; i<4; i++)
  {
    deta_eqs[i] = &deta_eqs_views[i].matrix;
    deta_vals[i] = &deta_vals_views[i].matrix;
  }
  
  detaRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &eta_vals.matrix, deta_vals, deta_eqs, params);

  d2I_Inv(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &I_vals.matrix, dI_vals, y+5+16+1+16+16+64+64+64, f+5+16+1+16+16+64+64+64, params);
   
  d2xi(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, dxi_vals, &I_vals.matrix, dI_vals, y+5+16+1+16+16+64+64+64+256, f+5+16+1+16+16+64+64+64+256, params);
  
  const double *d2xi_vals = y+5+16+1+16+16+64+64+64+256;
  d2eta(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &eta_vals.matrix, &q_vals.matrix, dxi_vals, d2xi_vals, deta_vals, y+5+16+1+16+16+64+64+64+256+256, f+5+16+1+16+16+64+64+64+256+256, params);
  
  /* Calculate Box SqrtDelta */
  double box_sqrt_delta = 0;
  boxSqrtDelta (tau, y, &box_sqrt_delta, &params);
  V0RHS (tau, &q_vals.matrix, &box_sqrt_delta, y+5+16+1+16+16+64+64+64+256+256+1, f+5+16+1+16+16+64+64+64+256+256+1, params);
  
  return GSL_SUCCESS;
}

/* Calculate the Jacobian Matrix J_{ij} = df_i/dy_j and also the vector df_i/dt */
int jac (double tau, const double y[], double *dfdy, double dfdtau[], void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  
  /* df_i/dy_j */
  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 5, 5);
  gsl_matrix * m = &dfdy_mat.matrix; 

  gsl_matrix_set_zero(m);
  
  gsl_matrix_set (m, 0, 1, 1.0); /* dr/dr' */
  gsl_matrix_set (m, 1, 0, (3*gsl_pow_2(p.l)*(4*p.m - y[0]) - 2*p.m*p.type*gsl_pow_2(y[0]))/gsl_pow_5(y[0]) ); /* dr'/dr */
  gsl_matrix_set (m, 3, 0, -2*p.l/gsl_pow_3(y[0]) ); /* dphi/dr */
  gsl_matrix_set (m, 4, 0, -2*p.e*p.m/gsl_pow_2(y[0]-2*p.m) ); /* dt/dr */


  /* df_i/dtau */
  gsl_vector_view dfdtau_vec = gsl_vector_view_array (dfdtau, 5);
  gsl_vector * v = &dfdtau_vec.vector;
  
  gsl_vector_set_zero(v);

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
  
  gsl_odeiv_system sys = {func, jac, NUM_EQS, &params};

  double tau = 0.0, tau1 = 1000.0;
  double h = 1e-6;
  double r0 = 10.0;
  double m = 1.0;
  
  /* These are used in the initial conditions for d2xi */
  double r1 = m/(3*gsl_pow_2(r0)*(2*m-r0));
  double r2 = m/(3*r0);
  double r3 = (2*m-r0)*m/(3*gsl_pow_4(r0));
  
  /* Initial Condidions */
  double y[NUM_EQS] = { 
    r0, 0.0, 0.0, 0.0, 0.0, /* r, r', theta, phi, t */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* Q^a'_b' */
    1.0, /* Delta^1/2 */
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, /* I^a_b' */
   -1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, /* eta^a_b' */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a'_{ b' c'}, i.e. dxi */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a'_{ b' c'} */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a'_{ b' c'} */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a'_{ b' c'} */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a_{ b' c'}, i.e. deta */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a_{ b' c'} */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a_{ b' c'} */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* sigma^a_{ b' c'} */
    
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I FIXME: These should be something like Riemann */
    0.0, (-m/r0)/2, 0.0, 0.0, (m/r0)/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, (-m/r0)/2, 0.0, 0.0, 0.0, 0.0, 0.0, (m/r0)/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, (2*(2*m-r0)*m/gsl_pow_4(r0))/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -(2*(2*m-r0)*m/gsl_pow_4(r0))/2, 0.0, 0.0, 0.0, /* d2I */
    
    0.0, (-m/(gsl_pow_2(r0)*(2*m-r0)))/2, 0.0, 0.0, -(-m/(gsl_pow_2(r0)*(2*m-r0)))/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2*m/r0/2, 0.0, 0.0, -2*m/r0/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ((r0-2*m)*m/gsl_pow_4(r0))/2, 0.0, 0.0, 0.0, 0.0, 0.0, -((r0-2*m)*m/gsl_pow_4(r0))/2, 0.0, 0.0, /* d2I */
    
    0.0, 0.0, (m/(gsl_pow_2(r0)*(r0-2*m)))/2, 0.0, 0.0, 0.0, 0.0, 0.0, -(m/(gsl_pow_2(r0)*(r0-2*m)))/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2*m/r0/2, 0.0, 0.0, 2*m/r0/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ((r0-2*m)*m/gsl_pow_4(r0))/2, 0.0, 0.0, -((r0-2*m)*m/gsl_pow_4(r0))/2, 0.0, /* d2I */
    
    0.0, 0.0, 0.0, (2*m/(gsl_pow_2(r0)*(2*m-r0)))/2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -(2*m/(gsl_pow_2(r0)*(2*m-r0)))/2, 0.0, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m/r0/2, 0.0, 0.0, 0.0, 0.0, 0.0, -m/r0/2, 0.0, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, m/r0/2, 0.0, 0.0, -m/r0/2, 0.0, /* d2I */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* d2I */
    
    /* d2xi^{a'}_{  b' c' d'} (0) = -2/3* R^{a'}_{  (c' | b' | d')}*/
    /*        r                       theta                      phi                        t            */
    /* r, theta,phi,  t        r,  theta,  phi,  t       r,  theta,  phi,  t       r,  theta,  phi,  t   */
    0.0,  0.0,  0.0,  0.0,    0.0,  2*r2, 0.0,  0.0,    0.0,  0.0,  2*r2, 0.0,    0.0,  0.0,  0.0, -4*r3,/* r,r */
    0.0,  -r2,  0.0,  0.0,    -r2,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* r,theta */
    0.0,  0.0,  -r2,  0.0,    0.0,  0.0,  0.0,  0.0,    -r2,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* r,phi */
    0.0,  0.0,  0.0,  2*r3,   0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    2*r3, 0.0,  0.0,  0.0, /* r,t */
    
    0.0,   r1,  0.0,  0.0,     r1,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* theta,r */
    -2*r1,0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  -4*r2,0.0,    0.0,  0.0,  0.0,  2*r3,/* theta,theta */
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  2*r2, 0.0,    0.0,  2*r2, 0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* theta,phi */
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  -r3,    0.0,  0.0,  0.0,  0.0,    0.0,  -r3,  0.0,  0.0, /* theta,t */
    
    0.0,  0.0,   r1,  0.0,    0.0,  0.0,  0.0,  0.0,     r1,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* phi,r */
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  2*r2, 0.0,    0.0,  2*r2, 0.0,  0.0,    0.0,  0.0,  0.0,  0.0, /* phi,theta */
    -2*r1,0.0,  0.0,  0.0,    0.0,  -4*r2,0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  2*r3,/* phi,phi */ 
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  -r3,    0.0,  0.0,  -r3,  0.0, /* phi,t */
    
    0.0,  0.0,  0.0,  -2*r1,  0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    -2*r1,0.0,  0.0,  0.0, /* t,r */
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  -r2,    0.0,  0.0,  0.0,  0.0,    0.0,  -r2,  0.0,  0.0, /* t,theta */
    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  0.0,    0.0,  0.0,  0.0,  -r2,    0.0,  0.0,  -r2,  0.0, /* t,phi */
    4*r1, 0.0,  0.0,  0.0,    0.0,  2*r2, 0.0,  0.0,    0.0,  0.0,  2*r2, 0.0,    0.0,  0.0,  0.0,  0.0, /* t,t */
    }; 

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
