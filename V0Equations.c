/* Numerically solve the transport equation for V_0
 * along a geodesic. This file contains the RHS's for all the transport equations
 * required to calculate V_0 from Delta^1/2. Most are derivatives of sigma and in
 * that case they are named sigma_n_m_RHS to indicate n unprimed and m primed derivatives
 * of sigma and u means upstairs, d downstairs.
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

#define EPS	10e-18

/* Bivector of parallel displacement, g_{a'}^{~ a}. We use the defining equation
   g_{a' ~ ;b'}^{a} \sigma^{b'} = 0 */
int IRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, gsl_matrix * f, void * params)
{
  /* Gamma*u */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  /* RHS */
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, gu, I, 0., f);

  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

int etaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, const gsl_matrix * eta, gsl_matrix * f, void * params)
{
  int i;
  
  /* eta*Gu */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  gsl_matrix * eta_gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eta, gu, 0., eta_gu);
  
  /* Xi */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
  }

  /* RHS */
  gsl_matrix * eta_rhs = gsl_matrix_calloc(4,4);
  if(tau!=0.0)
  {
    gsl_matrix_memcpy(eta_rhs, eta);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, eta, xi, 1.0, eta_rhs);
    gsl_matrix_scale(eta_rhs, 1.0/tau);
  }
  gsl_matrix_add(eta_rhs, eta_gu);
  gsl_matrix_memcpy(f, eta_rhs);
  
  gsl_matrix_free(gu);
  gsl_matrix_free(eta_gu);
  gsl_matrix_free(xi);
  gsl_matrix_free(eta_rhs);

  return GSL_SUCCESS;
}

int dIinvRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, const gsl_matrix *q, const gsl_vector * dIinv, gsl_vector * f, void * params)
{
  int i, j, k, l;

  /* First, we need I^-1 */
  int signum=0;
  gsl_permutation * p = gsl_permutation_calloc (4);
  gsl_matrix * I_inv = gsl_matrix_calloc(4,4);
  gsl_matrix * lu = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(lu, I);
  gsl_linalg_LU_decomp (lu, p, &signum);
  gsl_linalg_LU_invert (lu, p, I_inv);

  /* And calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  gsl_vector_set_zero(f);

  /* Now, calculate sigma_R * I^(-1) */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i+4*j+k, gsl_vector_get(f, 16*i + 4*j + k) + gsl_vector_get(sigma_R, 16*j + 4*l +k)*gsl_matrix_get(I_inv, i, l));

  if(tau!=0.0)
  {
      /* And calculate dIinv * xi*/
      gsl_matrix * xi = gsl_matrix_calloc(4,4);
      gsl_matrix_memcpy(xi, q);

      for( i=0; i<4; i++)
      {
          gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
      }

      for(i=0; i<4; i++)
          for(j=0; j<4; j++)
              for(k=0; k<4; k++)
                  for(l=0; l<4; l++)
                      gsl_vector_set(f, 16*i+4*j+k, gsl_vector_get(f, 16*i + 4*j + k) - gsl_vector_get(dIinv, 16*i + 4*j + l)*gsl_matrix_get(xi, l, k)/tau);

    gsl_matrix_free(xi);
  } else {
      for(i=0; i<4; i++)
        for(j=0; j<4; j++)
          for(k=0; k<4; k++)
            gsl_vector_set(f, 16*i+4*j+k, gsl_vector_get(f, 16*i + 4*j + k) - 0.5 * gsl_vector_get(sigma_R, 16*j + 4*i + k));
  }

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i+4*j+k, gsl_vector_get(f, 16*i + 4*j + k) - gsl_vector_get(dIinv, 16*i + 4*l + k)*gsl_matrix_get(gu, j, l)
            + gsl_vector_get(dIinv, 16*i + 4*j + l)*gsl_matrix_get(gu, l, k));

  gsl_permutation_free(p);
  gsl_matrix_free(I_inv);
  gsl_matrix_free(lu);
  gsl_vector_free(sigma_R);
  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

/* RHS of transport equation for dxi. */
int dxiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, gsl_vector * f, void * params)
{
  int i, j, k, l;
  
  /* And calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  /* First, we need xi from q */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
  }

  gsl_vector_set_zero(f);
  if(tau!=0.0){
      /* The first term on the RHS is just dxi/tau */
      gsl_vector_memcpy(f, dxi);

      /* Now we  work out the three xi*dxi terms */
      for(i=0; i<4; i++)
          for(j=0; j<4; j++)
              for(k=0; k<4; k++)
                  for(l=0; l<4; l++)
                      gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                              - gsl_matrix_get(xi, i, l)*gsl_vector_get(dxi, 16*l + 4*j + k)
                              - gsl_matrix_get(xi, l, j)*gsl_vector_get(dxi, 16*i + 4*l + k)
                              - gsl_matrix_get(xi, l, k)*gsl_vector_get(dxi, 16*i + 4*l + j)
                              );

      gsl_vector_scale(f, 1./tau);
  } else {
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    gsl_vector_set(f, 16*0 + 4*0 + 3,   ((-1. + r * r) * ut) / 3.);
    gsl_vector_set(f, 16*0 + 4*3 + 0,  - ((-1. + r * r) * ut) / 3.);
    gsl_vector_set(f, 16*1 + 4*1 + 2,   uph / 3.);
    gsl_vector_set(f, 16*1 + 4*2 + 1,  - uph / 3.);
    gsl_vector_set(f, 16*3 + 4*0 + 3,   (1 / (-1 + r * r) * ur) / 3.);
    gsl_vector_set(f, 16*3 + 4*3 + 0,  - (1 / (-1 + r * r) * ur) / 3.);

    for(i=0; i<4; i++)
        for(j=0; j<4; j++)
            for(k=0; k<4; k++)
                gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                               -(2*gsl_vector_get(sigma_R, 16*i + 4*k + j)+gsl_vector_get(sigma_R, 16*i + 4*j + k)));
    }

  /* Now, calculate the three sigma_R * xi terms 
     FIXME: Missing the Riemann derivative term */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                         - gsl_matrix_get(xi, i, l)*gsl_vector_get(sigma_R, 16*l + 4*j + k)
                         + gsl_matrix_get(xi, l, j)*gsl_vector_get(sigma_R, 16*i + 4*l + k)
                         + gsl_matrix_get(xi, l, k)*gsl_vector_get(sigma_R, 16*i + 4*l + j)
                         );

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                         - gsl_matrix_get(gu, i, l)*gsl_vector_get(dxi, 16*l + 4*j + k)
                         + gsl_matrix_get(gu, l, j)*gsl_vector_get(dxi, 16*i + 4*l + k)
                         + gsl_matrix_get(gu, l, k)*gsl_vector_get(dxi, 16*i + 4*j + l));

  gsl_vector_free(sigma_R);
  gsl_matrix_free(xi);
  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

/* RHS of transport equation for deta. The matrix elements are the first two indices. */
int detaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * eta, const gsl_vector * deta, gsl_vector * f, void * params)
{
  int i, j, k, l;
  
  /* Calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  gsl_vector_set_zero(f);
  if(tau!=0.0)
  {
      /* First, we need xi from q */
      gsl_matrix * xi = gsl_matrix_calloc(4,4);
      gsl_matrix_memcpy(xi, q);
      for( i=0; i<4; i++)
      {
          gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
      }

      /* The first term on the RHS is just deta/tau */
      gsl_vector_memcpy(f, deta);

      /* Now we  work out the two xi*deta terms and one eta dxi term*/
      for(i=0; i<4; i++)
          for(j=0; j<4; j++)
              for(k=0; k<4; k++)
                  for(l=0; l<4; l++)
                      gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                              - gsl_matrix_get(xi, l, j)*gsl_vector_get(deta, 16*i + 4*l + k)
                              - gsl_matrix_get(xi, l, k)*gsl_vector_get(deta, 16*i + 4*j + l) /* Why the hell is this not ilj??? */
                              - gsl_matrix_get(eta, i, l)*gsl_vector_get(dxi, 16*l + 4*j + k)
                              );

      gsl_vector_scale(f, 1./tau);

      gsl_matrix_free(xi);
  } else {

    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    gsl_vector_set(f, 16*0 + 4*0 + 3,   2*((-1. + r * r) * ut) / 3.);
    gsl_vector_set(f, 16*0 + 4*3 + 0,  - 2*((-1. + r * r) * ut) / 3.);
    gsl_vector_set(f, 16*1 + 4*1 + 2,   2*uph / 3.);
    gsl_vector_set(f, 16*1 + 4*2 + 1,  - 2*uph / 3.);
    gsl_vector_set(f, 16*3 + 4*0 + 3,   2*(1 / (-1 + r * r) * ur) / 3.);
    gsl_vector_set(f, 16*3 + 4*3 + 0,  - 2*(1 / (-1 + r * r) * ur) / 3.);
  }

  /* Now, calculate the sigma_R * eta term */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                         - gsl_matrix_get(eta, i, l)*gsl_vector_get(sigma_R, 16*l + 4*j + k));

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          gsl_vector_set(f, 16*i + 4*j + k, gsl_vector_get(f, 16*i + 4*j + k)
                         + gsl_matrix_get(gu, l, j)*gsl_vector_get(deta, 16*i + 4*l + k)
                         + gsl_matrix_get(gu, l, k)*gsl_vector_get(deta, 16*i + 4*j + l));

  gsl_vector_free(sigma_R);
  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

int d2IinvRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * I, const gsl_vector * dIinv, const gsl_vector * d2Iinv, gsl_vector * f, void * params)
{
  int i, j, k, l, m;

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  /* xi */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);

  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
  }

  /* And calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  gsl_vector_set_zero(f);

  /* FIXME: Riemann derivative term missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            gsl_vector_set(f, 64*i+16*j+4*k+l, gsl_vector_get(f, 64*i+16*j+4*k+l)
                           /* gu * d2I */
                           + gsl_vector_get(d2Iinv, 64*i+16*j+4*m+l) * gsl_matrix_get(gu, m, k)
                           + gsl_vector_get(d2Iinv, 64*i+16*j+4*k+m) * gsl_matrix_get(gu, m, l)
                           - gsl_vector_get(d2Iinv, 64*m+16*j+4*k+l) * gsl_matrix_get(gu, i, m)

                           /* xi * d2I */
                           - (
                           + gsl_vector_get(d2Iinv, 64*i+16*j+4*m+l) * gsl_matrix_get(xi, m, k)
                           + gsl_vector_get(d2Iinv, 64*i+16*j+4*m+k) * gsl_matrix_get(xi, m, l)

                           /* dxi * dIinv */
                           + gsl_vector_get(dIinv, 16*i+4*j+m) * gsl_vector_get(dxi, 16*m+4*k+l)
                           )/(tau+EPS)

                           /* R_sigma * dIinv */
                           + gsl_vector_get(dIinv, 16*i+4*m+k)*gsl_vector_get(sigma_R, 16*j + 4*m + l)
                           + gsl_vector_get(dIinv, 16*i+4*m+l)*gsl_vector_get(sigma_R, 16*j + 4*m + k)
                           - gsl_vector_get(dIinv, 16*i+4*j+m)*gsl_vector_get(sigma_R, 16*m + 4*k + l)
                           );

  gsl_vector_free(sigma_R);
  gsl_matrix_free(xi);
  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

int d2xiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * I, const gsl_vector * dIinv, const gsl_vector * d2xi, gsl_vector * f, void * params)
{
  int i, j, k, l, m, n, o;

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  /* xi */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);

  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1);
  }

  /* Compute metric (in primed coordinates) */
  gsl_matrix   * metric     = gsl_matrix_calloc(4,4);
  gsl_matrix   * metric_dn  = gsl_matrix_calloc(4,4);
  metric_up_up(y->data, metric, params);
  metric_dn_dn(y->data, metric_dn, params);

  /* And calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  gsl_vector_memcpy(f, d2xi);
  gsl_vector_scale(f, 1./(tau+EPS));

  /* FIXME: Riemann derivative terms missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            gsl_vector_set(f, 64*i+16*j+4*k+l,
                           gsl_vector_get(f, 64*i+16*j+4*k+l)

                           + (

                           /* dxi * dxi */
                           - gsl_vector_get(dxi, 16*i+4*m+k) * gsl_vector_get(dxi, 16*m+4*j+l)
                           - gsl_vector_get(dxi, 16*i+4*m+j) * gsl_vector_get(dxi, 16*m+4*k+l)
                           - gsl_vector_get(dxi, 16*i+4*m+l) * gsl_vector_get(dxi, 16*m+4*j+k)

                           /* d2xi * xi */
                           - gsl_vector_get(d2xi, 64*m+16*j+4*k+l) * gsl_matrix_get(xi, i, m)
                           - gsl_vector_get(d2xi, 64*i+16*m+4*k+l) * gsl_matrix_get(xi, m, j)
                           - gsl_vector_get(d2xi, 64*i+16*m+4*j+l) * gsl_matrix_get(xi, m, k)
                           - gsl_vector_get(d2xi, 64*i+16*m+4*j+k) * gsl_matrix_get(xi, m, l)
                           )/(tau+EPS)

                           /* gu * d2xi */
                           + gsl_vector_get(d2xi, 64*i+16*m+4*k+l) * gsl_matrix_get(gu, m, j)
                           + gsl_vector_get(d2xi, 64*i+16*j+4*m+l) * gsl_matrix_get(gu, m, k)
                           + gsl_vector_get(d2xi, 64*i+16*j+4*k+m) * gsl_matrix_get(gu, m, l)
                           - gsl_vector_get(d2xi, 64*m+16*j+4*k+l) * gsl_matrix_get(gu, i, m)

                           /* r_sigma * r_sigma */
                          + (tau+EPS)*(
                           + gsl_vector_get(sigma_R, 16*i+4*m+l)*gsl_vector_get(sigma_R, 16*m + 4*k + j)
                           + gsl_vector_get(sigma_R, 16*i+4*m+k)*gsl_vector_get(sigma_R, 16*m + 4*l + j)
                           + gsl_vector_get(sigma_R, 16*i+4*m+j)*gsl_vector_get(sigma_R, 16*m + 4*l + k)
                           )

                           /* r_sigma * dxi */
                           - gsl_vector_get(dxi, 16*i+4*j+m)*gsl_vector_get(sigma_R, 16*m + 4*k + l)
                           - gsl_vector_get(dxi, 16*i+4*k+m)*gsl_vector_get(sigma_R, 16*m + 4*j + l)
                           - gsl_vector_get(dxi, 16*i+4*l+m)*gsl_vector_get(sigma_R, 16*m + 4*j + k)
                           );

    /* FIXME: Riemann derivative terms missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            for(n=0; n<4; n++)
              for(o=0; o<4; o++)
                gsl_vector_set(f, 64*i+16*j+4*k+l,
                               gsl_vector_get(f, 64*i+16*j+4*k+l)

                               /* r_sigma * r_sigma */
                               + (tau+EPS)*(
                               + gsl_vector_get(sigma_R, 16*m+4*l+n)*gsl_vector_get(sigma_R, 16*o + 4*m + k)*gsl_matrix_get(metric, n, i)*gsl_matrix_get(metric_dn, o, j)
                               + gsl_vector_get(sigma_R, 16*m+4*k+n)*gsl_vector_get(sigma_R, 16*o + 4*m + l)*gsl_matrix_get(metric, n, i)*gsl_matrix_get(metric_dn, o, j)
                               )

                               /* r_sigma * dxi */
                               + gsl_vector_get(dxi, 16*o+4*k+n)*gsl_vector_get(sigma_R, 16*i + 4*m + l)*gsl_matrix_get(metric_dn, o, j)*gsl_matrix_get(metric, n, m)
                               + gsl_vector_get(dxi, 16*o+4*l+n)*gsl_vector_get(sigma_R, 16*i + 4*m + k)*gsl_matrix_get(metric_dn, o, j)*gsl_matrix_get(metric, n, m)
                               + gsl_vector_get(dxi, 16*o+4*l+n)*gsl_vector_get(sigma_R, 16*i + 4*m + j)*gsl_matrix_get(metric_dn, o, k)*gsl_matrix_get(metric, n, m)
                               );

  gsl_vector_free(sigma_R);
  gsl_matrix_free(xi);
  gsl_matrix_free(gu);
  gsl_matrix_free(metric);
  gsl_matrix_free(metric_dn);

  return GSL_SUCCESS;
}

/* d2eta */
int d2etaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *eta, const gsl_matrix *q, const gsl_vector * dxi, const gsl_vector * d2xi, const gsl_vector * deta, const gsl_vector * d2eta, gsl_vector * f, void * params)
{
  int i, j, k, l, m;

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);

  /* xi */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);

  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1.);
  }

  /* And calculate sigma_R */
  gsl_vector * sigma_R = gsl_vector_calloc(4*4*4);
  R_sigma(y, yp, sigma_R, params);

  gsl_vector_set_zero( f );

  gsl_vector_memcpy(f, d2eta);
  gsl_vector_scale(f, 1./(tau+EPS));

  /* FIXME: Riemann derivative term missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            gsl_vector_set(f, 64*i+16*j+4*k+l,
                           gsl_vector_get(f, 64*i+16*j+4*k+l) + (

                           /* deta * dxi */
                           - gsl_vector_get(deta, 16*i+4*m+j) * gsl_vector_get(dxi, 16*m+4*k+l)
                           - gsl_vector_get(deta, 16*i+4*m+k) * gsl_vector_get(dxi, 16*m+4*j+l)
                           - gsl_vector_get(deta, 16*i+4*m+l) * gsl_vector_get(dxi, 16*m+4*j+k)

                           /* d2xi * eta and d2eta * xi */
                           - gsl_vector_get(d2xi, 64*m+16*j+4*k+l) * gsl_matrix_get(eta, i, m)
                           - gsl_vector_get(d2eta, 64*i+16*m+4*k+l) * gsl_matrix_get(xi, m, j)
                           - gsl_vector_get(d2eta, 64*i+16*m+4*j+l) * gsl_matrix_get(xi, m, k)
                           - gsl_vector_get(d2eta, 64*i+16*m+4*j+k) * gsl_matrix_get(xi, m, l)
                           )/(tau+EPS)

                           /* gu * d2eta */
                           + gsl_vector_get(d2eta, 64*i+16*m+4*k+l) * gsl_matrix_get(gu, m, j)
                           + gsl_vector_get(d2eta, 64*i+16*j+4*m+l) * gsl_matrix_get(gu, m, k)
                           + gsl_vector_get(d2eta, 64*i+16*j+4*k+m) * gsl_matrix_get(gu, m, l)

                           /* r_sigma * deta */
                           - gsl_vector_get(deta, 16*i+4*j+m)*gsl_vector_get(sigma_R, 16*m + 4*k + l)
                           - gsl_vector_get(deta, 16*i+4*k+m)*gsl_vector_get(sigma_R, 16*m + 4*j + l)
                           - gsl_vector_get(deta, 16*i+4*l+m)*gsl_vector_get(sigma_R, 16*m + 4*j + k)
                           );

  gsl_vector_free(sigma_R);
  gsl_matrix_free(xi);
  gsl_matrix_free(gu);

  return GSL_SUCCESS;
}

/* gamma is the matrix inverse of eta */
int gammaBitensor ( const gsl_matrix * eta, gsl_matrix * gamma )
{
  /* Gamma is the matrix inverse of eta */
  int signum;
  gsl_permutation * p = gsl_permutation_alloc (4);
  gsl_matrix * lu = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(lu, eta);
  gsl_linalg_LU_decomp (lu, p, &signum);
  gsl_linalg_LU_invert (lu, p, gamma);

  gsl_matrix_free(lu);
  gsl_permutation_free(p);

  return GSL_SUCCESS;
}

/* Box SqrtDelta */
int boxSqrtDelta (double tau, const double * y, double * f, void * params)
{
  int i, j, k, l, m, n;
  const double * I          = y+5+16+1;
  const double * dI_Inv     = y+5+16+1+16+16;
  const double * dEta       = y+5+16+1+16+16+64+64;
  const double * d2I_Inv    = y+5+16+1+16+16+64+64+64;
  const double * d2Eta      = y+5+16+1+16+16+64+64+64+256+256;
  const double * SqrtDelta  = y+5+16;
  gsl_matrix_const_view eta = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
  gsl_matrix   * gamma      = gsl_matrix_calloc(4,4);
  gsl_matrix   * metric     = gsl_matrix_calloc(4,4);

  /* Compute gamma matrix */
  gammaBitensor( &eta.matrix, gamma );

  /* Compute metric (in primed coordinates) */
  metric_up_up(y, metric, params);

  /* Initialize the RHS to 0 */
  *f = 0.;
  
  /* The vector tr(I dI^-1) */
  double trIdI_inv[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        trIdI_inv[i] += I[j*4+k]*dI_Inv[k*16+j*4+i];
      }
    }
  }

  /* The vector tr(gamma dEta^-1) */
  double trGammadEta[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        trGammadEta[i] += gsl_matrix_get(gamma, j, k) * dEta[16*k+4*j+i];
      }
    }
  }
  
  /* Now contracting over the free index) */
  double tr2 = 0.;
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      tr2 += (trIdI_inv[i]+trGammadEta[i])*(trIdI_inv[j]+trGammadEta[j])*gsl_matrix_get(metric,i,j);
    }
  }

  /* We really need half of this */
  tr2 /= 2.;
  
  /* Next, we need tr(I dI^-1 I dI^-1) */
  double trIdI_invIdI_inv = 0.;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            for(n=0; n<4; n++)
              trIdI_invIdI_inv += I[i*4+j]*dI_Inv[16*j+4*k+m]*I[4*k+l]*dI_Inv[16*l+4*i+n]*gsl_matrix_get(metric,m,n);

  /* Next, we need tr(gamma deta gamma deta) */
  double trGammadEtaGammadEta = 0.;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            for(n=0; n<4; n++)
              trGammadEtaGammadEta += gsl_matrix_get(gamma,i,j)*dEta[16*j+4*k+m]*gsl_matrix_get(gamma,k,l)*dEta[16*l+4*i+n]*gsl_matrix_get(metric,m,n);

  /* Now, tr(I * d2I_Inv) */
  double trId2I_Inv = 0.;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          trId2I_Inv += I[4*i+j]*d2I_Inv[64*j+16*i+4*k+l]*gsl_matrix_get(metric, k, l);

  /* Now, tr(gamma * d2I_Inv) */
  double trGammad2Eta = 0.;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          trGammad2Eta += gsl_matrix_get(gamma, i, j)*d2Eta[64*j+16*i+4*k+l]*gsl_matrix_get(metric, k, l);

  /* We have everything we need, now just calculate Box SqrtDelta */
  *(f) = (*SqrtDelta)*(tr2-trIdI_invIdI_inv-trGammadEtaGammadEta+trId2I_Inv+trGammad2Eta)/2.;

  gsl_matrix_free(metric);
  gsl_matrix_free(gamma);

  return GSL_SUCCESS;
}

/* tr2 term of Box SqrtDelta */
int tr2term (double tau, const double * y, double * f, void * params)
{
  int i, j, k;
  const double * I          = y+5+16+1;
  const double * dI_Inv     = y+5+16+1+16+16;
  const double * dEta       = y+5+16+1+16+16+64+64;
  gsl_matrix_const_view eta = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
  gsl_matrix   * gamma      = gsl_matrix_calloc(4,4);
  gsl_matrix   * metric     = gsl_matrix_calloc(4,4);

  /* Compute gamma matrix */
  gammaBitensor( &eta.matrix, gamma );

  /* Compute metric (in primed coordinates) */
  metric_up_up(y, metric, params);

  /* Initialize the RHS to 0 */
  *f = 0.;

  /* The vector tr(I dI^-1) */
  double trIdI_inv[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        trIdI_inv[i] += I[j*4+k]*dI_Inv[k*16+j*4+i];
      }
    }
  }

  /* The vector tr(gamma dEta^-1) */
  double trGammadEta[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        trGammadEta[i] += gsl_matrix_get(gamma, j, k) * dEta[16*k+4*j+i];
      }
    }
  }

  /* Now contracting over the free index) */
  double tr2 = 0.;
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      tr2 += (trIdI_inv[i]+trGammadEta[i])*(trIdI_inv[j]+trGammadEta[j])*gsl_matrix_get(metric,i,j);
    }
  }

  /* We have everything we need, now just calculate Box SqrtDelta */
  *(f) = tr2;

  gsl_matrix_free(metric);
  gsl_matrix_free(gamma);

  return GSL_SUCCESS;
}

/* V0: D'V_0 = -V0 - 1/2 V_0 ( xi - 4 ) - 1/2 Box (Delta^1/2) */
int V0RHS (double tau, const gsl_matrix * q, const double * dal_sqrt_delta, const double * v0, double * f, void * params)
{
  int i;
  double rhs = 0.;

  if(tau!=0.)
  {
      for(i=0; i<4; i++)
      {
          rhs -= gsl_matrix_get(q, i, i);
      }

      rhs = (rhs*(*v0)/2. - (*v0) - (*dal_sqrt_delta)/2.)/(tau);

      *f = rhs;
  }

  return GSL_SUCCESS;
}

/* Initial conditions */
int d2IinvInit(double * d2Iinv, double r0, void * params)
{
    int i;
    /* d2Iinv^{a'}_{  b' c' d'} (0) = 1/2* R^{a'}_{  b'  c'  d'}*/
    Riemann(d2Iinv, r0, params);
    for(i=0; i<4*4*4*4; i++)
      d2Iinv[i] *= 1./2.;

    return GSL_SUCCESS;
}

int d2xiInit(double * d2xi, double r0, void * params)
{
    int i;
    /* d2xi^{a'}_{  b' c' d'} (0) = -2/3* R^{a'}_{  (c' | b' | d')}*/
    RiemannSym(d2xi, r0, params);
    for(i=0; i<4*4*4*4; i++)
      d2xi[i] *= -2./3.;

    return GSL_SUCCESS;
}

int d2etaInit(double * d2eta, double r0, void * params)
{
    int i;
    /* d2xi^{a'}_{  b' c' d'} (0) = -2/3* R^{a'}_{  (c' | b' | d')}*/
    RiemannSym(d2eta, r0, params);

    gsl_vector * d2eta2 = gsl_vector_calloc(4*4*4*4);

    Riemann(d2eta2->data, r0, params);

    for(i=0; i<4*4*4*4; i++)
      d2eta[i] = -d2eta[i]/3. - d2eta2->data[i]/2.;

    gsl_vector_free(d2eta2);
    return GSL_SUCCESS;
}
