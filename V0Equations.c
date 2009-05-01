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

#define EPS	10e-12

/* Bivector of parallel displacement, g_{a'}^{~ a}. We use the defining equation
   g_{a' ~ ;b'}^{a} \sigma^{b'} = 0 */
int I_RHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, gsl_matrix * f, void * params)
{
  /* Gamma*u */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  /* RHS */
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, I, gu, 0, f);
  
  return GSL_SUCCESS;
}

int sigma_1u_1d_RHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, const gsl_matrix * eta, gsl_matrix * f, void * params)
{
  int i;
  
  /* eta*Gu */
  gsl_matrix * eta_gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, eta_gu, params);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eta, eta_gu, 0, eta_gu);
  
  /* Xi */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1);
  }

  /* RHS */
  gsl_matrix * eta_rhs = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(eta_rhs, eta);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), eta, xi, 1.0/(tau+EPS), eta_rhs);
  gsl_matrix_add(eta_rhs, eta_gu);
  gsl_matrix_memcpy(f, eta_rhs);
  
  return GSL_SUCCESS;
}

int dIRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, const gsl_matrix *q, const gsl_matrix * dI[], gsl_matrix * f[], void * params)
{
  int i, j, k;
  
  /* First, we need I^-1 */
  int signum;
  gsl_permutation * p = gsl_permutation_alloc (4);
  gsl_matrix * I_inv = gsl_matrix_calloc(4,4);
  gsl_matrix * lu = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(lu, I);
  gsl_linalg_LU_decomp (lu, p, &signum);
  gsl_linalg_LU_invert (lu, p, I_inv);
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);
  
  /* Now, calculate sigma_R * I^(-1) */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sigma_R[i], I_inv, 0.0, f[i]);
  }
  
  /* And calculate dI * xi and store it in f2 for now since the elements will be in the wrong order */
  gsl_matrix * f2[4] = {gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4)};
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1);
  }
  
  /* Assuming the tensor data is in a contiguous block of memory, creat a view where each matrix corresponds to indices 2 and 3 */
  gsl_matrix_view dI2_view[4] = {gsl_matrix_view_array_with_tda (&dI[0]->data[0], 4, 4, 16),
                            gsl_matrix_view_array_with_tda (&dI[0]->data[4], 4, 4, 16),
                            gsl_matrix_view_array_with_tda (&dI[0]->data[8], 4, 4, 16),
                            gsl_matrix_view_array_with_tda (&dI[0]->data[12], 4, 4, 16)};
  gsl_matrix * dI2[4];
  for(i=0; i<4; i++)
  {
    dI2[i] = &dI2_view[i].matrix;
  }
  
  /* When multiplying, we need to use the transpose of dI2 since the order of the indices is wrong */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0/(tau+EPS), dI2[i], xi, 1.0, f2[i]);
  }
  
  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, gu, dI[i], 1.0, f[i]);
  }
  
  /* Store this in f2 too since it's the wrong order */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, dI2[i], gu, 1.0, f2[i]);
  }
  
  /* Now, fix the ordering of f2 and add it into f */
  gsl_matrix * f2_reordered[4] = {gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4)};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        gsl_matrix_set(f2_reordered[i], j, k, gsl_matrix_get(f2[j], k, i));
      }
    }
  }
  
  return GSL_SUCCESS;
}

/* RHS of transport equation for dxi. The matrix elements are the first two indices. */
int sigma_dxiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], gsl_matrix * f[], void * params)
{
  int i, j, k;
  
  /* First, we need xi from q */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1);
  }
  
  /* Let's also create an array of matrices for dxi where the matrix elements are the first and third indices */
  gsl_matrix * dxi2[4] = {gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4)};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        gsl_matrix_set(dxi2[k], i, j, gsl_matrix_get(dxi[j], i, k));
      }
    }
  }
  
  /* The first term on the RHS is just dxi/tau */
  for(i=0; i<4; i++)
  {
    gsl_matrix_memcpy(f[i],dxi[i]);
    gsl_matrix_scale(f[i],1/(tau+EPS));
  }
  
  /* Now we can easily work out the three xi*dxi terms, one of which is in the wrong order so is stored in f2 for now */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), xi, dxi[i], 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), dxi[i], xi, 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), dxi2[i], xi, 1.0, f[i]);
  }
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);

  /* We need sigma_R also with the matrix elements the first and third free indices of the tensor */
  gsl_matrix * sigma_R2[4] = {gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4)};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        gsl_matrix_set(sigma_R2[k], i, j, gsl_matrix_get(sigma_R[j], i, k));
      }
    }
  }
  
  /* Now, calculate the three sigma_R * xi terms 
     FIXME: Missing the Riemann derivative term */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,-1.0, xi, sigma_R[i], 0.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sigma_R[i], xi, 0.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, sigma_R2[i], xi, 0.0, f[i]);
  }

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gu, dxi[i], 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,-1.0, dxi[i], gu, 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,-1.0, dxi2[i], gu, 1.0, f[i]);
  }

  return GSL_SUCCESS;
}

/* RHS of transport equation for deta. The matrix elements are the first two indices. */
int detaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * eta, const gsl_matrix * deta[], gsl_matrix * f[], void * params)
{
  int i, j, k;
  
  /* First, we need xi from q */
  gsl_matrix * xi = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(xi, q);
  for( i=0; i<4; i++)
  {
    gsl_matrix_set(xi,i,i, gsl_matrix_get(xi,i,i) + 1);
  }
  
  /* Let's also create an array of matrices for deta where the matrix elements are the first and third indices */
  gsl_matrix * deta2[4] = {gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4), gsl_matrix_calloc(4,4)};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
        gsl_matrix_set(deta2[k], i, j, gsl_matrix_get(deta[j], i, k));
      }
    }
  }
  
  /* The first term on the RHS is just deta/tau */
  for(i=0; i<4; i++)
  {
    gsl_matrix_memcpy(f[i],deta[i]);
    gsl_matrix_scale(f[i],1/(tau+EPS));
  }
  
  /* Now we can easily work out the two eta*deta terms, and one eta*dxi term */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), deta[i], xi, 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), deta2[i], xi, 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0/(tau+EPS), xi, dxi[i], 1.0, f[i]);
  }
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);

  /* Now, calculate the sigma_R * eta terms */
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,-1.0, eta, sigma_R[i], 1.0, f[i]);;
  }

  /* Christoffel terms */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  for(i=0; i<4; i++)
  {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, deta[i], gu, 1.0, f[i]);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, deta2[i], gu, 1.0, f[i]);
  }

  return GSL_SUCCESS;
}

int d2I_Inv (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * I, const gsl_matrix * dI[], const double * d2I_Inv, double * f, void * params)
{
  int i, j, k, l, m;
  
  /* Initialize the RHS to 0 */
  memset(f, 0, 256*sizeof(double));
  
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
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);
  
  /* FIXME: Riemann derivative term missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            f[64*i+16*j+4*k+l] += d2I_Inv[64*i+16*j+4*m+l] * gu->data[4*m+k]            /* gu * d2I */
                                + d2I_Inv[64*i+16*j+4*k+m] * gu->data[4*m+l]
                                - d2I_Inv[64*m+16*j+4*k+l] * gu->data[4*i+m]
                                - (d2I_Inv[64*i+16*j+4*m+l] * xi->data[4*m+k]            /* xi * d2I */
                                +  d2I_Inv[64*i+16*j+4*m+k] * xi->data[4*m+l]
                                + dxi[l]->data[4*m+k]* dI[m]->data[4*i+j])/(tau+EPS)    /* dxi * dI */
                                + sigma_R[l]->data[4*i+m]*dI[k]->data[4*m+j]            /* R_sigma * dI */
                                + sigma_R[k]->data[4*i+m]*dI[l]->data[4*m+j]
                                - sigma_R[l]->data[4*m+k]*dI[m]->data[4*i+j];

  return GSL_SUCCESS;
}

int d2xi (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * I, const gsl_matrix * dI[], const double * d2xi, double * f, void * params)
{
  int i, j, k, l, m;
  
  /* Initialize the RHS to 0 */
  memset(f, 0, 256*sizeof(double));
  
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
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);
  
  /* FIXME: Riemann derivative term missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            f[64*i+16*j+4*k+l] +=  d2xi[64*i+16*j+4*m+l] * gu->data[4*m+k]              /* gu * d2I */
                                + d2xi[64*i+16*j+4*k+m] * gu->data[4*m+l]
                                + d2xi[64*i+16*m+4*k+l] * gu->data[4*m+j]
                                - d2xi[64*m+16*j+4*k+l] * gu->data[4*i+m]
                                + (d2xi[64*m+16*j+4*k+l]
                                - dxi[k]->data[4*i+m] * dxi[l]->data[4*m+j]
                                - dxi[j]->data[4*i+m] * dxi[l]->data[4*m+k]
                                - dxi[l]->data[4*i+m] * dxi[k]->data[4*m+j]
                                - d2xi[64*m+16*j+4*k+l] * xi->data[4*i+m]
                                - d2xi[64*i+16*m+4*k+l] * xi->data[4*m+j]
                                - d2xi[64*i+16*m+4*j+l] * xi->data[4*m+k]
                                - d2xi[64*i+16*m+4*j+k] * xi->data[4*m+l])/(tau+EPS);           /* dxi * dxi */


  return GSL_SUCCESS;
}

/* d2eta */
int d2eta (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *eta, const gsl_matrix *q, const gsl_matrix * dxi[], const double * d2xi, const gsl_matrix * deta[], const double * d2eta, double * f, void * params)
{
  int i, j, k, l, m;
  
  /* Initialize the RHS to 0 */
  memset(f, 0, 256*sizeof(double));
  
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
  
  /* And calculate sigma_R */
  gsl_matrix * sigma_R[4];
  for (i=0; i<4; i++)
  {
    sigma_R[i] = gsl_matrix_alloc(4,4);
  }
  R_sigma(y, yp, sigma_R, params);
  
  /* FIXME: Riemann derivative term missing */
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            f[64*i+16*j+4*k+l] +=  d2eta[64*i+16*j+4*m+l] * gu->data[4*m+k]             /* gu * d2I */
                                + d2eta[64*i+16*j+4*k+m] * gu->data[4*m+l]
                                + d2eta[64*i+16*m+4*k+l] * gu->data[4*m+j]
                                + (d2eta[64*m+16*j+4*k+l]
                                - dxi[k]->data[4*m+j] * deta[l]->data[4*i+m]
                                - dxi[l]->data[4*m+j] * deta[k]->data[4*i+m]
                                - dxi[l]->data[4*m+k] * deta[j]->data[4*i+m]
                                - d2xi[64*m+16*j+4*k+l] * eta->data[4*i+m]
                                - d2eta[64*i+16*m+4*k+l] * xi->data[4*m+j]
                                - d2eta[64*i+16*m+4*j+l] * xi->data[4*m+k]
                                - d2eta[64*i+16*m+4*j+k] * xi->data[4*m+l])/(tau+EPS);          /* dxi * dxi */


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
  *f = 0;
  
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
  double tr2 = 0;
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      tr2 += (trIdI_inv[i]+trGammadEta[i])*(trIdI_inv[j]+trGammadEta[j])*gsl_matrix_get(metric,i,j);
    }
  }
  
  /* We really need half of this */
  tr2 /= 2;
  
  /* Next, we need tr(I dI^-1 I dI^-1) */
  double trIdI_invIdI_inv = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            for(n=0; n<4; n++)
              trIdI_invIdI_inv += I[i*4+j]*dI_Inv[16*j+4*k+m]*I[4*k+l]*dI_Inv[16*l+4*i+n]*gsl_matrix_get(metric,m,n);

  /* Next, we need tr(gamma deta gamma deta) */
  double trGammadEtaGammadEta = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          for(m=0; m<4; m++)
            for(n=0; n<4; n++)
              trGammadEtaGammadEta += gsl_matrix_get(gamma,i,j)*dEta[16*j+4*k+m]*gsl_matrix_get(gamma,k,l)*dEta[16*l+4*i+n]*gsl_matrix_get(metric,m,n);

  /* Now, tr(I * d2I_Inv) */
  double trId2I_Inv = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          trId2I_Inv += I[4*i+j]*d2I_Inv[64*j+16*i+4*k+l]*gsl_matrix_get(metric, k, l);

  /* Now, tr(gamma * d2I_Inv) */
  double trGammad2Eta = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
        for(l=0; l<4; l++)
          trGammad2Eta += gsl_matrix_get(gamma, i, j)*d2Eta[64*j+16*i+4*k+l]*gsl_matrix_get(metric, k, l);

  /* We have everything we need, now just calculate Box SqrtDelta */
  *(f) = (*SqrtDelta)*(tr2-trIdI_invIdI_inv-trGammadEtaGammadEta+trId2I_Inv+trGammad2Eta)/2;
  
  return GSL_SUCCESS;
}

/* V0: D'V_0 = -V0 - 1/2 V_0 ( xi - 4 ) - 1/2 Box (Delta^1/2) */
int V0RHS (double tau, const gsl_matrix * q, const double * dal_sqrt_delta, const double * v0, double * f, void * params)
{
  int i;
  double rhs = 0;
  
  for(i=0; i<4; i++)
  {
    rhs -= gsl_matrix_get(q, i, i);
  }
  
  rhs = (rhs*(*v0)/2 - (*v0) - (*dal_sqrt_delta)/2)/(tau + EPS);
  
  *f = rhs;
  
  return GSL_SUCCESS;
}
