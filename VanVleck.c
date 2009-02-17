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

#define NUM_EQS	(5+16+1+16+16+64+64+64+256+256+1)
#define EPS	10e-12

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
  
  /* (Q^2 + Q)/tau */
  gsl_matrix_add(q2, q);
  gsl_matrix_scale(q2, 1/(tau+EPS));
  
  /* tau * S */
  gsl_matrix * tau_S = gsl_matrix_calloc(4,4);
  S(y, yp, tau_S, params);
  gsl_matrix_scale(tau_S, tau);
  
  /* Add them all together to get the RHS */
  gsl_matrix_set_zero(f);
  gsl_matrix_add(f,q_gu);
  gsl_matrix_sub(f,gu_q);
  gsl_matrix_sub(f, q2);
  gsl_matrix_sub(f,tau_S);
  
  /* The theta,theta component blows up as theta*cot(theta) and makes the numerical scheme break down.
     Since we know the analytic form, don't compute it numerically */
  //gsl_matrix_set(f,1,1,0.0);
  
  return GSL_SUCCESS;
}

int sqrtDeltaRHS (double tau, const gsl_matrix * q, const double * sqrt_delta, double * f, void * params)
{
  int i;
  double rhs = 0;
  
  for(i=0; i<4; i++)
  {
    rhs -= gsl_matrix_get(q, i, i);
  }
  
  rhs = rhs * (*sqrt_delta) / 2 / (tau + EPS);
  
  *f = rhs;
  
  return GSL_SUCCESS;
}

int IRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, gsl_matrix * f, void * params)
{
  /* Gu */
  gsl_matrix * gu = gsl_matrix_calloc(4,4);
  Gu(y, yp, gu, params);
  
  /* RHS */
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, I, gu, 0, f);
  
  return GSL_SUCCESS;
}

int etaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, const gsl_matrix * eta, gsl_matrix * f, void * params)
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
int dxiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], gsl_matrix * f[], void * params)
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
	    f[64*i+16*j+4*k+l] += d2I_Inv[64*i+16*j+4*m+l] * gu->data[4*m+k] 		/* gu * d2I */
				+ d2I_Inv[64*i+16*j+4*k+m] * gu->data[4*m+l]
				- d2I_Inv[64*m+16*j+4*k+l] * gu->data[4*i+m]
				- (d2I_Inv[64*i+16*j+4*m+l] * xi->data[4*m+k]		 /* xi * d2I */
				+  d2I_Inv[64*i+16*j+4*m+k] * xi->data[4*m+l]
				+ dxi[l]->data[4*m+k]* dI[m]->data[4*i+j])/(tau+EPS)	/* dxi * dI */
				+ sigma_R[l]->data[4*i+m]*dI[k]->data[4*m+j]		/* R_sigma * dI */
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
	    f[64*i+16*j+4*k+l] +=  d2xi[64*i+16*j+4*m+l] * gu->data[4*m+k] 		/* gu * d2I */
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
				- d2xi[64*i+16*m+4*j+k] * xi->data[4*m+l])/(tau+EPS);		/* dxi * dxi */


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
	    f[64*i+16*j+4*k+l] +=  d2eta[64*i+16*j+4*m+l] * gu->data[4*m+k] 		/* gu * d2I */
				+ d2eta[64*i+16*j+4*k+m] * gu->data[4*m+l]
				+ d2eta[64*i+16*m+4*k+l] * gu->data[4*m+j]
				+ (d2eta[64*m+16*j+4*k+l]
				- dxi[k]->data[4*m+j] * deta[l]->data[4*i+m]
				- dxi[l]->data[4*m+j] * deta[k]->data[4*i+m]
				- dxi[l]->data[4*m+k] * deta[j]->data[4*i+m]
				- d2xi[64*m+16*j+4*k+l] * eta->data[4*i+m]
				- d2eta[64*i+16*m+4*k+l] * xi->data[4*m+j]
				- d2eta[64*i+16*m+4*j+l] * xi->data[4*m+k]
				- d2eta[64*i+16*m+4*j+k] * xi->data[4*m+l])/(tau+EPS);		/* dxi * dxi */


  return GSL_SUCCESS;
}

/* Box SqrtDelta */
int boxSqrtDelta (double tau, const double * y, double * f, void * params)
{
  int i, j, k, l, m, n;
  const double * I = y+5+16+1;
  const double * dI_Inv = y+5+16+1+16+16;
  const double * dEta = y+5+16+1+16+16+64+64;
  const double * d2I_Inv = y+5+16+1+16+16+64+64+64;
  const double * d2Eta = y+5+16+1+16+16+64+64+64+256+256;
  const double * SqrtDelta = y+5+16;
  
  /* Gamma is the matrix inverse of eta */
  int signum;
  gsl_permutation * p = gsl_permutation_alloc (4);
  gsl_matrix_const_view eta = gsl_matrix_const_view_array(y+5+16+1+16,4,4);
  gsl_matrix * gamma = gsl_matrix_calloc(4,4);
  gsl_matrix * lu = gsl_matrix_calloc(4,4);
  gsl_matrix_memcpy(lu, &eta.matrix);
  gsl_linalg_LU_decomp (lu, p, &signum);
  gsl_linalg_LU_invert (lu, p, gamma);
    
  /* Initialize the RHS to 0 */
  memset(f, 0, sizeof(double));
  
  /* The vector tr(I dI^-1) */
  double trIdI_inv[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
	trIdI_inv[i] += I[j*4+k]*dI_Inv[16*i+k*4+j];
      }
    }
  }
  
  /* The vector tr(I dI^-1) */
  double trGammadEta[4] = {0, 0, 0, 0};
  for(i=0; i<4; i++)
  {
    for(j=0; j<4; j++)
    {
      for(k=0; k<4; k++)
      {
	trGammadEta[i] += gamma->data[j*4+k]*dEta[16*i+k*4+j];
      }
    }
  }
  
  /* Now calculate the square of the sum of these (really we're contracting over the free index) */
  gsl_matrix * metric = gsl_matrix_calloc(4,4);
  metric_up_up(y, metric, params);
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
	      trIdI_invIdI_inv += I[i*4+j]*dI_Inv[16*m+j*4+k]*I[k*4+l]*dI_Inv[16*n+l*4+i]*gsl_matrix_get(metric,m,n);

  /* Next, we need tr(gamma deta gamma deta) */
  double trGammadEtaGammadEta = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
	for(l=0; l<4; l++)
	  for(m=0; m<4; m++)
	    for(n=0; n<4; n++)
	      trGammadEtaGammadEta += gamma->data[i*4+j]*dEta[16*m+j*4+k]*gamma->data[k*4+l]*dEta[16*n+l*4+i]*gsl_matrix_get(metric,m,n);

  /* Now, tr(I * d2I_Inv) */
  double trId2I_Inv = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
	trId2I_Inv += I[4*i+j]*d2I_Inv[64*j+16*i+4*k+k];

  /* Now, tr(gamma * d2I_Inv) */
  double trGammad2Eta = 0;
  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      for(k=0; k<4; k++)
	trGammad2Eta += gamma->data[4*i+j]*d2Eta[64*j+16*i+4*k+k];

  /* We have everything we need, now just calculate Box SqrtDelta */
  *(f) = (*SqrtDelta)*(tr2-trIdI_invIdI_inv-trGammadEtaGammadEta+trId2I_Inv+trGammad2Eta)/2;
  
  return GSL_SUCCESS;
}

/* V0 */
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
  
  dIRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &I_vals.matrix, &q_vals.matrix, dI_vals, dI_eqs, params);
  
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