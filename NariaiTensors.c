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
    double r = y[0];

    gsl_matrix_set_zero(metric);

    gsl_matrix_set(metric,0,0,1.-gsl_pow_2(r));
    gsl_matrix_set(metric,1,1,1);
    gsl_matrix_set(metric,2,2,1);
    gsl_matrix_set(metric,3,3,1./(gsl_pow_2(r)-1.));

    return GSL_SUCCESS;
}

/* The covariant metric components for Nariai (with theta=Pi/2) */
int metric_dn_dn(const double *y, gsl_matrix *metric, void *params)
{
    double r = y[0];

    gsl_matrix_set_zero(metric);

    gsl_matrix_set(metric,0,0,1./(1.-gsl_pow_2(r)));
    gsl_matrix_set(metric,1,1,1);
    gsl_matrix_set(metric,2,2,1);
    gsl_matrix_set(metric,3,3,gsl_pow_2(r)-1.);

    return GSL_SUCCESS;
}

/* Calculates the matrix S^a_b = R^a_{ c b d} u^c u^d and fill the values into s.
 * Note that we have already set theat=Pi/2
 */
int S (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *s, void *params)
{
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_matrix_set_zero(s);

    /* Set non-zero elements */
    gsl_matrix_set(s,0,0,(-1. + r * r) * ut * ut);
    gsl_matrix_set(s,0,3,-(-1. + r * r) * ut * ur);
    gsl_matrix_set(s,1,1, uph * uph);
    gsl_matrix_set(s,3,0,1. / (-1. + r * r) * ur * ut);
    gsl_matrix_set(s,3,3,-1. / (-1. + r * r) * ur * ur);

    return GSL_SUCCESS;
}

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ b d c} u^d and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int R_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_vector * r_sigma, void *params)
{
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_vector_set_zero(r_sigma);

    gsl_vector_set(r_sigma, 0*16 + 3*4 + 0, -(-1. + r * r) * ut);	// r,t,r
    gsl_vector_set(r_sigma, 0*16 + 3*4 + 3, (-1. + r * r) * ur);		// r,t,t
    gsl_vector_set(r_sigma, 1*16 + 2*4 + 1, -uph);			// theta,phi,theta
    gsl_vector_set(r_sigma, 2*16 + 1*4 + 1, uph);				// phi, theta, theta
    gsl_vector_set(r_sigma, 3*16 + 0*4 + 0, -ut / (-1. + r * r) );	// t,r,r
    gsl_vector_set(r_sigma, 3*16 + 0*4 + 3, ur / (-1. + r * r) );	// t,r,t

    return GSL_SUCCESS;
}

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ d b c} u^d and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int R_sigma_alt (const gsl_vector * y, const gsl_vector * yp, gsl_vector * r_sigma, void *params)
{
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_vector_set_zero(r_sigma);

    gsl_vector_set(r_sigma, 16*0 + 4*0 + 3,   (-1. + r * r) * ut);
    gsl_vector_set(r_sigma, 16*0 + 4*3 + 0,  - (-1. + r * r) * ut);
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 2,   uph);
    gsl_vector_set(r_sigma, 16*1 + 4*2 + 1,  - uph);
    gsl_vector_set(r_sigma, 16*3 + 4*0 + 3,   1 / (-1 + r * r) * ur);
    gsl_vector_set(r_sigma, 16*3 + 4*3 + 0,  - 1 / (-1 + r * r) * ur);

    return GSL_SUCCESS;
}
/* Calculates the tensor dRsigma2^a_{ b c} = R^a_{ d b e ;c} u^{d} u^{e} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int dR_sigma2 (const gsl_vector * y, const gsl_vector * yp, gsl_vector * dr_sigma2, void *params)
{
    /* Initialize all elements to 0 */
    gsl_vector_set_zero(dr_sigma2);

    return GSL_SUCCESS;
}

/* Calculates the tensor dRsigma^a_{ b c} = R^a_{ b e c;d} u^{e} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int dR_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_vector * dr_sigma, void *params)
{
    /* Initialize all elements to 0 */
    gsl_vector_set_zero(dr_sigma);

    return GSL_SUCCESS;
}


/* Calculates the tensor d2Rsigma^a_{ b c d} = R^a_{ b e c;d} u^{e} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int d2R_sigma2 (const gsl_vector * y, const gsl_vector * yp, gsl_vector * d2r_sigma2, void *params)
{
    /* Initialize all elements to 0 */
    gsl_vector_set_zero(d2r_sigma2);

    return GSL_SUCCESS;
}

/* Calculates the matrix Gu^a_b = \Gamma^a_{b c} u^c and fill the values into gu */
int Gu (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *gu, void *params)
{
    double ur = gsl_vector_get(yp,0);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    gsl_matrix_set_zero(gu);

    gsl_matrix_set(gu,0,0,-r / (-1. + r * r) * ur);
    gsl_matrix_set(gu,0,3,(-1. + r * r) * r * ut);
    gsl_matrix_set(gu,3,0,r / (-1. + r * r) * ut);
    gsl_matrix_set(gu,3,3,r / (-1. + r * r) * ur);

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

/* The Riemann tensor R^{a}_{~ b c d} */
int Riemann(double * R, double r0, void * params)
{
    int r=0, th=1, ph=2, t=3;

    R(r,t,r,t)      = r0*r0 - 1.;
    R(r,t,t,r)      = 1- r0*r0;
    R(th,ph,th,ph)  = 1;
    R(th,ph,ph,th)  = -1;
    R(ph,th,th,ph)  = -1;
    R(ph,th,ph,th)  = 1;
    R(t,r,r,t)      = 1./(r0*r0-1.);
    R(t,r,t,r)      = 1./(1.-r0*r0);

    return GSL_SUCCESS;
}

/* Riemann tensor symmetrized over second and 4th indices, RiemannSym^{a}_{~ b c d} = R^{a}_{~ (c |b| d)} */
int RiemannSym(double * R, double r0, void * params)
{
    int r=0, th=1, ph=2, t=3;

    R(r,r,t,t) = -1. + r0 * r0;
    R(r,t,r,t) = (1. - r0*r0 ) / 2.;
    R(r,t,t,r) = (1. - r0*r0) / 2.;
    R(th,th,ph,ph) = 1.;
    R(th,ph,th,ph) = -0.5;
    R(th,ph,ph,th) = -0.5;
    R(ph,th,th,ph) = -0.5;
    R(ph,th,ph,th) = -0.5;
    R(ph,ph,th,th) = 1.;
    R(t,r,r,t) = 1. / (-1. + r0*r0) / 2.;
    R(t,r,t,r) = 1. / (-1. + r0*r0) / 2.;
    R(t,t,r,r) = 1. / (1. - r0*r0);

    return GSL_SUCCESS;
}

double RicciScalar()
{
    return 4.;
}

/* Covariant derivative of Ricci scalar contracted with 4-velocity */
double d_RicciScalar (const gsl_vector * y, const gsl_vector * yp, void *params)
{
    return 0;
}

