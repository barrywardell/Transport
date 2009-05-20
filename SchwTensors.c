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

    gsl_matrix_set(metric,0,0,(r-2.0*m)/r);
    gsl_matrix_set(metric,1,1,1/gsl_pow_2(r));
    gsl_matrix_set(metric,2,2,1/gsl_pow_2(r));
    gsl_matrix_set(metric,3,3,-r/(r-2.0*m));

    return GSL_SUCCESS;
}

/* The covariant metric components for Schwarzschild (with theta=Pi/2) */
int metric_dn_dn(const double *y, gsl_matrix *metric, void *params)
{
    gsl_matrix_set_zero(metric);

    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;
    double r = y[0];

    gsl_matrix_set(metric,0,0,r/(r-2.0*m));
    gsl_matrix_set(metric,1,1,gsl_pow_2(r));
    gsl_matrix_set(metric,2,2,gsl_pow_2(r));
    gsl_matrix_set(metric,3,3,-(r-2.0*m)/r);

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

    gsl_matrix_set(s,0,0,(m*(4.0*m*gsl_pow_int(ut,2) - 2.0*r*gsl_pow_int(ut,2) - gsl_pow_int(r,3)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/gsl_pow_int(r,4));
    gsl_matrix_set(s,0,1,(m*ur*uth)/r);
    gsl_matrix_set(s,0,2,(m*up*ur)/r);
    gsl_matrix_set(s,0,3,(2.0*m*(-2.0*m + r)*ur*ut)/gsl_pow_int(r,4));
    gsl_matrix_set(s,1,0,-((m*ur*uth)/((2.0*m - r)*gsl_pow_int(r,2))));
    gsl_matrix_set(s,1,1,(m*gsl_pow_int(r,2)*(2.0*(2.0*m - r)*r*gsl_pow_int(up,2) + gsl_pow_int(ur,2)) - m*gsl_pow_int(-2.0*m + r,2)*gsl_pow_int(ut,2))/((2.0*m - r)*gsl_pow_int(r,4)));
    gsl_matrix_set(s,1,2,(-2.0*m*up*uth)/r);
    gsl_matrix_set(s,1,3,(m*(2.0*m - r)*ut*uth)/gsl_pow_int(r,4));
    gsl_matrix_set(s,2,0,-((m*up*ur)/((2.0*m - r)*gsl_pow_int(r,2))));
    gsl_matrix_set(s,2,1,(-2.0*m*up*uth)/r);
    gsl_matrix_set(s,2,2,(m*(-4.0*gsl_pow_int(m,2)*gsl_pow_int(ut,2) + r*(4.0*m*gsl_pow_int(ut,2) + r*(gsl_pow_int(ur,2) - gsl_pow_int(ut,2) - 2.0*r*(-2.0*m + r)*gsl_pow_int(uth,2)))))/((2.0*m - r)*gsl_pow_int(r,4)));
    gsl_matrix_set(s,2,3,(m*(2.0*m - r)*up*ut)/gsl_pow_int(r,4));
    gsl_matrix_set(s,3,0,(2.0*m*ur*ut)/((2.0*m - r)*gsl_pow_int(r,2)));
    gsl_matrix_set(s,3,1,(m*ut*uth)/r);
    gsl_matrix_set(s,3,2,(m*up*ut)/r);
    gsl_matrix_set(s,3,3,(m*(-2.0*gsl_pow_int(ur,2) + r*(-2.0*m + r)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/ ((2.0*m - r)*gsl_pow_int(r,2)));

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
    gsl_vector_set(r_sigma, 16*0 + 4*0 + 3, -(2.0*(-r+2.0*m))*m*ut/gsl_pow_4(r));
    gsl_vector_set(r_sigma, 16*3 + 4*0 + 3, (2.0*(-r+2.0*m))*m*ur/gsl_pow_4(r));
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 0, -m*ur/(gsl_pow_2(r)*(-r+2.0*m)));
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 2, -2.0*m*uph/r);
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 3, (-r+2.0*m)*m*ut/gsl_pow_4(r));
    gsl_vector_set(r_sigma, 16*0 + 4*2 + 0, m*uph/(gsl_pow_2(r)*(-r+2.0*m)));
    gsl_vector_set(r_sigma, 16*2 + 4*2 + 0, -m*ur/(gsl_pow_2(r)*(-r+2.0*m)));
    gsl_vector_set(r_sigma, 16*1 + 4*2 + 1, 2.0*m*uph/r);
    gsl_vector_set(r_sigma, 16*2 + 4*2 + 3, (-r+2.0*m)*m*ut/gsl_pow_4(r));
    gsl_vector_set(r_sigma, 16*3 + 4*2 + 3, -(-r+2.0*m)*m*uph/gsl_pow_4(r));
    gsl_vector_set(r_sigma, 16*0 + 4*3 + 0, -2.0*m*ut/(gsl_pow_2(r)*(-r+2.0*m)));
    gsl_vector_set(r_sigma, 16*3 + 4*3 + 0, 2.0*m*ur/(gsl_pow_2(r)*(-r+2.0*m)));
    gsl_vector_set(r_sigma, 16*1 + 4*3 + 1, -m*ut/r);
    gsl_vector_set(r_sigma, 16*2 + 4*3 + 2, -m*ut/r);
    gsl_vector_set(r_sigma, 16*3 + 4*3 + 2, m*uph/r);

    return GSL_SUCCESS;
}

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ d b c} u^d and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int R_sigma_alt (const gsl_vector * y, const gsl_vector * yp, gsl_vector * r_sigma, void *params)
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
    gsl_vector_set(r_sigma, 16*0 + 4*0 + 2, -m / r * uph);
    gsl_vector_set(r_sigma, 16*0 + 4*0 + 3, 2.0 *(-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*0 + 4*2 + 0, m / r * uph);
    gsl_vector_set(r_sigma, 16*0 + 4*3 + 0, -2.0 *(-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*1 + 4*0 + 1, -pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 0, pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 2, 2.0 *m / r * uph);
    gsl_vector_set(r_sigma, 16*1 + 4*1 + 3, -(-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*1 + 4*2 + 1, -2.0 *m / r * uph);
    gsl_vector_set(r_sigma, 16*1 + 4*3 + 1, (-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*2 + 4*0 + 2, -pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*2 + 4*2 + 0, pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*2 + 4*2 + 3, -(-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*2 + 4*3 + 2, (-r + 2.0 *m) * m * pow(r, -4.0) * ut);
    gsl_vector_set(r_sigma, 16*3 + 4*0 + 3, 2.0 *pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*3 + 4*2 + 3, m / r * uph);
    gsl_vector_set(r_sigma, 16*3 + 4*3 + 0, -2.0 *pow(r, -2.0) * m / (-r + 2.0 *m) * ur);
    gsl_vector_set(r_sigma, 16*3 + 4*3 + 2, -m / r * uph);

    return GSL_SUCCESS;
}

/* Calculates the tensor dRsigma2^a_{ b c} = R^a_{ d b e ;c} u^{d} u^{e} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int dR_sigma2 (const gsl_vector * y, const gsl_vector * yp, gsl_vector * dr_sigma2, void *params)
{
    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_vector_set_zero(dr_sigma2);

    gsl_vector_set(dr_sigma2, 16*0 + 4*0 + 0, -3.0 * m * (-uph * uph * pow(r, 3.0) - 2.0 * ut * ut * r + 4.0 * ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(dr_sigma2, 16*0 + 4*1 + 1, -3.0 * m * (-r + 2.0 * m) * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 * ut * ut * m) * pow(r, -4.0));
    gsl_vector_set(dr_sigma2, 16*0 + 4*2 + 0, -3.0 * m * pow(r, -2.0) * ur * uph);
    gsl_vector_set(dr_sigma2, 16*0 + 4*2 + 2, -3.0 * pow(-r + 2.0 * m, 2.0) * pow(r, -4.0) * m * ut * ut);
    gsl_vector_set(dr_sigma2, 16*0 + 4*3 + 0, 6.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut * ur);
    gsl_vector_set(dr_sigma2, 16*0 + 4*3 + 2, 3.0 * pow(-r + 2.0 * m, 2.0) * pow(r, -4.0) * m * ut * uph);
    gsl_vector_set(dr_sigma2, 16*1 + 4*0 + 1, 3.0 * m * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 * ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(dr_sigma2, 16*1 + 4*1 + 0, 3.0 * m * (-ur * ur * r * r + 2.0 * uph * uph * pow(r, 4.0) - 4.0 * m * uph * uph * pow(r, 3.0) + ut * ut * r * r - 4.0 * ut * ut * m * r + 4.0 * ut * ut * m * m) * pow(r, -5.0) / (-r + 2.0 * m));
    gsl_vector_set(dr_sigma2, 16*1 + 4*1 + 2, -6.0 * m * pow(r, -2.0) * ur * uph);
    gsl_vector_set(dr_sigma2, 16*1 + 4*2 + 1, 3.0 * m * pow(r, -2.0) * ur * uph);
    gsl_vector_set(dr_sigma2, 16*1 + 4*3 + 1, -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut * ur);
    gsl_vector_set(dr_sigma2, 16*2 + 4*0 + 0, 3.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ur * uph);
    gsl_vector_set(dr_sigma2, 16*2 + 4*0 + 2, 3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut * ut);
    gsl_vector_set(dr_sigma2, 16*2 + 4*1 + 1, 3.0 * m * pow(r, -2.0) * ur * uph);
    gsl_vector_set(dr_sigma2, 16*2 + 4*2 + 0, 3.0 * m * (-ur * ur * r * r + ut * ut * r * r - 4.0 * ut * ut * m * r + 4.0 * ut * ut * m * m) * pow(r, -5.0) / (-r + 2.0 * m));
    gsl_vector_set(dr_sigma2, 16*2 + 4*3 + 0, -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(dr_sigma2, 16*2 + 4*3 + 2, -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut * ur);
    gsl_vector_set(dr_sigma2, 16*3 + 4*0 + 0, -6.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ur * ut);
    gsl_vector_set(dr_sigma2, 16*3 + 4*0 + 2, -3.0 * m * pow(r, -2.0) * uph * ut);
    gsl_vector_set(dr_sigma2, 16*3 + 4*1 + 1, -3.0 * m * pow(r, -2.0) * ur * ut);
    gsl_vector_set(dr_sigma2, 16*3 + 4*2 + 0, -3.0 * m * pow(r, -2.0) * uph * ut);
    gsl_vector_set(dr_sigma2, 16*3 + 4*2 + 2, -3.0 * m * pow(r, -2.0) * ur * ut);
    gsl_vector_set(dr_sigma2, 16*3 + 4*3 + 0, 3.0 * m * (2.0 * ur * ur - uph * uph * r * r + 2.0 * uph * uph * r * m) * pow(r, -3.0) / (-r + 2.0 * m));
    gsl_vector_set(dr_sigma2, 16*3 + 4*3 + 2, 6.0 * m * pow(r, -2.0) * ur * uph);

    return GSL_SUCCESS;
}

/* Calculates the tensor dRsigma^a_{ b c} = R^a_{ b e c;d} u^{e} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int dR_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_vector * dr_sigma, void *params)
{
    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_vector_set_zero(dr_sigma);

    gsl_vector_set(dr_sigma, 64*0 + 16*1 + 4*1 + 0,  3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*0 + 16*1 + 4*1 + 2,  3.0 * m / r * (-r + 2.0 * m) * uph);
    gsl_vector_set(dr_sigma, 64*0 + 16*2 + 4*0 + 0,  -3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*0 + 16*2 + 4*1 + 1,  -3.0 * m / r * (-r + 2.0 * m) * uph);
    gsl_vector_set(dr_sigma, 64*0 + 16*2 + 4*2 + 0,  3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*0 + 16*3 + 4*0 + 0,  6.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*0 + 16*3 + 4*1 + 1,  3.0 * pow(-r + 2.0 * m, 2.0) * pow(r, -4.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*0 + 16*3 + 4*2 + 2,  3.0 * pow(-r + 2.0 * m, 2.0) * pow(r, -4.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*0 + 16*3 + 4*3 + 0,  -6.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ur);
    gsl_vector_set(dr_sigma, 64*0 + 16*3 + 4*3 + 2,  -3.0 * pow(-r + 2.0 * m, 2.0) * pow(r, -4.0) * m * uph);
    gsl_vector_set(dr_sigma, 64*1 + 16*0 + 4*1 + 0,  3.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ur);
    gsl_vector_set(dr_sigma, 64*1 + 16*0 + 4*1 + 2,  3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*1 + 16*2 + 4*0 + 1,  3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*1 + 16*2 + 4*1 + 0,  6.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*1 + 16*2 + 4*1 + 2,  3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*1 + 16*2 + 4*2 + 1,  -3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*1 + 16*3 + 4*0 + 1,  -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*1 + 16*3 + 4*1 + 0,  -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*1 + 16*3 + 4*3 + 1,  3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ur);
    gsl_vector_set(dr_sigma, 64*2 + 16*0 + 4*0 + 0,  -3.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * uph);
    gsl_vector_set(dr_sigma, 64*2 + 16*0 + 4*1 + 1,  -3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*2 + 16*0 + 4*2 + 0,  3.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ur);
    gsl_vector_set(dr_sigma, 64*2 + 16*1 + 4*0 + 1,  -3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*2 + 16*1 + 4*1 + 0,  -6.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*2 + 16*1 + 4*1 + 2,  -3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*2 + 16*1 + 4*2 + 1,  3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*2 + 16*3 + 4*0 + 2,  -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*2 + 16*3 + 4*2 + 0,  -3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ut);
    gsl_vector_set(dr_sigma, 64*2 + 16*3 + 4*3 + 0,  3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * uph);
    gsl_vector_set(dr_sigma, 64*2 + 16*3 + 4*3 + 2,  3.0 * (-r + 2.0 * m) * pow(r, -5.0) * m * ur);
    gsl_vector_set(dr_sigma, 64*3 + 16*0 + 4*0 + 0,  6.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*0 + 4*1 + 1,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*0 + 4*2 + 2,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*0 + 4*3 + 0,  -6.0 * pow(r, -3.0) / (-r + 2.0 * m) * m * ur);
    gsl_vector_set(dr_sigma, 64*3 + 16*0 + 4*3 + 2,  -3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*3 + 16*1 + 4*0 + 1,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*1 + 4*1 + 0,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*1 + 4*3 + 1,  -3.0 * m * pow(r, -2.0) * ur);
    gsl_vector_set(dr_sigma, 64*3 + 16*2 + 4*0 + 2,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*2 + 4*2 + 0,  3.0 * m * pow(r, -2.0) * ut);
    gsl_vector_set(dr_sigma, 64*3 + 16*2 + 4*3 + 0,  -3.0 * m * pow(r, -2.0) * uph);
    gsl_vector_set(dr_sigma, 64*3 + 16*2 + 4*3 + 2,  -3.0 * m * pow(r, -2.0) * ur);

    return GSL_SUCCESS;
}

/* Calculates the tensor d2Rsigma2^a_{ b c d} = R^a_{ e b f ;c d} u^{e} u^{f} and fill the values into r_sigma. Note that we have already
   set theta=Pi/2 and uth=0. */
int d2R_sigma2 (const gsl_vector * y, const gsl_vector * yp, gsl_vector * d2r_sigma2, void *params)
{
    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;
    double ur = gsl_vector_get(yp,0);
    double uph = gsl_vector_get(yp,3);
    double ut = gsl_vector_get(yp,4);
    double r = gsl_vector_get(y,0);

    /* Initialize all elements to 0 */
    gsl_vector_set_zero(d2r_sigma2);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*0 + 4*0 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * (-uph * uph * pow(r, 3.0) - 2.0 *ut * ut * r + 4.0 *ut * ut * m) * pow(r, -6.0) / (-r + 2.0 *m));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*0 + 4*1 + 1,  3.0 *m * (-r + 2.0 *m) * (-3.0 *uph * uph * pow(r, 3.0) - 4.0 *ut * ut * r + 8.0 *ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*0 + 4*2 + 2,  3.0 *m * (-r + 2.0 *m) * (-uph * uph * pow(r, 3.0) - 4.0 *ut * ut * r + 8.0 *ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*0 + 4*2 + 3,  6.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*0 + 4*3 + 3,  -3.0 *(-r + 2.0 *m) * m * m * (-uph * uph * pow(r, 3.0) - 2.0 *ut * ut * r + 4.0 *ut * ut * m) * pow(r, -8.0));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*1 + 4*0 + 1,  12.0 *m * (-r + 2.0 *m) * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 *ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*1 + 4*1 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 *ut * ut * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*0 + 16*1 + 4*1 + 2,  -3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*1 + 4*1 + 3,  -3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*1 + 4*2 + 1,  -3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*0 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*0 + 2,  12.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * ut);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*1 + 1,  9.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*2 + 0,  3.0 *m * (-r + 2.0 *m) * (-4.0 *r + 9.0 *m) * pow(r, -5.0) * ut * ut);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*2 + 2,  3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*2 + 3,  -3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*2 + 4*3 + 3,  -3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*0 + 0,  -6.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -6.0) * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*0 + 2,  -12.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*1 + 1,  -12.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*2 + 0,  -3.0 *m * (-r + 2.0 *m) * (-4.0 *r + 9.0 *m) * pow(r, -5.0) * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*2 + 2,  -12.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*2 + 3,  -3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * uph * ur);
    gsl_vector_set(d2r_sigma2, 64*0 + 16*3 + 4*3 + 3,  6.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -8.0) * m * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*0 + 4*0 + 1,  -12.0 *m * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 *ut * ut * m) * pow(r, -6.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*0 + 4*1 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * (-uph * uph * pow(r, 3.0) - ut * ut * r + 2.0 *ut * ut * m) * pow(r, -6.0) / (-r + 2.0 *m));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*0 + 4*1 + 2,  3.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*0 + 4*1 + 3,  3.0 *pow(r, -6.0) * m * m * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*0 + 4*2 + 1,  3.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*0 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * (-ur * ur * r * r + 2.0 *uph * uph * pow(r, 4.0) - 4.0 *m * uph * uph * pow(r, 3.0) + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -6.0) * pow((-r + 2.0 *m), -2.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*0 + 2,  24.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*1 + 1,  -3.0 *m * (-ur * ur * r * r + 4.0 *uph * uph * pow(r, 4.0) - 8.0 *m * uph * uph * pow(r, 3.0) + 3.0 *ut * ut * r * r - 12.0 *ut * ut * m * r + 12.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*2 + 0,  6.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*2 + 2,  -3.0 *m * (-3.0 *ur * ur * r * r + 4.0 *uph * uph * pow(r, 4.0) - 8.0 *m * uph * uph * pow(r, 3.0) + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*2 + 3,  -6.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*1 + 4*3 + 3,  3.0 *m * m * (-ur * ur * r * r + 2.0 *uph * uph * pow(r, 4.0) - 4.0 *m * uph * uph * pow(r, 3.0) + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -8.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*2 + 4*0 + 1,  -12.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*2 + 4*1 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*2 + 4*1 + 2,  -3.0 *m * (ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*2 + 4*1 + 3,  3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*2 + 4*2 + 1,  -3.0 *m * (ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*3 + 4*0 + 1,  12.0 *(-r + 2.0 *m) * pow(r, -6.0) * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*3 + 4*1 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -6.0) * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*3 + 4*1 + 2,  3.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*1 + 16*3 + 4*1 + 3,  -3.0 *m * m * (ur * ur - uph * uph * r * r + 2.0 *uph * uph * r * m) * pow(r, -6.0));
    gsl_vector_set(d2r_sigma2, 64*1 + 16*3 + 4*2 + 1,  3.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*0 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -4.0) * pow((-r + 2.0 *m), -2.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*0 + 2,  -12.0 *(-r + 2.0 *m) * pow(r, -6.0) * m * ut * ut);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*1 + 1,  -9.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*2 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -6.0) * ut * ut);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*2 + 2,  -3.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*2 + 3,  3.0 *pow(r, -6.0) * m * m * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*0 + 4*3 + 3,  3.0 *pow(r, -6.0) * m * m * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*1 + 4*0 + 1,  -12.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*1 + 4*1 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*1 + 4*1 + 2,  -3.0 *m * (ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*1 + 4*1 + 3,  3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*1 + 4*2 + 1,  -3.0 *m * (ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*2 + 4*0 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * (-ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -6.0) * pow((-r + 2.0 *m), -2.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*2 + 4*1 + 1,  -3.0 *m * (-3.0 *ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*2 + 4*2 + 2,  -3.0 *m * (-ur * ur * r * r + 3.0 *ut * ut * r * r - 12.0 *ut * ut * m * r + 12.0 *ut * ut * m * m) * pow(r, -5.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*2 + 4*3 + 3,  3.0 *m * m * (-ur * ur * r * r + ut * ut * r * r - 4.0 *ut * ut * m * r + 4.0 *ut * ut * m * m) * pow(r, -8.0));
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*0 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -6.0) * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*0 + 2,  12.0 *(-r + 2.0 *m) * pow(r, -6.0) * m * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*1 + 1,  3.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*2 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -6.0) * ut * ur);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*2 + 2,  9.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -5.0) * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*2 + 3,  -3.0 *pow(r, -6.0) * m * m * ur * ur);
    gsl_vector_set(d2r_sigma2, 64*2 + 16*3 + 4*3 + 3,  -3.0 *pow((-r + 2.0 *m), 2.0) * pow(r, -8.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*0 + 0,  6.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -4.0) * pow((-r + 2.0 *m), -2.0) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*0 + 2,  12.0 *m * pow(r, -3.0) * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*1 + 1,  12.0 *m * pow(r, -3.0) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*2 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*2 + 2,  12.0 *m * pow(r, -3.0) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*2 + 3,  3.0 *m * m * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*0 + 4*3 + 3,  -6.0 *pow(r, -6.0) * m * m * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*1 + 4*0 + 1,  12.0 *m * pow(r, -3.0) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*1 + 4*1 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*1 + 4*1 + 2,  3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*1 + 4*1 + 3,  -3.0 *m * m * (ur * ur - uph * uph * r * r + 2.0 *uph * uph * r * m) * pow(r, -3.0) / (-r + 2.0 *m));
    gsl_vector_set(d2r_sigma2, 64*3 + 16*1 + 4*2 + 1,  3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*0 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*0 + 2,  12.0 *m * pow(r, -3.0) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*1 + 1,  3.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*2 + 0,  3.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*2 + 2,  9.0 *(-r + 2.0 *m) * pow(r, -2.0) * m * uph * ut);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*2 + 3,  -3.0 *m * m * pow(r, -3.0) / (-r + 2.0 *m) * ur * ur);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*2 + 4*3 + 3,  -3.0 *(-r + 2.0 *m) * pow(r, -5.0) * m * m * ut * uph);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*0 + 0,  -3.0 *m * (-4.0 *r + 9.0 *m) * (2.0 *ur * ur - uph * uph * r * r + 2.0 *uph * uph * r * m) * pow(r, -4.0) * pow((-r + 2.0 *m), -2.0));
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*0 + 2,  -24.0 *m * pow(r, -3.0) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*1 + 1,  -3.0 *m * (4.0 *ur * ur - uph * uph * r * r + 2.0 *uph * uph * r * m) * pow(r, -3.0));
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*2 + 0,  -6.0 *m * (-4.0 *r + 9.0 *m) * pow(r, -3.0) / (-r + 2.0 *m) * ur * uph);
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*2 + 2,  -3.0 *m * (4.0 *ur * ur - 3.0 *uph * uph * r * r + 6.0 *uph * uph * r * m) * pow(r, -3.0));
    gsl_vector_set(d2r_sigma2, 64*3 + 16*3 + 4*3 + 3,  3.0 *m * m * (2.0 *ur * ur - uph * uph * r * r + 2.0 *uph * uph * r * m) * pow(r, -6.0));

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

/* The Riemann tensor R^{a}_{~ b c d} */
int Riemann(double * R, double r0, void * params)
{
    int r=0, th=1, ph=2, t=3;

    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;

    /* All expressions are related to one of these three */
    double r1 = m/(gsl_pow_2(r0)*(2*m-r0));
    double r2 = m/r0;
    double r3 = (2*m-r0)*m/gsl_pow_4(r0);

    R(r,th,r,th) 	= -r2;
    R(r,ph,r,ph) 	= -r2;
    R(r,t,r,t) 		= 2*r3;
    R(th,r,r,th) 	= -r1;
    R(th,ph,th,ph) 	= 2*r2;
    R(th,t,th,t) 	= -r3;
    R(ph,r,r,ph) 	= -r1;
    R(ph,th,th,ph) 	= -2*r2;
    R(ph,t,ph,t) 	= -r3;
    R(t,r,r,t) 		= 2*r1;
    R(t,th,th,t) 	= r2;
    R(t,ph,ph,t) 	= r2;
    R(r,th,th,r) 	= r2;
    R(r,ph,ph,r) 	= r2;
    R(r,t,t,r) 		= -2*r3;
    R(th,r,th,r) 	= r1;
    R(th,ph,ph,th) 	= -2*r2;
    R(th,t,t,th) 	= r3;
    R(ph,r,ph,r) 	= r1;
    R(ph,th,ph,th) 	= 2*r2;
    R(ph,t,t,ph) 	= r3;
    R(t,r,t,r) 		= -2*r1;
    R(t,th,t,th) 	= -r2;
    R(t,ph,t,ph) 	= -r2;

    return GSL_SUCCESS;
}

/* Riemann tensor symmetrized over second and 4th indices, RiemannSym^{a}_{~ b c d} = R^{a}_{~ (c |b| d)} */
int RiemannSym(double * R, double r0, void * params)
{
    int r=0, th=1, ph=2, t=3;

    struct geodesic_params p = *(struct geodesic_params *)params;
    double m = p.m;

    /* All expressions are related to one of these three */
    double r1 = m/(gsl_pow_2(r0)*(2*m-r0));
    double r2 = m/r0;
    double r3 = (2*m-r0)*m/gsl_pow_4(r0);

    R(th,th,r,r) 	= r1;
    R(ph,ph,r,r) 	= r1;
    R(t,t,r,r) 		= -2.0*r1;
    R(th,r,th,r) 	= -r1/2.0;
    R(r,th,th,r) 	= r2/2.0;
    R(ph,r,ph,r) 	= -r1/2.0;
    R(r,ph,ph,r) 	= r2/2.0;
    R(t,r,t,r) 		= r1;
    R(r,t,t,r) 		= -r3;
    R(th,r,r,th) 	= -r1/2.0;
    R(r,th,r,th) 	= r2/2.0;
    R(r,r,th,th) 	= -r2;
    R(ph,ph,th,th) 	= 2*r2;
    R(t,t,th,th) 	= -r2;
    R(ph,th,ph,th) 	= -r2;
    R(th,ph,ph,th) 	= -r2;
    R(t,th,t,th) 	= r2/2.0;
    R(th,t,t,th) 	= r3/2.0;
    R(ph,r,r,ph) 	= -r1/2.0;
    R(r,ph,r,ph) 	= r2/2.0;
    R(ph,th,th,ph) 	= -r2;
    R(th,ph,th,ph) 	= -r2;
    R(r,r,ph,ph) 	= -r2;
    R(th,th,ph,ph) 	= 2*r2;
    R(t,t,ph,ph) 	= -r2;
    R(t,ph,t,ph) 	= r2/2.0;
    R(ph,t,t,ph) 	= r3/2.0;
    R(t,r,r,t) 		= r1;
    R(r,t,r,t) 		= -r3;
    R(t,th,th,t) 	= r2/2.0;
    R(th,t,th,t) 	= r3/2.0;
    R(t,ph,ph,t) 	= r2/2.0;
    R(ph,t,ph,t) 	= r3/2.0;
    R(r,r,t,t) 		= 2.0*r3;
    R(th,th,t,t) 	= -r3;
    R(ph,ph,t,t) 	= -r3;

    return GSL_SUCCESS;
}

double RicciScalar()
{
    return 0;
}

/* Covariant derivative of Ricci scalar contracted with 4-velocity */
double d_RicciScalar (const gsl_vector * y, const gsl_vector * yp, void *params)
{

    return GSL_SUCCESS;
}

