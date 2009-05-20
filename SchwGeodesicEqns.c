/* Numerically integrate the geodesic equations of Schwarzschild spacetime.
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
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_math.h>

/* Parameters of the motion */
struct geodesic_params {
    double m; /* Black Hole Mass */
    double e; /* "Energy" constant of motion */
    double l; /* "Angular momentum" constant of motion */
    int type; /* Type of geodesic. 0=null, -1=time-like */
};

/* RHS of geodesic equations */
int geodesicRHS (double tau, const gsl_vector * y, gsl_vector * rhs, void *params)
{
    struct geodesic_params p = *(struct geodesic_params *)params;

    double r = gsl_vector_get(y,0);
    double rp = gsl_vector_get(y,1);

    gsl_vector_set(rhs,0,rp);
    gsl_vector_set(rhs,1,(gsl_pow_2(p.l)*(r-3*p.m)+p.m*gsl_pow_2(r)*p.type)/gsl_pow_4(r));
    gsl_vector_set(rhs,2,0.0);
    gsl_vector_set(rhs,3,p.l/gsl_pow_2(r));
    gsl_vector_set(rhs,4,r/(r-2*p.m)*p.e);

    return GSL_SUCCESS;
}

/* RHS of our system of ODEs */
int func (double tau, const double y[], double f[], void *params)
{
    /* Geodesic equations: 5 coupled equations for r,r',theta,phi,t */
    gsl_vector_view geodesic_eqs = gsl_vector_view_array(f,5);
    gsl_vector_const_view geodesic_coords = gsl_vector_const_view_array(y,5);
    geodesicRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, params);

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

int main (int argc, const char* argv[])
{
    /* Use Burlisch-Stoer method */
    const gsl_odeiv_step_type * T = gsl_odeiv_step_bsimp;

    gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, 5);
    gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-8, 1e-8);
    gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (5);

    struct geodesic_params params = {1,0.950382,3.59211,-1};
    gsl_odeiv_system sys = {func, jac, 5, &params};

    double tau = 0.0, tau1 = 1000.0;
    double h = 1e-6;
    double y[5] = { 10.0, 0.0, 0.0, 0.0, 0.0 };

    while (tau < tau1)
    {
        int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &tau, tau1, &h, y);

        if (status != GSL_SUCCESS)
            break;

        if (h > 1.0)
            h=1.0;

        printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", tau, y[0], y[1], y[2], y[3], y[4]);
    }

    gsl_odeiv_evolve_free (e);
    gsl_odeiv_control_free (c);
    gsl_odeiv_step_free (s);

    return 0;
}
