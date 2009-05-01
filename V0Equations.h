/* Numerically solve the transport equation for V_0
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
#include <gsl/gsl_matrix.h>

int IRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, gsl_matrix * f, void * params);
int etaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, const gsl_matrix * eta, gsl_matrix * f, void * params);
int dIRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, const gsl_matrix *q, const gsl_matrix * dI[], gsl_matrix * f[], void * params);
int dxiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], gsl_matrix * f[], void * params);
int detaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * eta, const gsl_matrix * deta[], gsl_matrix * f[], void * params);
int d2I_Inv (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * I, const gsl_matrix * dI[], const double * d2I_Inv, double * f, void * params);
int d2xi (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_matrix * dxi[], const gsl_matrix * I, const gsl_matrix * dI[], const double * d2xi, double * f, void * params);
int d2eta (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *eta, const gsl_matrix *q, const gsl_matrix * dxi[], const double * d2xi, const gsl_matrix * deta[], const double * d2eta, double * f, void * params);
int gammaBitensor ( const gsl_matrix * eta, gsl_matrix * gamma );
int boxSqrtDelta (double tau, const double * y, double * f, void * params);
int V0RHS (double tau, const gsl_matrix * q, const double * dal_sqrt_delta, const double * v0, double * f, void * params);
