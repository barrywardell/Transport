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
int dIinvRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * I, const gsl_matrix *q, const gsl_vector * dIinv, gsl_vector * f, void * params);
int dxiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, gsl_vector * f, void * params);
int detaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * eta, const gsl_vector * deta, gsl_vector * f, void * params);
int d2IinvRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * I, const gsl_vector * dIinv, const gsl_vector * d2I_Inv, gsl_vector * f, void * params);
int d2xiRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *q, const gsl_vector * dxi, const gsl_matrix * I, const gsl_vector * dIinv, const gsl_vector * d2xi, gsl_vector * f, void * params);
int d2etaRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix *eta, const gsl_matrix *q, const gsl_vector * dxi, const gsl_vector * d2xi, const gsl_vector * deta, const gsl_vector * d2eta, gsl_vector * f, void * params);
int gammaBitensor ( const gsl_matrix * eta, gsl_matrix * gamma );
int boxSqrtDelta (double tau, const double * y, double * f, void * params);
int tr2term (double tau, const double * y, double * f, void * params);
int V0RHS (double tau, const gsl_matrix * q, const double * dal_sqrt_delta, const double * v0, double * f, void * params);

/* Initial values */
int d2IinvInit(double * d2Iinv, double r0, void * params);
int d2xiInit(double * d2xi, double r0, void * params);
int d2etaInit(double * d2eta, double r0, void * params);
