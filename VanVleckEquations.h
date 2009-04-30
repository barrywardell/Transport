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
#include <gsl/gsl_matrix.h>

int qRHS (double tau, const gsl_vector * y, const gsl_vector * yp, const gsl_matrix * q, gsl_matrix * f, void * params);
int sqrtDeltaRHS (double tau, const gsl_matrix * q, const double * sqrt_delta, double * f, void * params);
