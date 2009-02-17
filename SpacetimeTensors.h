/* For any spacetime, we need to specify the tensors in this file.
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

struct geodesic_params {
  double m; /* Black Hole Mass */
  double e; /* "Energy" constant of motion */
  double l; /* "Angular momentum" constant of motion */
  int type; /* Type of geodesic. 0=null, -1=time-like */
};

/* The contravariant metric components */
int metric_up_up(const double *y, gsl_matrix *metric, void *params);

/* The matrix S^a_b = R^a_{ c b d} u^c u^d */
int S (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *s, void *params);

/* The tensor Rsigma^a_{ b c} = R^a_{ b d c} u^d */
int R_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *r_sigma[], void *params);

/* The matrix Gu^a_b = \Gamma^a_{b c} u^c */
int Gu (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *gu, void *params);

/* RHS of geodesic equations */
int geodesicRHS (double tau, const gsl_vector * y, gsl_vector * f, void *params);