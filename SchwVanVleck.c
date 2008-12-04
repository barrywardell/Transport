/* Numerically integrate the geodesic equations in Schwarzschild */
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>

#define NUM_EQS	21
#define EPS	10e-12
/* Parameters of the motion */
struct geodesic_params {
  double m; /* Black Hole Mass */
  double e; /* "Energy" constant of motion */
  double l; /* "Angular momentum" constant of motion */
  int type; /* Type of geodesic. 0=null, -1=time-like */
};

/* Calculates the matrix S^a_b = R^a_{ c b d} u^c u^d and fill the values into s */
int S (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *s, void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double ur = gsl_vector_get(yp,0);
  double uth = gsl_vector_get(yp,2);
  double up = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  
  gsl_matrix_set(s,0,0,(m*(4*m*gsl_pow_int(ut,2) - 2*r*gsl_pow_int(ut,2) - gsl_pow_int(r,3)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/gsl_pow_int(r,4));
  gsl_matrix_set(s,0,1,(m*ur*uth)/r);
  gsl_matrix_set(s,0,2,(m*up*ur)/r);
  gsl_matrix_set(s,0,3,(2*m*(-2*m + r)*ur*ut)/gsl_pow_int(r,4));
  gsl_matrix_set(s,1,0,-((m*ur*uth)/((2*m - r)*gsl_pow_int(r,2))));
  gsl_matrix_set(s,1,1,(m*gsl_pow_int(r,2)*(2*(2*m - r)*r*gsl_pow_int(up,2) + gsl_pow_int(ur,2)) - m*gsl_pow_int(-2*m + r,2)*gsl_pow_int(ut,2))/((2*m - r)*gsl_pow_int(r,4)));
  gsl_matrix_set(s,1,2,(-2*m*up*uth)/r);
  gsl_matrix_set(s,1,3,(m*(2*m - r)*ut*uth)/gsl_pow_int(r,4));
  gsl_matrix_set(s,2,0,-((m*up*ur)/((2*m - r)*gsl_pow_int(r,2))));
  gsl_matrix_set(s,2,1,(-2*m*up*uth)/r);
  gsl_matrix_set(s,2,2,(m*(-4*gsl_pow_int(m,2)*gsl_pow_int(ut,2) + r*(4*m*gsl_pow_int(ut,2) + r*(gsl_pow_int(ur,2) - gsl_pow_int(ut,2) - 2*r*(-2*m + r)*gsl_pow_int(uth,2)))))/((2*m - r)*gsl_pow_int(r,4)));
  gsl_matrix_set(s,2,3,(m*(2*m - r)*up*ut)/gsl_pow_int(r,4));
  gsl_matrix_set(s,3,0,(2*m*ur*ut)/((2*m - r)*gsl_pow_int(r,2)));
  gsl_matrix_set(s,3,1,(m*ut*uth)/r);
  gsl_matrix_set(s,3,2,(m*up*ut)/r);
  gsl_matrix_set(s,3,3,(m*(-2*gsl_pow_int(ur,2) + r*(-2*m + r)*(gsl_pow_int(up,2) + gsl_pow_int(uth,2))))/ ((2*m - r)*gsl_pow_int(r,2)));
  
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
  gsl_matrix_add(f, q2);
  gsl_matrix_add(f,tau_S);
  
  return GSL_SUCCESS;
}

/* RHS of our system of ODEs */
int func (double tau, const double y[], double f[], void *params)
{
  /* Geodesic equations: 5 coupled equations for r,r',theta,phi,t */
  gsl_vector_view geodesic_eqs = gsl_vector_view_array(f,5);
  gsl_vector_const_view geodesic_coords = gsl_vector_const_view_array(y,5);
  geodesicRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, params);
  
  /* Equations for Q^a_b */
  gsl_matrix_view q_eqs = gsl_matrix_view_array(f+5,4,4);
  gsl_matrix_const_view q_vals = gsl_matrix_const_view_array(y+5,4,4);
  qRHS(tau, &geodesic_coords.vector, &geodesic_eqs.vector, &q_vals.matrix, &q_eqs.matrix, params);
    
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
  /* Use a Burlisch-Stoer integrator with adaptive step-size */
  const gsl_odeiv_step_type * T = gsl_odeiv_step_rkf45;
  gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, NUM_EQS);
  gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-8, 1e-8);
  gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (NUM_EQS);

  /* Time-like geodesic starting at r=10M and going in to r=4M */
  struct geodesic_params params = {1,0.950382,3.59211,-1};
  
  gsl_odeiv_system sys = {func, jac, NUM_EQS, &params};

  double tau = 0.0, tau1 = 1000.0;
  double h = 1e-6;
  double y[NUM_EQS] = { 10.0, 0.0, 0.0, 0.0, 0.0, /* r, r', theta, phi, t */
		  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; /* Q^a_b */

  while (tau < tau1)
  {
    int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &tau, tau1, &h, y);

    if (status != GSL_SUCCESS)
      break;
    
    /* Don't let the step size get bigger than 1 */
    if (h > 1.0)
    {
      //fprintf(stderr,"Warning: step size %e greater than 1 is not allowed.\n",h);
      h=1.0;
    }
      
    /* Don't let the step size get smaller than 10^-6 */
    if (h < 1e-12)
    {
      fprintf(stderr,"Warning: step size %e less than 1e-12 is not allowed.\n",h);
      h=1e-12;
    }
    
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, ", tau, y[0], y[1], y[2], y[3], y[4]);
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f", y[5], y[6], y[7], y[8], y[9], y[10]);
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f", y[11], y[12], y[13], y[14], y[15], y[16]);
    printf ("%.5f, %.5f, %.5f, %.5f\n", y[17], y[18], y[19], y[20]);
  }

  gsl_odeiv_evolve_free (e);
  gsl_odeiv_control_free (c);
  gsl_odeiv_step_free (s);
  return 0;
}
