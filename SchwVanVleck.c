/* Numerically integrate the geodesic equations in Schwarzschild */
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

#define NUM_EQS	(5+16+1+16+16+64)
#define EPS	10e-12

/* Parameters of the motion */
struct geodesic_params {
  double m; /* Black Hole Mass */
  double e; /* "Energy" constant of motion */
  double l; /* "Angular momentum" constant of motion */
  int type; /* Type of geodesic. 0=null, -1=time-like */
};

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

/* Calculates the tensor Rsigma^a_{ b c} = R^a_{ b d c} u^d and fill the values into r_sigma, which is an array of matrices.
   We use the convention that c is the index of the array and a and b are the indices of the matrices. Note that we have already
   set theat=Pi/2 and uth=0. */
int R_sigma (const gsl_vector * y, const gsl_vector * yp, gsl_matrix *r_sigma[], void *params)
{
  struct geodesic_params p = *(struct geodesic_params *)params;
  double m = p.m;
  double ur = gsl_vector_get(yp,0);
  double uph = gsl_vector_get(yp,3);
  double ut = gsl_vector_get(yp,4);
  double r = gsl_vector_get(y,0);
  int i;
  
  /* Initialize all elements to 0 */
  for(i=0; i<4; i++)
  {
    gsl_matrix_set_zero(r_sigma[i]);
  }
  
  /* Now, set the non-zero elements */
  gsl_matrix_set(r_sigma[1],0, 1, -m*ur/r);
  gsl_matrix_set(r_sigma[0],0, 2, m*uph/r);
  gsl_matrix_set(r_sigma[2],0, 2, -m*ur/r);
  gsl_matrix_set(r_sigma[0],0, 3, -(2*(-r+2*m))*m*ut/gsl_pow_4(r));
  gsl_matrix_set(r_sigma[3],0, 3, (2*(-r+2*m))*m*ur/gsl_pow_4(r));
  gsl_matrix_set(r_sigma[1],1, 0, -m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_matrix_set(r_sigma[1],1, 2, -2*m*uph/r);
  gsl_matrix_set(r_sigma[1],1, 3, (-r+2*m)*m*ut/gsl_pow_4(r));
  gsl_matrix_set(r_sigma[0],2, 0, m*uph/(gsl_pow_2(r)*(-r+2*m)));
  gsl_matrix_set(r_sigma[2],2, 0, -m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_matrix_set(r_sigma[1],2, 1, 2*m*uph/r);
  gsl_matrix_set(r_sigma[2],2, 3, (-r+2*m)*m*ut/gsl_pow_4(r));
  gsl_matrix_set(r_sigma[3],2, 3, -(-r+2*m)*m*uph/gsl_pow_4(r));
  gsl_matrix_set(r_sigma[0],3, 0, -2*m*ut/(gsl_pow_2(r)*(-r+2*m)));
  gsl_matrix_set(r_sigma[3],3, 0, 2*m*ur/(gsl_pow_2(r)*(-r+2*m)));
  gsl_matrix_set(r_sigma[1],3, 1, -m*ut/r);
  gsl_matrix_set(r_sigma[2],3, 2, -m*ut/r);
  gsl_matrix_set(r_sigma[3],3, 2, m*uph/r);
  
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
  gsl_matrix_sub(f, q2);
  gsl_matrix_sub(f,tau_S);
  
  /* The theta,theta component blows up as theta*cot(theta) and makes the numerical scheme break down.
     Since we know the analytic form, don't compute it numerically */
  gsl_matrix_set(f,1,1,0.0);
  
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
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, dI2[i], gu, 1.0, f2[i]);
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
  double y[NUM_EQS] = { 
    10.0, 0.0, 0.0, 0.0, 0.0, /* r, r', theta, phi, t */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* Q^a'_b' */
    1.0, /* Delta^1/2 */
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, /* I^a_b' */
   -1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0,-1.0, /* eta^a_b' */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /* dI */
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
    
    /* Output the results */
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, ", tau, y[0], y[1], y[2], y[3], y[4]);
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, ", y[5], y[6], y[7], y[8], y[9], y[10]);
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, ", y[11], y[12], y[13], y[14], y[15], y[16]);
    printf ("%.5f, %.5f, %.5f, %.5f, %.5f, ", y[17], y[18], y[19], y[20], y[21]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[22], y[23], y[24], y[25]); /* I^a_b' */
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[26], y[27], y[28], y[29]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[30], y[31], y[32], y[33]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[34], y[35], y[36], y[37]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[38], y[39], y[40], y[41]); /* eta^a_b' */
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[42], y[43], y[44], y[45]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[46], y[47], y[48], y[49]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[50], y[51], y[52], y[53]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", gsl_matrix_get(gamma,0,0), gsl_matrix_get(gamma,0,1), gsl_matrix_get(gamma,0,2), gsl_matrix_get(gamma,0,3)); 
    printf ("%.5f, %.5f, %.5f, %.5f, ", gsl_matrix_get(gamma,1,0), gsl_matrix_get(gamma,1,1), gsl_matrix_get(gamma,1,2), gsl_matrix_get(gamma,1,3));
    printf ("%.5f, %.5f, %.5f, %.5f, ", gsl_matrix_get(gamma,2,0), gsl_matrix_get(gamma,2,1), gsl_matrix_get(gamma,2,2), gsl_matrix_get(gamma,2,3));
    printf ("%.5f, %.5f, %.5f, %.5f, ", gsl_matrix_get(gamma,3,0), gsl_matrix_get(gamma,3,1), gsl_matrix_get(gamma,3,2), gsl_matrix_get(gamma,3,3));
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[54], y[55], y[56], y[57]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[58], y[39], y[60], y[61]); /* eta^a_b' */
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[62], y[63], y[64], y[65]);
    printf ("%.5f, %.5f, %.5f, %.5f, ", y[66], y[67], y[68], y[69]);
    printf ("%.5f, %.5f, %.5f, %.5f\n", y[70], y[71], y[72], y[73]);
    
    /* Don't let the step size get bigger than 1 */
    /*if (h > .10)
    {
      fprintf(stderr,"Warning: step size %e greater than 1 is not allowed. Using step size of 1.0.\n",h);
      h=.10;
    }*/
      
    /* Exit if step size get smaller than 10^-12 */
    if (h < 1e-12)
    {
      fprintf(stderr,"Error: step size %e less than 1e-12 is not allowed.\n",h);
      break;
    }
  }

  gsl_odeiv_evolve_free (e);
  gsl_odeiv_control_free (c);
  gsl_odeiv_step_free (s);
  return 0;
}
