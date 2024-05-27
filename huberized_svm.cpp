#include "ssvm.h"
#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::interfaces(r,cpp)]]

// [[Rcpp::plugins("cpp11")]]


// [[Rcpp::export]]
Eigen::VectorXd huberized_svm_dense(const Eigen::MatrixXd& X, const Eigen::ArrayXd& y,
                                    double alpha = 0.001,
                                    double beta = 0.0,
                                    double gamma = 0.0, 
                                    double tol = 1e-5, 
                                    int max_iter = 2000, 
                                    int chk_fre = 1){

  
  return huberized_svm(X, y, alpha, beta, gamma, tol, max_iter, chk_fre);
}


// [[Rcpp::export]]
Eigen::VectorXd huberized_svm_sparse(const SpMatRd& X, const Eigen::ArrayXd& y,
                                     double alpha = 0.001,
                                     double beta = 0.0,
                                     double gamma = 0.0,
                                     double tol = 1e-5,
                                     int max_iter = 2000,
                                     int chk_fre = 1){

  return huberized_svm(X, y, alpha, beta, gamma, tol, max_iter, chk_fre);
}

