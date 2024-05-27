// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::interfaces(r,cpp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <iterator>
#include <chrono>
// [[Rcpp::plugins("cpp11")]]

using namespace Rcpp;
using namespace std;

using SpMatRd = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpMatCd = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using scm_iit = Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator;
using srm_iit = Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator;

using sec = std::chrono::seconds;
using mil_sec = std::chrono::milliseconds;
using sys_clk = std::chrono::system_clock;


Eigen::VectorXi min_k(Eigen::VectorXd& vec, int k, bool sort_by_value=false ) {
  Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);  // [0 1 2 3 ... N-1]
  auto rule = [vec](int i, int j) -> bool { return vec(i) < vec(j); };              // sort rule
  std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
  if (sort_by_value) {
    std::sort(ind.data(), ind.data() + k, rule);
  } else {
    std::sort(ind.data(), ind.data() + k);
  }
  return ind.head(k).eval();
}//

Eigen::VectorXi max_k(Eigen::VectorXd& vec, int k, bool sort_by_value = false) {
  Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);  // [0 1 2 3 ... N-1]
  
  auto rule = [vec](int i, int j) -> bool { return vec(i) > vec(j); };              // sort rule
  
  std::nth_element(ind.data(), ind.data() + k, ind.data() + ind.size(), rule);
  
  if (sort_by_value) {
    std::sort(ind.data(), ind.data() + k, rule);
  } else {
    std::sort(ind.data(), ind.data() + k);
  }
  return ind.head(k).eval();
}//

Eigen::VectorXi diff_union(Eigen::VectorXi A, Eigen::VectorXi &B, Eigen::VectorXi &C) {
  unsigned int k;
  for (unsigned int i = 0; i < B.size(); i++) {
    for (k = 0; k < A.size(); k++) {
      if (B(i) == A(k)) {
        A(k) = C(i);
        break;
      }
    }
  }
  sort(A.data(), A.data() + A.size());
  return A;
}

Eigen::VectorXi vector_slice(Eigen::VectorXi nums, Eigen::VectorXi ind) {
  Eigen::VectorXi sub_nums(ind.size());
  if (ind.size() != 0) {
    for (int i = 0; i < ind.size(); i++) {
      sub_nums(i) = nums(ind(i));
    }
  }
  return sub_nums;
}//

Eigen::VectorXi complement(Eigen::VectorXi A, int N) {
  int A_size = A.size();
  if (A_size == 0) {
    return Eigen::VectorXi::LinSpaced(N, 0, N - 1);
  } else if (A_size == N) {
    Eigen::VectorXi I(0);
    return I;
  } else {
    Eigen::VectorXi I(N - A_size);
    int cur_index = 0;
    int A_index = 0;
    for (int i = 0; i < N; i++) {
      if (A_index >= A_size) {
        I(cur_index) = i;
        cur_index += 1;
        continue;
      }
      if (i != A(A_index)) {
        I(cur_index) = i;
        cur_index += 1;
      } else {
        A_index += 1;
      }
    }
    return I;
  }
}

// Eigen::VectorXd Hard(const Eigen::VectorXd& beta, int s) {
//   Eigen::VectorXd abs_beta = beta.cwiseAbs();
//   // Partially sort the absolute values in decreasing order
//   std::nth_element(abs_beta.data(), abs_beta.data() + s - 1, abs_beta.data() + abs_beta.size(), std::greater<double>());
//   
//   // Compute the result
//   Eigen::VectorXd result = (beta.cwiseAbs().array() >= abs_beta[s-1]).select(beta, 0.0);
//   return result;
// }

// [[Rcpp::export]]
Eigen::VectorXd Soft(const Eigen::VectorXd& w, const double& lambda) {
  // Compute the result
  Eigen::VectorXd result = w;
  for(size_t i=0; i < w.size(); i++){
    if(w(i) > lambda)
      result(i) -= lambda;
    else if(w(i)>0)
      result(i) = 0;
  }
  
  return result;
}

// [[Rcpp::export]]
double logC(const int n, const int m){
  double ans = 0;
  for(int i =1;i<=m;i++){
    ans = ans + log(n-m+i)-log(i); //
  }
  return ans;
}


template<class T1>
Eigen::VectorXd bess_lm(const T1& X, const Eigen::VectorXd& y,
                        const Eigen::ArrayXd& Xj_norm, const int T0,
                        const double lambda, const int max_steps=100) {
  int n = X.rows(), p = X.cols();
  // T1 X_CM = X;
  
  double max_T = 0.0;
  
  vector<int> E(p) ;
  std::iota(E.begin(), E.end(), 0);
  
  Eigen::VectorXi I(p - T0), A(T0), J(p - T0), B(T0);
  
  Eigen::MatrixXd X_A = T1(n, T0);
  T1 X_I = T1(n, p - T0);
  
  Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(T0);
  Eigen::VectorXd d_I = Eigen::VectorXd::Zero(p - T0);
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
  
  Eigen::VectorXd beta = beta0;
  
  Eigen::VectorXd d = (X.transpose() * (y - X * beta)).array() / (Xj_norm + 2*n*lambda);
  
  
  Eigen::VectorXd bd = (beta+d).array().square() * (Xj_norm/n + 2*lambda);
  
  A = max_k(bd, T0);
  
  std::set_difference(E.begin(), E.end(), A.begin(), A.end(), I.begin());
  
  for(int l = 1; l <= max_steps; l++) {
    
    for(int mm = 0; mm <= T0 - 1; mm++)
      X_A.col(mm) = X.col(A[mm]);
    
    
    beta_A =  ((X_A.transpose() * X_A) + 2*lambda*n*Eigen::MatrixXd::Identity(T0,T0)).llt().solve(X_A.transpose()*y);
    
    d(A) = Eigen::VectorXd::Zero(T0);
    beta(A) = beta_A;
    
    //   d_I = (X_I.transpose() * (y - X_A * beta_A)).array() / (Xj_norm(I) + 2*n*lambda);
    d_I = (X.transpose() * (y - X_A * beta_A))(I).array() / (Xj_norm(I) + 2*n*lambda);
    
    
    beta(I) = Eigen::VectorXd::Zero(p - T0);
    d(I) = d_I;
    // bd = abs(beta + d);
    bd = (beta+d).array().square() * (Xj_norm/n + 2*lambda);
    
    B = max_k(bd, T0);
    // std::cout<<"l: "<<l<<"B: "<<B<<std::endl;
    
    std::set_difference(E.begin(), E.end(), B.begin(), B.end(), J.begin());
    if((A == B)) { break; } else {
      A = B;
      I = J;
    }
  }
  
  return beta;
}


template<class T1>
List svm_admm_l0(const T1& X, const Eigen::VectorXd& y, const int s, Eigen::VectorXd w0,
                 const double lambda_2 = 0.001, double rho = 1,
                 const double tol = 1e-4, const int maxit = 500, const bool rho_change=true) {
  
  int n = X.rows(), p = X.cols();
  
  // constant
  T1 X_ = X;
  for (size_t i = 0; i < n; ++i){
    X_.row(i) *= y[i];
  }
  
  Eigen::ArrayXd Xj_norm(p);
  for (int j = 0; j < p; ++j) {
    Xj_norm[j] = X.col(j).squaredNorm();
  }
  // // Xj_norm = Xj_norm.square();
  
  
  Eigen::VectorXd w(p), old_w(p), Xw(n);
  Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
  if(w0.size() < p){
    w0.resize(p) ;
    w0 = Eigen::VectorXd::Zero(p);
  }
  w = w0;
  
  Eigen::VectorXd old_A(n), A =  1.0 - (X_*w).array();
  // Xw = A;
  Xw = A;
  
  //TODO obj
  double obj_error, old_obj,
  obj = Xw.cwiseMax(0).sum()/n  + lambda_2/2 * w.squaredNorm();
  
  double threashold1, lambda_ ;
  // Eigen::SparseMatrix::Iden
  
  T1 Ip(p,p);
  Ip.setIdentity();
  
  // Eigen::MatrixXd Ip = Eigen::MatrixXd::Identity(p,p);
  
  std::vector<double> hist_obj;
  int iter = 0;
  
  while (iter < maxit){
    iter += 1;
    threashold1 = 1.0/(n*rho); lambda_ = lambda_2/(2*rho*n);
    
    old_w = w; old_A = A; old_obj = obj;
    
    // update w
    
    w = bess_lm(X_, 1.0 - (A - u).array(), Xj_norm, s, lambda_);
    
    Xw = (1.0 - (X_*w).array());
    
    //update A
    A = Soft(Xw + u, threashold1);
    
    //compute primal residual and save to hist_pres
    obj = Xw.cwiseMax(0).sum()/n + lambda_2/2 * w.squaredNorm();
    
    hist_obj.push_back(obj);
    
    obj_error = abs(obj-old_obj) / old_obj;
    
    double res_p = (Xw-A).norm()/sqrt(n);
    double res_d = rho * (X_.transpose()*(A - old_A)).norm();
    if(obj_error <= tol && res_p <= tol) break;
    
    // cout<<"iter:"<<iter<< " obj:"<<obj<<" res_p:"<<res_p<<" res_d:"<<res_d<<endl;
    
    // update Lagrangian multiplier
    u = u + (Xw - A);
    
    // change rho
    if(rho_change){
      if( 10 * res_d < res_p) rho = rho * 2;
      else if(res_d > 10*res_p) rho = rho/2;
    }
    
  }
  
  return List::create(Named("w")=w, Named("iter")=iter, Named("hist_obj")=hist_obj);
}



template<class T1>
List Adsvm_admm_l0(const T1& X, const Eigen::VectorXd& y, int s_max , Eigen::VectorXd beta_init,
                   bool warm_start = true, int max_iter = 200){
  int n = X.rows(), p = X.cols();
  Eigen::VectorXd beta, Xw;
  
  T1 X_ = X;
  for (int i = 0; i < n; ++i){
    X_.row(i) *= y[i];
  }
  
  int s_Max = min(p, int(n/(log(p))));
  if(s_max == -1) s_max = min(n, s_Max);
  s_max = min(s_Max, s_max);
  
  cout<<"s_max:" <<s_max<<endl;
  int size;
  double L;
  double L1 = log(log(n)), L2 = sqrt(log(n)), L3 = log(n), L4 = pow(n, -1.0/3.0);
  Eigen::VectorXd ebic(s_max), sic(s_max), sic1(s_max), sic2(s_max), sic3(s_max), sic4(s_max);
  
  std::vector<double> supp(s_max), HingeLoss(s_max);
  
  List result;
  Eigen::SparseVector<double> betas;
  
  // Eigen::ArrayXd Xw;
  if(beta_init.size() < p){
    beta_init.resize(p);
    // beta_init = Eigen::VectorXd::Zero(p);
    beta_init.setZero();
  }
  
  for(int s = 0; s < s_max; s++){
    std::cout<<s<<"start"<<endl;
    if(warm_start){
      if(s>0)
        result = svm_admm_l0(X, y, s+1, beta);
      else
        result = svm_admm_l0(X, y, s+1, beta_init);
    }else result = svm_admm_l0(X, y, s+1, beta_init);
    
    beta = result["w"];
    // betas = beta.sparseView();
    
    size = (beta.array() != 0.0).count();
    
    // cout<<"size:"<<betas.nonZeros()<<endl;
    
    Xw = 1 - (X_ * beta).array();
    L = Xw.cwiseMax(0.0).sum();
    
    
    HingeLoss[s] = L;
    supp[s] = size;
    ebic[s] = L  + size * log(n) + logC(p, size);
    sic[s] = L + size * log(n);
    sic1[s] = L + size * log(n) * L1;
    sic2[s] = L + size * log(n) * L2;
    sic3[s] = L + size * log(n) * L3;
    sic4[s] = L + size * log(n) * L4;
    
  }
  
  return List::create(Named("ebic")=ebic, Named("sic")=sic, Named("sic1")=sic1,
                            Named("sic2")=sic2, Named("sic3")=sic3, Named("sic4")=sic4,
                                  Named("HingeLoss") = HingeLoss, Named("supp") = supp);
  
}


// [[Rcpp::export]]
List svm_admm_l0_dense(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const int s, Eigen::VectorXd w0,
                       const double lambda_2 = 0.001, double rho = 1,
                       const double tol = 1e-4, const int maxit = 500, const bool rho_change=true){
  // cout << "X has " << X.nonZeros() << X.coeffs().transpose() << endl;
  return svm_admm_l0(X, y, s, w0, lambda_2, rho, tol, maxit, rho_change);
}

// [[Rcpp::export]]
List svm_admm_l0_sparse(const SpMatRd& X, const Eigen::VectorXd& y, const int s, Eigen::VectorXd w0,
                        const double lambda_2 = 0.001, double rho = 1,
                        const double tol = 1e-4, const int maxit = 500, const bool rho_change=true){
  
  return svm_admm_l0(X, y, s, w0, lambda_2, rho, tol, maxit, rho_change);
}

// [[Rcpp::export]]
List Adsvm_admm_l0_dense(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, int s_max , Eigen::VectorXd beta_init,
                         bool warm_start = true, int max_iter = 200){
  
  return Adsvm_admm_l0(X, y, s_max, beta_init, warm_start, max_iter);
}

// [[Rcpp::export]]
List Adsvm_admm_l0_sparse(const SpMatRd& X, const Eigen::VectorXd& y, int s_max , Eigen::VectorXd beta_init,
                          bool warm_start = true, int max_iter = 200){
  
  return Adsvm_admm_l0(X, y, s_max, beta_init, warm_start, max_iter);
}
