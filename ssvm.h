#pragma once
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

template <typename T> inline T val_sign(T val) {
  return 1.0 - (val <= 0.0) - (val < 0.0);
}

template<class T>
class ssvm {
public:
  ssvm(const T& X, const Eigen::ArrayXd& y,
       const double& alpha = 0.001, 
       const double& beta = 0.0,
       const double& gamma = 0.5,
       const double& tol = 1e-8,
       const int& max_iter = 10000,
       const int& chk_fre = 1);
  
  ~ssvm() {};
  
  int get_n_sams(void) const { return n_sams_; };
  int get_n_feas(void) const { return n_feas_; };
  int get_iter(void) const{ return iter_; };
  
  double get_duality_gap(void) const { return duality_gap_; };
  
  Eigen::VectorXd get_dual_sol(void) const { return dsol_; };
  Eigen::VectorXd get_primal_sol(void) const { return psol_; };
  
  void set_stop_tol(const double& tol) { tol_ = tol; };
  // compute primal objective
  void compute_primal_obj(const bool& flag_comp_loss) ;
  // compute dual objective
  void compute_dual_obj(const bool& flag_comp_XTdsol); 
  // compute duality gap
  void compute_duality_gap(const bool& flag_comp_loss,
                           const bool& flag_comp_XTdsol); 
  
  void update_psol(const bool& flag_comp_XTdsol);
  
  void train(void);
  
private:
  int n_sams_;
  int n_feas_;
  
  T X_;  // \bar{X}, each row contains one sample
  // T X_CM_;
  Eigen::VectorXd y_;  // training labels
  
  double alpha_;
  double beta_;
  
  double gamma_;
  double tol_;
  int max_iter_;
  int chk_fre_;
  int iter_;
  
  double pobj_;
  double dobj_;
  double duality_gap_;
  double loss_;
  
  double inv_n_sams_;
  double inv_alpha_;
  double inv_gamma_;
  
  Eigen::ArrayXd one_over_XTones_;  // \bar{X}^T * ones / n
  Eigen::VectorXd psol_;
  Eigen::VectorXd dsol_;
  Eigen::VectorXd XTdsol_; // \bar{X}^T * theta / n
  Eigen::ArrayXd Xw_comp_; // 1 - \bar{X} * w
  
  Eigen::ArrayXd Xi_norm_;
  Eigen::ArrayXd Xi_norm_sq_;
  Eigen::ArrayXd Xj_norm_;
  Eigen::ArrayXd Xj_norm_sq_;
  
  std::vector<int> all_ins_index_;
  
};

template<class T>
ssvm<T>::ssvm(const T& X, const Eigen::ArrayXd& y,
              const double& alpha, 
              const double& beta,
              const double& gamma,
              const double& tol,
              const int& max_iter, 
              const int& chk_fre)
  : X_(X), y_(y), alpha_(alpha), beta_(beta),
    gamma_(gamma), tol_(tol),
    max_iter_(max_iter), chk_fre_(chk_fre){
  
  for (int i = 0; i < X_.rows(); ++i){
    X_.row(i) *= y_[i];
  }
  
  // X_CM_ = X_;
  
  n_sams_ = X_.rows();
  n_feas_ = X_.cols();
  
  inv_n_sams_ = 1.0 / static_cast<double>(n_sams_);
  inv_gamma_ = 1.0 / gamma_;
  inv_alpha_ = 1.0 / alpha_;
  
  dsol_ = Eigen::VectorXd::Ones(n_sams_);
  Xw_comp_ = dsol_;
  psol_ = Eigen::VectorXd::Zero(n_feas_);
  XTdsol_ = psol_;
  
  Xi_norm_.resize(n_sams_);
  for (int i = 0; i < n_sams_; ++i) {
    Xi_norm_[i] = X_.row(i).norm();
  }
  
  Xj_norm_.resize(n_feas_);
  for (int j = 0; j < n_feas_; ++j) {
    Xj_norm_[j] = X_.col(j).norm();
  }
  
  Xi_norm_sq_ = Xi_norm_.square();
  Xj_norm_sq_ = Xj_norm_.square();
  
  all_ins_index_.resize(n_sams_);
  std::iota(std::begin(all_ins_index_), std::end(all_ins_index_), 0);
  
  iter_ = 0;
  duality_gap_ = std::numeric_limits<double>::max();
}

template<class T>
void ssvm<T>::compute_primal_obj(const bool& flag_comp_loss) {
  pobj_ = (.5 * alpha_) * psol_.squaredNorm() + (beta_) * psol_.lpNorm<1>();
  
  loss_ = 0.0;
  if (flag_comp_loss){
    Xw_comp_ = 1 - (X_ * psol_).array();
    double Xw_comp_i;
    if (gamma_ >0){
      for (int i = 0; i < n_sams_; ++i) {
        Xw_comp_i = Xw_comp_[i];
        if (Xw_comp_i > gamma_){
          loss_ += Xw_comp_i - .5 * gamma_;
        } else if (Xw_comp_i > 0.0) {
          loss_ += .5 * Xw_comp_i * Xw_comp_i * inv_gamma_;
        }
      }
    } else{
      loss_ = (Xw_comp_ >= 0).select(Xw_comp_, 0.0).sum();
    }
  }
  pobj_ = pobj_ + loss_ * inv_n_sams_;
};

// compute dual objective
template<class T>
void ssvm<T>::compute_dual_obj(const bool& flag_comp_XTdsol){
  if (flag_comp_XTdsol)
    update_psol(true);
  
  Eigen::ArrayXd temp = XTdsol_.array().abs() - beta_ ;
  dobj_ =
    (dsol_.sum() - (.5 * gamma_) * dsol_.squaredNorm()) * inv_n_sams_
  - (0.5 * inv_alpha_) * (temp >= 0.0).select(temp, 0.0).square().sum();
}; 

// compute duality gap
template<class T>
void ssvm<T>::compute_duality_gap(const bool& flag_comp_loss,
                                  const bool& flag_comp_XTdsol){
  compute_primal_obj(flag_comp_loss);
  compute_dual_obj(flag_comp_XTdsol);
  
  duality_gap_ = std::max(0.0, pobj_ - dobj_);
}; 

template<class T>
void ssvm<T>::update_psol(const bool& flag_comp_XTdsol){
  if (flag_comp_XTdsol) {
    XTdsol_.setZero();
    for (int i = 0; i < n_sams_; ++i) {
      if (dsol_[i] > 0.0) {
        XTdsol_ += dsol_[i] * X_.row(i);
      }
    }
    XTdsol_ *= inv_n_sams_;
  }
  
  // psol_ = inv_alpha_ * XTdsol_; 
  for (int i = 0; i < n_feas_; ++i) {
    psol_[i] = val_sign(XTdsol_[i]) * inv_alpha_ *
      std::max(0.0, std::abs(XTdsol_[i]) - beta_);
  }
};

template<class T>
void ssvm<T>::train(void){
  int ind = 0;
  const double inv_nalpha_ = inv_n_sams_ * inv_alpha_;
  double delta_ind = 0.0;
  double p_theta_ind = 0.0;
  
  std::default_random_engine rg;
  std::uniform_int_distribution<> uni_dist(0, n_sams_ - 1);
  
  update_psol(true);
  compute_duality_gap(true, false);
  
  const auto ins_begin_it = std::begin(all_ins_index_);
  auto random_it = std::next(ins_begin_it, uni_dist(rg));
  for (iter_ = 1; iter_ < max_iter_ && duality_gap_ > tol_; ++iter_) {
    for (int jj = 0; jj < n_sams_; ++jj) {
      random_it = std::next(ins_begin_it, uni_dist(rg));
      ind = *random_it;
      
      p_theta_ind = dsol_[ind];
      
      delta_ind = (1 - gamma_ * p_theta_ind - (X_.row(ind) * psol_)(0)) /
        (gamma_ + Xi_norm_sq_[ind] * inv_nalpha_);
      // auto end_time3 = sys_clk::now();
      
      delta_ind = std::max(-p_theta_ind, std::min(1.0 - p_theta_ind,
                                                  delta_ind));
      dsol_[ind] += delta_ind;
      XTdsol_ +=  (delta_ind * inv_n_sams_) * X_.row(ind);
      
      if(beta_ > 0){
        for (int kk = 0; kk < n_feas_; ++kk) {
          psol_[kk] = val_sign(XTdsol_[kk]) * inv_alpha_ *
            std::max(0.0, std::abs(XTdsol_[kk]) - beta_);
        }
      }
      else{
        psol_ = inv_alpha_ * XTdsol_;
      }
      
    }
    
    if (iter_ % chk_fre_ == 0) {
      compute_duality_gap(true, false);
    }
  }
};



template<class T>
Eigen::VectorXd huberized_svm(const T& X, const Eigen::ArrayXd& y,
                              double alpha = 0.001,
                              double beta = 0.0,
                              double gamma = 0.0, 
                              double tol = 1e-5, 
                              int max_iter = 2000, 
                              int chk_fre = 1){
  
 
  ssvm<T> solver(X, y, alpha, beta, gamma, tol, max_iter, chk_fre);
  solver.train();
  
  Eigen::VectorXd w = solver.get_primal_sol();
  
  Eigen::VectorXd a = solver.get_dual_sol();
  double duality_gap = solver.get_duality_gap();
  int iter = solver.get_iter();
  // cout<<"duality gap:"<<duality_gap<<endl;
  
  return w;
}


