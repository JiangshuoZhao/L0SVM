#include "ssvm.h"
#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::interfaces(r,cpp)]]

using sec = std::chrono::seconds;
using mil_sec = std::chrono::milliseconds;
using sys_clk = std::chrono::system_clock;

// [[Rcpp::plugins("cpp11")]]


Eigen::VectorXd check(Eigen::ArrayXd val, double t = 0.5) {
  for(int i=0; i < val.size(); i++){
    val[i] = 1.0 - (val[i] < t); 
  }
  return val.matrix();
}


// [[Rcpp::export]]
double hinge_loss(Eigen::MatrixXd X_, Eigen::VectorXd psol,
                  double alpha = 0.01, double beta = 0.0, double gamma = 0.0) {
  
  double n_sams = X_.rows();
  double pobj = (.5 * alpha) * psol.squaredNorm() + (beta) * psol.lpNorm<1>();
  
  double loss = 0.0;
  Eigen::ArrayXd Xw_comp = 1 - (X_ * psol).array();
  
  double Xw_comp_i;
  
  if (gamma > 0){
    for (int i = 0; i < n_sams; ++i) {
      Xw_comp_i = Xw_comp[i];
      if (Xw_comp_i > gamma){
        loss += Xw_comp_i - .5 * gamma;
      } else if (Xw_comp_i > 0.0) {
        loss += .5 * Xw_comp_i * Xw_comp_i / gamma;
      }
    }
  } else{
    loss = (Xw_comp >= 0).select(Xw_comp, 0.0).sum();
    // for (int i = 0; i < n_sams_; ++i) {
    //   Xw_comp_i = Xw_comp_[i];
    //   if (Xw_comp_i > 0){
    //         loss_ += Xw_comp_i;
    //     }
    // }
  }
  
  pobj = pobj + loss / n_sams;
  
  return pobj;
};

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


Eigen::VectorXd Hard(Eigen::VectorXd beta, int s) {
  Eigen::VectorXd abs_beta = beta.cwiseAbs();
  // Partially sort the absolute values in decreasing order
  std::nth_element(abs_beta.data(), abs_beta.data() + s - 1, abs_beta.data() + abs_beta.size(), std::greater<double>());
  
  // Compute the result
  Eigen::VectorXd result = (beta.cwiseAbs().array() >= abs_beta[s-1]).select(beta, 0.0);
  return result;
}

Eigen::VectorXd project(const Eigen::VectorXd& x, double a = 0.0, double b = 1.0) {
  return x.cwiseMax(a).cwiseMin(b);
}

// [[Rcpp::export]]
List bess_svm(Eigen::MatrixXd X, Eigen::ArrayXd y,
              int s, Eigen::VectorXd beta_init,
              double lambda_2 = 1e-3, 
              double a = 1e-2, double b = 1e3, double gamma = 0.5,
              int c_max = 2, int splicing_type = 2,
              double eps= 1e-6, int max_iter = 500, 
              bool refitted = true){
  
  auto start_time = sys_clk::now();
  
  int cnt = 0;
  int n = X.rows();
  int p = X.cols();
  
  int s0 = min(n/2, int(2*n/log(n)));
  
  Eigen::MatrixXd X_ = X;
  Eigen::MatrixXd X_A0(n,s0), X_A(n,s), X_I(n,p-s);
  
  for (int i = 0; i < X_.rows(); ++i){
    X_.row(i) *= y[i];
  }
  
  Eigen::VectorXd alpha0 = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd beta0(p), beta1(p), beta(p), beta_new(p), alpha(n), abs_beta(p);
  Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(s);
  Eigen::VectorXi A0(s), A_new(s), A1(s), A_old(s), A, A0_(s0);
  Eigen::VectorXi I0(p-s), I1(p-s), I_new(p-s), I(p-s);
  Eigen::VectorXd e(p), bd(p), bd_A(s), bd_I(p-s), delta(p);
  double bias, L0, L1, L2;
  int iter = 0;//iter number
  delta = Eigen::VectorXd::Zero(p);
  
  Eigen::VectorXd Bb(p+1), Bb0(s+1);
  vector<int> ind(s), Ind(p);
  // std::iota(ind.begin(), ind.end(), 1);
  std::iota(Ind.begin(), Ind.end(), 1);
  double b0;
  
  cout<<"beta size:"<<beta_init.size()<<endl;
  
  if(beta_init.size() != 1){
    beta0 = beta_init;
    // cout<<"1 init"<<endl;
  }else{
    beta0 = huberized_svm(X, y, lambda_2);
    // cout<<"2 init"<<endl;
  }
  
  auto end_time1 = sys_clk::now();
  double time1 = 1e-3 * static_cast<double>(
    std::chrono::duration_cast<mil_sec>(end_time1 - start_time).count());
  
  if(s < p){
    
    //alpha = check(1.0 - (X_ * beta0).array(), 0);
    alpha = project(1.0 - (X_ * beta0).array());
    
    abs_beta = beta0.cwiseAbs();
    A0 = max_k(abs_beta, s);
    
    I0 = complement(A0, p);
    
    bool d = true;
    
    while(iter < max_iter && d ){
      for(int mm=0;mm< s;mm++) {
        X_A.col(mm) = X.col(A0[mm]);
      }
      
      beta0.setZero();
      beta_A = huberized_svm(X_A, y, lambda_2);

      cnt++;
      
      beta0(A0) = beta_A;
      
      L0 = hinge_loss(X_, beta0, lambda_2);
      
      e = beta0 +  gamma * delta;
      
      alpha = project(alpha.array() + 1.0/a/n*(1.0 - (X_ * e).array())) ;
      
      //bd = 1.0/n/(lambda_1) * X_.transpose() * alpha;
      bd = b/(lambda_2 + b)*beta0 + 1.0/n/(lambda_2 + b) * X_.transpose() * alpha;
      // beta = Hard(bd, s);
      bd = bd.cwiseAbs();
      A1 = max_k(bd, s);
      //cout<<"A1: "<<A1.transpose()<<endl;
      I1 = complement(A1, p);
      
      
      for(int mm=0;mm< s;mm++) {
        X_A.col(mm) = X.col(A1[mm]);
      }
      
      bd_A = bd(A1);
      bd_I = bd(I1);

      beta1.setZero();
      beta_A = huberized_svm(X_A, y, lambda_2);

      cnt++;
      // cout<<"ok2"<<endl;
      for(int mm=0; mm < s;mm++) {
        beta1(A1[mm]) = beta_A[mm];
      }
      L1 = hinge_loss(X_, beta1, lambda_2);
      //               
      int s_change = min(min(s, p-s) - 1, c_max);
      if(s_change >0){
        A = A1;
        I = I1;
        int j = s_change;
        while(j >= 1){
          Eigen::VectorXi A_min_k_ind = min_k(bd_A, j, false);// bd(A)
          Eigen::VectorXi I_max_k_ind = max_k(bd_I, j, false);//bd(I)
          
          Eigen::VectorXi j_out = vector_slice(A, A_min_k_ind);
          Eigen::VectorXi j_in = vector_slice(I, I_max_k_ind);
          //cout<<"j_in:"<<j_in.transpose()<<endl;
          
          A_new = diff_union(A, j_out, j_in);
          I_new = complement(A_new, p);
          
          for(int mm=0;mm< s;mm++) {
            X_A.col(mm) = X.col(A_new[mm]);
          }
          
          beta_new.setZero();
          beta_A = huberized_svm(X_A, y, lambda_2);
          cnt++;

          beta_new(A_new) = beta_A;

          L2 =  hinge_loss(X_, beta_new, lambda_2);
          if(L1 > L2){
            A1 = A_new;
            I1 = I_new;
            beta1 = beta_new;
            L1 = L2;    
          }
          if(splicing_type == 2) j/=2;
          else j--;
        }
      }
      A_old = A0;
      //             
      if(L0 > L1 + eps)
        A0 = A1;
      
      iter += 1;
      if(A_old == A0)
        d = false;
      delta = beta1-beta0;
    }
  }
  
  auto end_time2 = sys_clk::now();
  double time2 = 1e-3 * static_cast<double>(
    std::chrono::duration_cast<mil_sec>(end_time2 - end_time1).count());
  
  if(refitted == true){
    
    int s_t = 0;
    std::vector<int> idx;
    for(int i = 0; i < p; i++){
      if(beta0[i] != 0){
        idx.push_back(i);
        s_t += 1;
      }
    }
    
    //A <- which(beta0!=0)
    A.resize(s_t);
    beta_A.resize(s_t);
    X_A.resize(n,s_t);
    A = Eigen::Map<Eigen::VectorXi>(idx.data(), s_t);
    
    for(int mm=0;mm< s_t;mm++) {
      X_A.col(mm) = X.col(A[mm]);
    }
    
    beta.setZero();
    beta_A = huberized_svm(X_A, y, lambda_2);

    beta(A) = beta_A;
    bias = 0;
  }
  
  auto end_time3 = sys_clk::now();
  double time3 = 1e-3 * static_cast<double>(
    std::chrono::duration_cast<mil_sec>(end_time3 - end_time2).count());
  
  std::cout<<"time1: "<<time1 <<"time2: "<<time2<<"time3: "<<time3<<endl;
  //cout<<cnt<<endl;
  return List::create(Named("w")=beta, Named("A")=A, Named("Loss")=hinge_loss(X_, beta, lambda_2), Named("iter") = iter);
  
}

// [[Rcpp::export]]
double logC(int n, int m){
  double ans = 0;
  for(int i =1;i<=m;i++){
    ans = ans + log(n-m+i)-log(i); //
  }
  return ans;
}

// [[Rcpp::export]]
List abess(Eigen::MatrixXd X, Eigen::ArrayXd y, int s_max , Eigen::VectorXd beta_init,
           bool warm_start = true, int max_iter = 200){
  int n = X.rows();
  int p = X.cols();
  Eigen::VectorXd beta(p), beta0(p);

  int s_Max = min(p, int(n/(log(p))));
  if(s_max == -1) s_max = min(n, s_Max);
  s_max = max(s_Max, s_max);

  cout<<"s_max:" <<s_max<<endl;
  int size;
  double L;
  double L1 = log(log(n)), L2 = sqrt(log(n)), L3 = log(n), L4 = pow(n, -1.0/3.0);
  Eigen::VectorXd ebic(s_max), sic(s_max), sic1(s_max), sic2(s_max), sic3(s_max), sic4(s_max);

  std::vector<double> supp(s_max), HingeLoss(s_max);

  List result;

  Eigen::MatrixXd X_ = X;
  for (int i = 0; i < n; ++i){
    X_.row(i) *= y[i];
  }

  Eigen::ArrayXd Xw;
  if(beta_init.size() == 1)
    beta0 = huberized_svm(X, y);
  else
    beta0 = beta_init;

  for(int s = 0; s < s_max; s++){
    std::cout<<s<<"start"<<endl;
    if(warm_start){
      if(s>0)
        result = bess_svm(X, y, s+1, beta);
      else
        result = bess_svm(X, y, s+1, beta0);
    }else result = bess_svm(X, y, s+1, beta0);

    beta = result["w"];

    Eigen::VectorXi A = result["A"];

    Xw = 1 - (X_ * beta).array();

    L = (Xw >= 0).select(Xw, 0.0).sum();
    size = A.size();

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

