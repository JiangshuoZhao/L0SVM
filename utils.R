require(caret)
require(Matrix)
require(lpSolve)
require(LiblineaR)
require(pracma)
library(gurobi)

library(glmnet)
library(ncvreg)
library(abess)
# library(gcdnet)
Rcpp::sourceCpp("/home/zhaojs/svm/huberized_svm.cpp", verbose = FALSE)
Rcpp::sourceCpp("/home/zhaojs/svm/demo.cpp", verbose = FALSE)
Rcpp::sourceCpp("/home/zhaojs/svm/admm2.cpp", verbose = FALSE)

require(L0Learn)
# read.libsvm.csr <- function(file, fac = FALSE, ncol = NULL){
#   
#   l <- strsplit(readLines(file), "[ ]+")
#   
#   ## extract y-values, if any
#   y <- if (is.na(l[[1]][1]) || length(grep(":",l[[1]][1])))
#     NULL
#   else
#     vapply(l, function(x) as.integer(x[1]), integer(1))
#   
#   ## x-values
#   rja <- do.call("rbind",
#                  lapply(l, function(x)
#                    do.call("rbind",
#                            strsplit(if (is.null(y)) x else x[-1], ":")
#                    )
#                  )
#   )
#   ja <- as.integer(rja[,1])
#   ia <- cumsum(c(1, vapply(l, length, integer(1)) - !is.null(y)))
#   
#   max.ja <- max(ja)
#   dimension <- c(length(l), if (is.null(ncol)) max.ja else max(ncol, max.ja))
#   
#   x = new(getClass("matrix.csr", where = asNamespace("SparseM")),
#           ra = as.numeric(rja[,2]), ja = ja,
#           ia = as.integer(ia), dimension = as.integer(dimension))
#   if (length(y))
#     return(list(x = x, y = if (fac) as.factor(y) else as.numeric(y)))
#   else return(list(x=x))
# }

# libsvm格式读取数据
read.libsvm.dgR <- function(file, fac = FALSE, ncol = NULL){
  
  l <- strsplit(readLines(file), "[ ]+")
  
  ## extract y-values, if any
  y <- if (is.na(l[[1]][1]) || length(grep(":",l[[1]][1])))
    NULL
  else
    vapply(l, function(x) as.integer(x[1]), integer(1))
  
  ## x-values
  rja <- do.call("rbind",
                 lapply(l, function(x)
                   do.call("rbind",
                           strsplit(if (is.null(y)) x else x[-1], ":")
                   )
                 )
  )
  ja <- as.integer(rja[,1])-1
  # print(ja)
  p <- cumsum(c(0, vapply(l, length, integer(1)) - !is.null(y)))
  
  max.ja <- max(ja + 1)
  dimension <- c(length(l), if (is.null(ncol)) max.ja else max(ncol, max.ja))
  
  x = new('dgRMatrix', j = as.integer(ja), p = as.integer(p), x = as.numeric(rja[,2]), Dim = as.integer(dimension))
  if (length(y))
    return(list(x = x, y = if (fac) as.factor(y) else as.numeric(y)))
  else return(list(x=x))
}

# 数据处理
scale.train <- function(x, l = -1, u = 1){
  x_min = apply(x, 2, min)
  x_max = apply(x, 2, max)
  x_range = x_max - x_min
  x_mean = (x_min + x_max)/2
  
  scale_vector <- function(y){
    if(diff(range(y)) == 0) return(y)
    else return( (u-l) * (y - (max(y)+min(y))/2 ) / diff(range(y)) )
  }
  xx <- apply(x, 2, scale_vector)
  return(list(x = xx, x_mean = x_mean, x_range = x_range))
}


scale.test <- function(x.t, x_mean, x_range){
  n = nrow(x.t)
  p = ncol(x.t)
  xx = matrix(NA, n, p)
  for (j in 1:p) {
    if(x_range[j] == 0) {xx[,j] = x.t[,j]}
    else {xx[,j] = 2 * scale(x.t[,j], x_mean[j], x_range[j])}
  }
  return(xx)
}

# 硬阈值算子
hard <- function(x, s){
  p <- length(x)
  if(s<p) x[order(abs(x), decreasing = TRUE)[(s+1):p]] <- 0
  x
}
# 硬阈值算子
# hard <- function(x, lambda){
#   
#   x[x^2 <= lambda] <- 0
#   x
# }

# 软阈值算子
soft <- function(x, lambda){
  sign(x) * pmax(abs(x)-lambda, 0)
}

accuracy <- function(y_hat, y){
  accuracy <-  mean((y * sign(as.vector(y_hat))) > 0)
  return(accuracy)
}

precision_comp<- function(actual, predicted){
  TP = sum(actual==1 & sign(predicted)==1)
  FP = sum(actual==-1 & sign(predicted)==1)
  if(TP+FP==0)
    return(NA)
  return(TP/(TP+FP))
}

recall_comp<-function(actual, predicted){
  TP = sum(actual==1 & sign(predicted)==1)
  FN = sum(actual==1 & sign(predicted)==-1)
  if(TP+FN==0)
    return(NA)
  return(TP/(TP+FN))
}

# ### PAM
# svmpam <- function(X, y, s, beta_init=NULL, method = c("L0", "L1"), a = 1e-2, b = 1e3, lambda_2 = 0.001,
#                    gamma=0, eps= 1e-6, max.iter = 2000, refitted = TRUE){
# 
#   n <- length(y)
#   p <- dim(X)[2]
#   X_ = y*X
#   # alpha0 <- rep(0, n)
#   check <- function(x, tau = 1)  tau - (x<0)
# 
#   if(is.null(beta_init)){
#     beta0 <- huberized_svm(X, y, lambda_2)
#   }
#   else{
#     beta0 <- beta_init
#   }
# 
#   alpha0 <- check(1 - X_ %*% beta0)
#   k <- 0
#   d <- 1
#   bd <- 0
#   beta <- beta0
# 
#   while(k < max.iter & d > eps){
#     # print(k)
#     e <- beta +  gamma*bd
#     # alpha <- 1 - (alpha0 + 1/a/n*(1-X_%*%e)<=0)
#     alpha = pmin( pmax(alpha0 + 1/a/n*(1-X_%*%e), 0) ,1)
# 
#     # beta <- hard(b/(lambda_2 + b)*beta0 + 1/(b+lambda_2)/n*t(X_)%*%alpha, s)
#     beta <- soft((b-lambda_2)/b *beta0 + 1/(b)/n*t(X_)%*%alpha, s)
#     cat("k: ",k, "active set: ", which(beta!=0),'\n')
# 
#     bd <- beta - beta0
#     d <- sum(bd^2)
# 
#     k <- k + 1
#     beta0 <- beta
#     alpha0 <- alpha
#   }
# 
#   if(refitted == TRUE){
#     A <- which(beta0!=0)
#     beta <- rep(0, p)
#     beta[A] <- huberized_svm(as.matrix(X[,A]),y,lambda_2)
#   }
# 
#   return(list(beta = beta, alpha = alpha0, k=k))
# }

huberized_svm <- function(X, y, alpha = 1e-3, beta =0.0,
                          gamma = 0.0, tol =1e-5, max.iter =2e3, chk_fre=1){
  
  M <- list()
  M <- huberized_svm_dense(X, y, alpha, beta, gamma, tol, max.iter, chk_fre)
  # if (is(X, "sparseMatrix")){
  #   M <- huberized_svm_sparse(X, y, alpha, beta, gamma, tol, max.iter, chk_fre)
  # }else{
  #   M <- huberized_svm_dense(X, y, alpha, beta, gamma, tol, max.iter, chk_fre)
  # }
  # 
  return(M)
}


svm_admm_l0 <- function(X, y, s, w0, lambda_2 = 1e-3, rho = 1,
                              tol = 1e-4, max.it = 500, rho_change = TRUE){

  M <- list()
  if (is(X, "sparseMatrix")){
    M <- svm_admm_l0_sparse(X, y, s, w0, lambda_2, rho, tol, max.it, rho_change)
  }else{
    M <- svm_admm_l0_dense(X, y, s, w0, lambda_2, rho, tol, max.it, rho_change)
  }
  
  return(M)
}


Adsvm_admm_l0 <- function(X, y, s_max, beta_init, 
                          warm_start = TRUE, max.iter = 200){
  
  M <- list()
  if (is(X, "sparseMatrix")){
    M <- Adsvm_admm_l0_sparse(X, y, s_max, beta_init, warm_start, max.iter);
  }else{
    M <- Adsvm_admm_l0_dense(X, y, s_max, beta_init, warm_start, max.iter)
  }
    return(M)
}



# 信息准则
ic.method <- function(X, y, method = "l1", cost_list = NULL, w_init = NULL){
  n = nrow(X); p = ncol(X)
  if (is.null(cost_list)) cost_list = logspace(-7, 2, n=100)
  
  X_ = y*X
  ebic = sic = sic1 = sic2 = sic3 = sic4 = vector()
  HingeLoss = supp = vector()
  L1 = log(log(n))
  L2 = sqrt(log(n))
  L3 = log(n)
  L4 = n^(-1/3)
  for(s in 1:length(cost_list)){
    if(s>1){
      if(method == "l1")
        w <- svm.l1(X, y, cost_list[s])$w
      else if(method == "scad")
        w <- svm.scad(X, y, cost_list[s], w_init = w)$w
      else if(method == "l2")
        w <- drop(LiblineaR(X, y, type = 3, bias = 0, cost = cost_list[s])$W)
      else if(method == "admml1")
        w <- drop(svm.admm_elastic(X, y, cost_list[s], w0=w)$Z)
      else if(method == "l1l2")
        w <- huberized_svm(X, y, alpha = 0.001, beta = cost_list[s], gamma = 0.05)
      else if(method == "mcp")
        w <- svm.mcp(X, y, cost_list[s], w_init = w)$w
    }else{
      if(method == "l1")
        w <- svm.l1(X, y, cost_list[s])$w
      else if(method == "scad")
        w <- svm.scad(X, y, cost_list[s], w_init = w_init)$w
      else if(method == "l2")
        w <- drop(LiblineaR(X, y, type = 3, bias = 0, cost = cost_list[s])$W)
      else if(method == "admml1")
        w <- drop(svm.admm_elastic(X, y, cost_list[s])$Z)
      else if(method == "l1l2")
        w <- huberized_svm(X, y, alpha = 0.001, beta = cost_list[s], gamma = 0.05)
      else if(method == "mcp")
        w <- svm.mcp(X, y, cost_list[s], w_init = w_init)$w
    }
    size = sum(w != 0)
    if (size==0) break
    
    supp[s] = size
    X_w = 1 - X_ %*% w
    L = sum(X_w[X_w>0])
    HingeLoss[s] = L
    ebic[s] = L  + size *log(n) + logC(p,size)
    sic[s] = L + size * log(n)
    sic1[s] = L + size*log(n)*L1
    sic2[s] = L + size*log(n)*L2
    sic3[s] = L + size*log(n)*L3
    sic4[s] = L + size *log(n)* L4
  }
  
  return(list("ebic" = ebic, "sic" = sic, "sic1" = sic1, "sic2" = sic2, "sic3" = sic3, "sic4" = sic4,
              "HingeLoss"= HingeLoss, "supp" = supp))
  
}



# 交叉验证
cv.method = function(X, y, nfolds = 5, method = "l1", w_init = NULL, cost_list = NULL, seed=123){
  
  set.seed(seed)
  
  n = nrow(X)
  if (is.null(cost_list)) cost_list = logspace(-7, 2, n=100)
  
  L = length(cost_list)
  score = rep(0, L)
  
  all.folds = createFolds(1:n,nfolds)
  for (i in 1:nfolds) {
    # cat("method:", method, i, "fold start \n")
    omit <- all.folds[[i]]
    for(j in 1:L){
      cat("method:", method, i, "fold", j, "start \n")
      if(j>1){## warm start 
        if(method == "l1")
          w <- svm.l1(X[-omit, ], y[-omit], cost_list[j])$w
        else if(method == "scad")
          w <- svm.scad(X[-omit, ], y[-omit], cost_list[j], w_init = w)$w
        else if(method == "svm")
          w <- huberized_svm(X[-omit, ], y[-omit], cost_list[j])
        else if(method == "l2")
          w <- drop(LiblineaR(X[-omit, ], y[-omit], type = 3, bias = 0, cost = cost_list[j])$W)
        else if(method == "admml1")
          w <- drop(svm.admm_elastic(X[-omit, ], y[-omit], cost_list[j], w0=w)$Z)
        else if(method == "l1l2")
          w <- huberized_svm(X[-omit, ], y[-omit], alpha = 1e-3, beta = cost_list[j], gamma = 0.05)
        else if(method == "mcp")
          w <- svm.mcp(X[-omit, ], y[-omit], cost_list[j], w_init = w)$w
      }else{
        if(method == "l1")
          w <- svm.l1(X[-omit, ], y[-omit], cost_list[j])$w
        else if(method == "scad")
          w <- svm.scad(X[-omit, ], y[-omit], cost_list[j], w_init = w_init)$w
        else if(method == "svm")
          w <- huberized_svm(X[-omit, ], y[-omit], cost_list[j])
        else if(method == "l2")
          w <- drop(LiblineaR(X[-omit, ], y[-omit], type = 3, bias = 0, cost = cost_list[j])$W)
        else if(method == "admml1")
          w <- drop(svm.admm_elastic(X[-omit, ], y[-omit], cost_list[j], w0 = w_init)$Z)
        else if(method == "l1l2")
          w <- huberized_svm(X[-omit, ], y[-omit], alpha = 1e-3, beta = cost_list[j], gamma = 0.05)
        else if(method == "mcp")
          w <- svm.mcp(X[-omit, ], y[-omit], cost_list[j], w_init = w_init)$w
      }
      
      if(sum(w!=0)==0) break
      score[j] = score[j] + accuracy(X[omit, ] %*% w, y[omit])
    }
  }
  score = score / nfolds
  index = which.max(score)
  best.cost = cost_list[index]
  cat(method, "best_cost:", index, '\n')
  return(list(best.cost = best.cost , score = score))
}



cv.l0 <- function(X, y, nfolds = 5, s_max = NULL, beta_init = 0, lambda2 = 1e-3, s_list = NULL,gamma = 0.5, seed=123, 
                  c_max = 2, splicing_type = 2, admm=FALSE){
  
  set.seed(seed)
  
  n = nrow(X)
  p = ncol(X)
  s_Max = min(p, floor(n/log(p)))
  if(is.null(s_list)){
    if(is.null(s_max)) s_max = min(n, s_Max)
    s_max = min(p, s_max)
    s_list = 1:s_max
  }
  
  L = length(s_list)
  score = rep(0, L)
  all.folds = createFolds(1:n, nfolds)
  
  cat("s_list length:",L, '\n')
  
  for (i in 1:nfolds) {
    # cat(i,"fold start \n")
    omit <- all.folds[[i]]
    for(s in seq(L)){
      cat(i,"fold",s,"start \n")
      if(s>100){
        if(admm)
          w <- svm_admm_l0(X[-omit, ], y[-omit], s_list[s], w, lambda_2 = lambda2)$w
        else{
          w <- bess_svm(X[-omit, ], y[-omit], s_list[s], c(b,w), lambda2,
                        c_max = c_max, splicing_type = splicing_type)$w
          }
      }else{
        if(admm)
          w <- svm_admm_l0(X[-omit, ], y[-omit], s_list[s], beta_init, lambda_2 = lambda2)$w
        else{
          w <- bess_svm(X[-omit, ], y[-omit], s_list[s], beta_init, lambda2,
                        c_max = c_max, splicing_type = splicing_type)$w
          }
      }
      
      score[s] = score[s] + accuracy(X[omit, ] %*% w , y[omit])
    }
  }
  score = score / nfolds
  s_best = s_list[which.max(score)]
  cat("l0 s_best:", s_best, '\n')
  return(list(s_best = s_best , score = score))
}

## solve L1 norm SVM using linear programming problem R packages
## X (n*p)
# svm.l1 <- function(X, Y, lambda = .01, bias = FALSE){
#   n = nrow(X)
#   p = ncol(X)
#   if(length(lambda) != p) lambda = rep(lambda, p)
#   
#   X_ = Y*X
#   # lambda = 0 # sqrt(log(p)/n)
#   if(bias){
#     f.obj <- c(lambda, lambda, 0, rep(1/n,n))
#     #         pvec  qvec        b,  uvec
#     f.con <- cbind(X_, -X_, Y, diag(1,n))
#   } else{
#     f.obj <- c(lambda, lambda, rep(1/n,n))
#     #         pvec  qvec            uvec
#     f.con <- cbind(X_, -X_, diag(1,n))
#   }
#   f.rhs <- c(rep(1,n))
#   f.dir <- rep(">=",n)
#   ## Minimize the cost
#   
#   res <- lp("min", f.obj, f.con, f.dir, f.rhs)
#   # summary(res)
#   # print(res)
#   w <- res$solution[1:p] - res$solution[(p+1):(2*p)]
#   b = 0
#   if(bias) {b = res$solution[(2*p+1)]}
#   return(list(w = w, b = b))
# }


# gurobi
svm.l1 <- function(X, Y, lambda, bias = FALSE){
  n = nrow(X)
  p = ncol(X)
  if(length(lambda) != p) lambda = rep(lambda, p)
  
  X_ = Y*X
  model <- list()
  # lambda = 0 # sqrt(log(p)/n)
  if(bias){
    model$obj <- c(lambda, lambda, 0, rep(1/n,n))
    #         pvec  qvec        b,  uvec
    model$A <- cbind(X_, -X_, Y, diag(1,n))
  } else{
    model$obj <- c(lambda, lambda, rep(1/n,n))
    #         pvec  qvec            uvec
    model$A <- cbind(X_, -X_, diag(1,n))
  }
  model$rhs <- c(rep(1,n))
  model$sense <- rep(">=",n)
  ## Minimize the cost
  params <- list(OutputFlag=0)
  res <- gurobi(model, params)
  # summary(res)
  # print(res)
  w <- res$x[1:p] - res$x[(p+1):(2*p)]
  b = 0
  if(bias) {b = res$x[(2*p+1)]}
  return(list(w = w, b = b))
}


# svm.l1 <-function(X, Y, lambda = 1/n, bias = FALSE){
#   n = nrow(X)
#   p = ncol(X)
#   if(length(lambda) != p) lambda = rep(lambda, p)
# 
#   X_ = Y*X
#   model <- list()
#   # lambda = 0 # sqrt(log(p)/n)
#   if(bias){
#     obj <- c(lambda, lambda, 0, rep(1/n,n))
#     #         pvec  qvec        b,  uvec
#     A <- cbind(X_, -X_, Y, diag(1,n))
#   } else{
#     obj <- c(lambda, lambda, rep(1/n,n))
#     #         pvec  qvec            uvec
#     A <- cbind(X_, -X_, diag(1,n))
#   }
#   rhs <- c(rep(1,n))
#   dir <- rep(">=",n)
#   ## Minimize the cost
#   res <- lp("min", obj,A ,dir, rhs)$solution
#   # summary(res)
#   # print(res)
#   w <- res[1:p] - res[(p+1):(2*p)]
#   b = 0
#   if(bias) {b = res[(2*p+1)]}
#   return(list(w = w, b = b))
# }


## solve hinge loss SVM using linear programming problem R packages
## X (n*p)
# svm.l0 <- function(X, Y, bias = FALSE){
#   n = nrow(X)
#   p = ncol(X)
#   
#   X_ = Y*X
#   if(bias){
#     f.obj <- c(rep(0, 2*p+1), rep(1/n,n) )
#     
#     ## Matrix A
#     f.con <- cbind(X_, -X_, Y, diag(1,n))
#   } else{
#     f.obj <- c(rep(0, 2*p), rep(1/n,n) )
#     ## Matrix A
#     f.con <- cbind(X_, -X_, diag(1,n))
#   }
#   ## Constraints (minimum (<0) and maximum (>0) contents)
#   f.rhs <- c( rep(1,n) )
#   f.dir <- rep(">=",n)
#   ## Minimize the cost
#   res <- lp("min", f.obj, f.con, f.dir, f.rhs)
#   # summary(res)
#   # print(res)
#   w <- res$solution[1:p] - res$solution[(p+1):(2*p)]
#   b <- 0
#   if(bias)  b <- res$x[(2*p+1)]
#   return(list(w = w, b = b))
# }

scad_derivative <- function(beta, lambda_val, a_val = 3.7){
  return (lambda_val * ((beta <= lambda_val) + (a_val * lambda_val - beta)*((a_val * lambda_val - beta) > 0) / ((a_val - 1) * lambda_val) * (beta > lambda_val)))
}

mcp_derivative <- function(beta, lambda_val, a_val = 3){
  return ((lambda_val - beta/a_val) *(beta <= a_val * lambda_val))
}

svm.scad <- function(X, Y, lambda, bias = FALSE, w_init = NULL, max_iter = 1e4, tol = 1e-7){
  n = nrow(X)
  p = ncol(X)
  
  if(is.null(w_init)) w_init = numeric(p)
  
  gap = 1
  t = 0
  wp1 = scad_derivative(abs(w_init), lambda)
  while (t < max_iter & gap > tol) {
    t = t+1
    # for (t in 1:3) {
    res = svm.l1(X, Y, wp1)
    w = res$w
    b = res$b
    
    wp2 = scad_derivative(abs(w), lambda)
    gap = sum((wp2 - wp1)^2)
    
    wp1 = wp2
  }
  
  return(list(w = w, b=b))
}

svm.mcp <- function(X, Y, lambda, bias = FALSE, w_init = NULL, max_iter = 1e4, tol = 1e-7){
  n = nrow(X)
  p = ncol(X)
  
  if(is.null(w_init)) w_init = numeric(p)
  # w_init = svm.l1(X, Y, lambda)$w
  
  gap = 1
  t = 0
  wp1 = mcp_derivative(abs(w_init), lambda)
  
  while (t < max_iter & gap > tol) {
    t = t+1
    # for(t in 1:3){
    
    res = svm.l1(X, Y, wp1)
    w = res$w
    b = res$b
    
    wp2 = mcp_derivative(abs(w), lambda)
    gap = sum((wp2 - wp1)^2)
    
    wp1 = wp2
  }
  
  return(list(w = w, b=b, gap =gap))
}


# ## this is using admm algorithm sovle standard svm
# svm.admm <- function(X, y, lambda, rho = 1,tol = 1e-3, maxit = 5000){
#   ## Alternating Direction Method of Multipliers (ADMM) for solving SVM
#   n = nrow(X)
#   p = ncol(X)
#   y = as.vector(y)
#   
#   w0 = rnorm(p)
#   b0 = 0
#   
#   ## constant
#   X = cbind(X, 1)
#   X = y * X
#   Q = diag(1, p+1)
#   Q[p+1, p+1] = 0
#   left = (lambda/rho)*Q + t(X)%*%X
#   rho_inv = 1 / rho
#   
#   w = c(w0, b0) # dependent variable for subproblem 1
#   
#   T = 1 - X%*%w # dependent variable for subproblem 2
#   u = rep(0,n) # Lagrangian multiplier
#   
#   # history residual
#   hist_pres = vector()
#   hist_dres = vector()
#   hist_obj = vector()
#   
#   iter = 0
#   while (TRUE) {
#     iter = iter + 1
#     
#     if (iter >= maxit) break
#     
#     old_w = w
#     old_T = T
#     
#     ## update w
#     w = solve(left, -t(X)%*%( rho_inv*u + T - 1) )
#     
#     ## update b
#     C = 1 - X%*% w - u*rho_inv
#     for (i in 1:n) {
#       if(rho_inv <C[i])
#         T[i] = C[i] - rho_inv
#       else{
#         if(0<=C[i])
#           T[i] = 0
#       }
#       
#     }
#     
#     ## compute primal residual and save to hist_pres
#     pres = sqrt(norm((T + (X %*% w) - 1)))
#     hist_pres = c(hist_pres, pres)
#     
#     ## compute the dual residual and save to hist_dres
#     dres = rho * sqrt(norm(( t(X) %*% ( T - old_T ) )))
#     # dres = beta * norm( X * ( W - old_W ) );
#     hist_dres = c(hist_dres, dres)
#     
#     obj = sum(sapply(T, max, 0)) + lambda/2 * norm(Q%*%w)
#     obj_2 = t(u)%*%(T + (X %*% w) - 1) + rho/2*norm(T + (X %*% w) - 1)^2
#     hist_obj = c(hist_obj, obj)
#     
#     cat(iter,"primal dual: ",pres, "dual residual:", dres,"obj_2",obj_2,'\n')
#     
#     if (max(pres, dres) <= tol)
#       break
#     
#     ## update Lagrangian multiplier
#     u = u + rho*(T + X %*% w -1)
#     
#   }
#   
#   return(list(w=w, hist_pres = hist_pres, hist_dres = hist_dres, hist_obj = hist_obj))
#   
#   
# } 
## this is using admm algorithm sovle elastic net(L1+L2) svm

svm.admm_elastic <- function(X, y, lambda_1, lambda_2 = 1e-3,  rho1 = 1, rho2 = 50,
                             w0 = NULL, tol = 1e-4, maxit = 500){
  ## Alternating Direction Method of Multipliers (ADMM) for solving SVM
  n = nrow(X); p = ncol(X)
  y = as.vector(y)
  if(is.null(w0)){
    #w0 = rnorm(p)
    w0 = numeric(p)
  }
  ## constant
  X = y*X
  
  Ip = diag(p)
  
  # left = diag(1, p+1)
  #left = (lambda_2 + rho2)* Ip + rho1 * H
  # left[1:p, p+1] = rho1 * colSums(X)
  # left[p+1, 1:p] = rho1 * colSums(X)
  # left[p+1, p+1] = rho1*n
  rho = (lambda_2+rho2)/rho1
  U = fractor(X, rho)
  U_inv = solve(U)
  
  w = w0 # dependent variable for subproblem 1
  
  A = 1 - X%*%w # dependent variable for subproblem 2
  Z = w # dependent variable for subproblem 3
  
  obj = mean(sapply(A, max, 0)) + lambda_1 * sum(abs(Z)) +lambda_2/2 * sum(w^2)
  
  u = rep(0,n) # Lagrangian multiplier
  v = rep(0,p)
  
  thr1 = 1/(n*rho1)
  
  # history residual
  hist_pres = vector()
  hist_dres = vector()
  hist_obj = vector()
  
  iter = 0
  while (iter<maxit) {
    iter = iter + 1
    
    old_w = w
    old_A = A
    old_Z = Z
    old_obj = obj
    
    ## update w
    # right = numeric(p+1)
    # right = t(X) %*% u - rho1 * t(X) %*% (A-1) - v + rho2*Z
    # right[p+1] = sum(y*u)- rho1*sum(A-1)
    
    # w = left_inv %*% right
    # tmp = solve(left, right)
    # w = tmp[1:p]
    # b = tmp[p+1]
    
    fk = t(X) %*% (1- (A - u / rho1)) + rho2/rho1 * (Z - v/rho2) ;
    
    if(n > p)
      w = U_inv %*% (t(U_inv) %*% fk)
    else
      w = fk/rho - (t(X) %*%( U_inv %*% (t(U_inv) %*% (X %*% fk) ) ) ) / rho^2;
    
    Xw =  1-X%*%w 
    
    ## update A
    A = Soft(Xw + u/rho1, thr1) 
    
    # for (i in 1:n) {
    #   if( thr1 < A[i] )
    #     A[i] = A[i] - thr1
    #   else{
    #     if(0<=A[i] & A[i] <= thr1)
    #       A[i] = 0
    #   }
    # }
    
    ## update Z
    Z = soft(v/rho2 + w, lambda_1/ rho2)
    
    
    ## compute primal residual and save to hist_pres
    cons2 = sqrt(sum((w-Z)^2)/p)
    cons1 = sqrt(sum((Xw -A)^2)/n)
    
    obj = mean(sapply(A, max, 0)) + lambda_1 * sum(abs(Z)) +lambda_2/2 * sum(w^2)
    obj_error = abs(obj - old_obj) / old_obj
    
    ## cat(iter,"obj:", obj, "cons1:",cons1,"cons2:", cons2, "nnz:",sum(Z!=0), '\n')
    
    if(obj_error <= tol && cons1 <= tol && cons2<=tol) {
      #cat(iter,"obj:", obj_error, "cons1:",cons1,"cons2:", cons2, "nnz:",sum(Z!=0), '\n')
      break
    }
    
    hist_obj = c(hist_obj, obj)
    
    #cat(iter,"primal dual: ",pres, "dual residual:", dres,"obj_2",obj_2,'\n')
    
    ## update Lagrangian multiplier
    u = u + rho1*(Xw- A)
    v = v + rho2*(w - Z)
    # cat("v: ", v, '\n')
    
  }
  return(list(w=w, Z = Z, hist_obj = hist_obj, iter = iter))
  
} 

svm.admm_l0 <- function(X, y, s, w0,lambda_2 = 1e-3, type =2, rho = 1,  tol = 1e-4, maxit = 500){
  ## Alternating Direction Method of Multipliers (ADMM) for solving SVM
  n = nrow(X); p = ncol(X)
  y = as.vector(y)

  # w0 = as.vector(LiblineaR(X, y, type = 3, bias = 0)$W)
  # w0 = numeric(p)
  # cat("0 active set :", which(hard(w0,s)!=0),'\n')
  ## constant
  X = y*X

  w = w0 # dependent variable for subproblem 1
  A = 1 - X%*%w # dependent variable for subproblem 2

  obj = mean(pmax(A, 0)) +  lambda_2/2 * sum(w^2)

  u = rep(0,n) # Lagrangian multiplier

  lambda_ = lambda_2/(2*rho*n)
  thr1 = 1/(n*rho)

  # history residual
  hist_pres = vector()
  hist_dres = vector()
  hist_obj = vector()

  iter = 0
  while (iter < maxit) {
    iter = iter + 1
    thr1 = 1/(n*rho)
    lambda_ = lambda_2/(2*rho*n)
    
    old_w = w; old_A = A; old_obj = obj

    ## update w
    if(type == 1)
      w = as.vector(abess(X, 1 - A + u, family = "gaussian", lambda = lambda_, support.size = s, fit.intercept = FALSE,normalize = 0)$beta)
    else if(type==2)
      w = as.vector(bess.one(X, 1 - A + u, family = "gaussian", s = s, normalize = FALSE)$beta)
    else w = as.vector(bess_lm(X, 1 - A + u, s, lambda = lambda_))
    # cat("active set :", which(w!=0),'\n')
    # cat("w.norm: ",sum(w^2),'\n')

    Xw = 1 - X%*%w

    ## update A
    A = Xw + u

    for (i in 1:n) {
      if( thr1 < A[i] )
        A[i] = A[i] - thr1
      else{
        if(0 < A[i])
          A[i] = 0
      }

    }

    ## compute primal residual and save to hist_pres

    obj = mean(pmax(A, 0)) +  lambda_2/2 * sum(w^2)


    obj_error = abs(obj-old_obj) / old_obj

    res_p = sqrt(sum((Xw -A)^2))
    res_d = rho*sqrt(sum((t(X)%*%(A - old_A))^2))

    hist_obj[iter] = obj; hist_pres[iter] = res_p; hist_dres[iter] = res_d

    # if(cons1<=tol & cons2<=tol & obj_error < tol) break
    if(res_p<=tol && res_d < tol && obj_error<tol) break
    
    cat("rho:",rho,'\n')
    cat(iter,"obj:", obj, "res_p:", res_p, "res_d:", res_d, "nnz:",sum(w!=0), '\n')
    #cat(iter,"primal dual: ",pres, "dual residual:", dres,"obj_2",obj_2,'\n')

    ## update Lagrangian multiplier
    u = u + (Xw - A)
    ##change rho
    if( 10 * res_d < res_p) rho = rho * 2
      else if(res_d > 10*res_p) rho = rho/2
    
  }

  return(list(w=w, iter = iter, hist_obj = hist_obj))
}

# 
# plotfun <- function(x_train, y_train,x_test = NULL,y_test= NULL,max_supp){
#   accs = aucs = loss = matrix(NA, max_supp, 11)
#   for (j in -5:5) {
#   for (s in seq(max_supp)) {
#     res <- bess_svm(x_train, y_train, s, 0, lambda_1 = 10^j)
#     loss[s,j] = res$Loss
#     y_pred = x_test %*% res$w
#     accs[s,j] = accuracy(y_test, y_pred)
#     aucs[s,j] = auc(y_test, y_pred)
#   }
#   
#   }
#   return(list(accs = accs, aucs = aucs, loss =loss))
# }
# 

# svm_IHT <- function(X, y, s, step.size = 0.1, lambda_2  = 1e-3,
#                     eps= 1e-4, max_iter = 200,
#                     refitted = TRUE, verbose=FALSE){
# 
#   n <- dim(X)[1]
#   p <- dim(X)[2]
# 
#   # beta.star = c(1.39, 1.47, 1.56, 1.65, 1.74, rep(0,p-5))
#   # beta.star = c(0.011, -0.809, 0.216, 0.421, 0.626, 0.011, -0.809, 0.126, 0.421, 0.626, rep(0,p-10))
#   X_ = y*X
# 
#   check <- function(x, tau = 1)  tau - (x<=0)
#   # beta0 <- as.vector(LiblineaR(data$x, data$y, type=3, bias=0)$W)
#   beta0 <- rep(0,p)
# 
#   k <- 0
#   d <- 1
#   bd <- 0
#   beta <- beta0
# 
#   while(k < max_iter & d > eps){
# 
#     alpha <- drop(check(1 - X_ %*% beta))
#     
#     gt = - (t(X)%*%alpha)/n + lambda_2 * beta
# 
#     beta = hard(beta - step.size * gt, s)
#     
#     # bd <- beta - beta.star
#     # 
#     # d <- sum(bd^2)
#     # cat()
# 
#     k <- k + 1
#     beta0 <- beta
#     # alpha0 <- alpha
#   }
# 
#   # if(refitted == TRUE){
#   #   A <- which(beta0!=0)
#   #   beta <- rep(0, p)
#   #   beta[A] <- huberized_svm(as.matrix(X[,A]),y,lambda_2)
#   # }
# 
#   return(list(beta = beta, alpha = alpha, k=k))
# }


gendata<- function(n,p=500){
  v = c(0.2, -0.2, 0.3, 0.4, 0.5, 0.2, -0.2, 0.3, 0.4, 0.5, rep(0,p-10))
  
  Sig = matrix(0, p, p)
  Sig[1:10, 1:10] <- -0.2
  diag(Sig) <- 1
  
  y = sample(c(1,-1), n, replace = T)
  
  X = matrix(NA, n, p)
  for (i in 1:n) {
    if(y[i] == 1){
      X[i,] <- rnorm(p, v, Sig)
    } else{
      X[i,] <- rnorm(p, -v, Sig)
    }
  }
  
  beta.star = c(0.011, -0.809, 0.216, 0.421, 0.626, 0.011, -0.809, 0.126, 0.421, 0.626, rep(0,p-10))
  
  return(list(X = X, y=y, beta.star = beta.star))
}


GenSynthetic <- function(n, p, k, seed=1, rho=0, s=1, sigma=NULL, shuffle_B=FALSE) 
{
  set.seed(seed) 
  sig = matrix(0, p, p)
  for (i in 1:p) {
    for (j in 1:p) {
      sig[i,j] = sigma^(abs(i-j))
    }
  }
  
  X = mvrnorm(n, mu=rep(0, p), Sigma=sig)
  # if (is.null(sigma)){
  #   X = matrix(rnorm(n*p), n, p)
  # } else {
  #   if ((ncol(sigma) != p) || (nrow(sigma) != p)){
  #     stop("sigma must be a semi positive definite matrix of side length p")
  #   }
  #   X = mvrnorm(n, mu=rep(0, p), Sigma=sigma)
  # }
  
  X[abs(X) < rho] <- 0.
  B = c(rep(1,k),rep(0,p-k))
  
  if (shuffle_B){
    B = sample(B)
  }
  
  y = rbinom(n, 1, 1/(1 + exp(-s*X%*%B)))
  
  y[y==0] = -1
  return(list(X=X, B=B, y=y, s=s))
}


cv.logistic <- function(X, y, method=NULL,cost_list=NULL, nfolds=5,seed=123){
  
  set.seed(seed)
  
  n = nrow(X)
  if (is.null(cost_list)) cost_list = logspace(-7, 2, n=100)
  
  
  L = length(cost_list)
  score = rep(0, L)
  all.folds = createFolds(1:n, nfolds)
  
  for (i in 1:nfolds) {
    # cat("method:", method, i, "fold start \n")
    omit <- all.folds[[i]]
    for(j in 1:L){
      
      if(method == "l1"){
        fit = glmnet(X[-omit, ], y[-omit], family = "binomial", intercept=FALSE, lambda = cost_list[j], nlambda = 1)
        w <- as.vector(fit$beta)
        score[j] = score[j] +  accuracy(X[omit, ] %*% w, y[omit])
      }
        
      else if(method == "mcp"){
        fit.mcp = ncvreg(X[-omit, ], y[-omit], family = "binomial", penalty = "MCP", lambda = rep(cost_list[j],2), nlambda = 2)
        w <- fit.mcp$beta[,1]
        score[j] = score[j] + accuracy(X[omit, ] %*% w[-1] + w[1], y[omit])
      }
        
      else {
        fit.l0 = abess(X[-omit, ], y[-omit], family = "binomial", tune.type = "gic", fit.intercept = FALSE, normalize=0, support.size = cost_list[j])
        w <- as.vector(fit.l0$beta)
        score[j] = score[j] + accuracy(X[omit, ] %*% w, y[omit])
      }
      
      if(sum(w!=0)==0) break
    }
  }
  score = score / nfolds
  index = which.max(score)
  best.cost = cost_list[index]
  cat(method, "best_cost:", best.cost, '\n')
  return(list(best.cost = best.cost , score = score))
  
} 
