Sys.setenv("PKG_CXXFLAGS"="-w")  # 关闭编译器警告

source("./utils.R")

require(MASS)
require(ModelMetrics)
options(digits = 4)

normalized <- function(x){
  return(x/sqrt(sum(x^2)))
}

scale <- function(beta_hat,beta){
  
  index <- which.max(beta_hat)
  beta_hat = (beta[index]/beta_hat[index])*beta_hat
  return(beta_hat)
}

model1 <- function(n, p=100, seed=123){
  set.seed(seed)
  
  mu = c(0.1,0.2,0.3,0.4,0.5,rep(0,p-5))
  sigma = matrix(0, nrow = p, ncol = p)
  sigma[1:5,1:5] =-0.2
  diag(sigma) = 1
  
  Xp = matrix(mvrnorm(n/2,mu,sigma), n/2, p)
  Xf = matrix(mvrnorm(n/2,-mu,sigma), n/2, p)
  X = rbind(Xp,Xf)
  colnames(X)=paste0('X',1:ncol(X))
  y = c(rep(1,n/2),rep(-1,n/2))
  beta = solve(sigma) %*% mu
  Tbeta = beta/sqrt(sum(beta^2))
  return(list(x=X, y=y,Tbeta = Tbeta))
}


simulation = function(n, p, q = 5, rep = 100){
  filename = "log1_final.txt"
  cat('simulation', file = filename, append = TRUE)
  cat('n = ',n,' p = ', p, '\n', file = filename, append = TRUE)
  k = vector()
  
  acc = acco = acc0 = acc1 = accs = accm = acc2 = accl0learn = vector()
  Auc = Auco = Auc0 = Auc1 = Aucs = Aucm = Auc2 = Aucl0learn = vector()
  Pre = Preo = Pre0 = Pre1 = Pres = Prem = Pre2 = Prel0learn = vector() # precision
  Rec = Reco = Rec0 = Rec1 = Recs = Recm = Rec2 = Recl0learn = vector() # recall
  F1S = F1So = F1S0 = F1S1 = F1Ss = F1Sm = F1S2 = F1Sl0learn = vector() #F1 score
  
  TP0 = TP1 = TPs = TPm =TP2 = TPl0learn = vector()
  FP0 = FP1 = FPs = FPm = FP2 = FPl0learn = vector()
  
  bias.l0 = bias.l1 = bias.scad = bias.mcp = bias.l2 = numeric(rep)
  time0 = time1 = times = timem = time2 = timel2lossl1reg = timel1l2 = timel0learn = vector()
  
  Correct0 = Correct1 = Corrects = Correctm = Correct2 = Correctl0learn = numeric(rep)
  Overfit0 = Overfit1 = Overfits = Overfitm = Overfit2 = Overfitl0learn = numeric(rep)
  Underfit0 = Underfit1 = Underfits = Underfitm = Underfit2 = Underfitl0learn = numeric(rep)
  
  beta.l0 = beta.l1 = beta.scad = beta.mcp = beta.l2 = beta.l0learn = matrix(NA, p, rep)
  Y.l0 = Y.l1 = Y.scad = Y.mcp = Y.l2 =  Y.l0learn = matrix(NA, 10*n, rep)
  beta.star = c(1.39, 1.47, 1.56, 1.65, 1.74, rep(0,p-5))
  # scad_error = numeric(rep)
  
  Err = Erro = Err0 = Err1 = Errs = Errm = Err2 = Errl0learn = vector()
  Errnormal = Erronormal = Err0normal = Err1normal = Errsnormal = Errmnormal = Err2normal = Errl0learnnormal = vector()
  Errscale = Erroscale = Err0scale = Err1scale = Errsscale = Errmscale = Err2scale = Errl0learnscale = vector()
  
  for (i in 1:rep){
    set.seed(i)
    
    cat("runing ", i ,"-th iter \n")
    data = model1(n,p, seed= i) # train data
    test_data = model1(10*n,p, seed=i) #test data:10 times
    
    # # Bayses
    # y_hat = test_data$x %*% beta.star
    # acc[i] = accuracy(y_hat, test_data$y) # Bayes
    # Auc[i] = auc(test_data$y, y_hat)
    # Pre[i] = precision_comp(test_data$y, y_hat)
    # Rec[i] = recall_comp(test_data$y,y_hat)
    # F1S[i] = 2*Pre[i]*Rec[i]/(Pre[i] + Rec[i])
    # 
    # #oracle
    # beta_oracle = huberized_svm(data$x[,1:5], data$y, gamma =0)
    # y_hato = test_data$x[,1:5] %*%beta_oracle
    # acco[i] = accuracy(y_hato, test_data$y) # oracle
    # Auco[i] = auc(test_data$y, y_hato)
    # Preo[i] = precision_comp(test_data$y, y_hato)
    # Reco[i] = recall_comp(test_data$y, y_hato)
    # F1So[i] = 2*Preo[i]*Reco[i]/(Preo[i] + Reco[i])
    
    
    ## L2 norm lambda_2
    # lambda.2 = 1e-3
    # #l0 svm
    # t1 = Sys.time()
    # s_best = cv.l0(data$x, data$y, nfolds=5, s_max = 20,splicing_type = 2, lambda2 = lambda.2)$s_best
    # 
    # fit = bess_svm(data$x, data$y, s_best, beta_init = 0, lambda_2 = lambda.2)
    # 
    # t2 = Sys.time()
    # bess_beta = fit$w
    # 
    # supp_size = sum(bess_beta != 0)
    # active_set = which(bess_beta != 0)
    # 
    # if(supp_size==q & setequal(active_set, 1:q))
    #   Correct0[i] = 1
    # else if(supp_size > q)
    #   Overfit0[i] = 1
    # else if(supp_size < q)
    #   Underfit0[i] = 1
    # 
    # beta.l0[,i] = bess_beta
    # 
    # y_hat0 = drop(test_data$x %*% bess_beta)
    # Y.l0[,i] = y_hat0
    # acc0[i] = accuracy(y_hat0, test_data$y)
    # Auc0[i] = auc(test_data$y, y_hat0)
    # Pre0[i] = precision_comp(test_data$y,y_hat0)
    # Rec0[i] = recall_comp(test_data$y,y_hat0)
    # F1S0[i] = 2*Pre0[i]*Rec0[i]/(Pre0[i] + Rec0[i])
    # 
    # TP0[i]=sum(abs(bess_beta[1:5])>0)
    # FP0[i]=sum(abs(bess_beta[6:p])>0)
    # time0[i] = as.double(t2-t1, units = "secs")
    # Err0[i] = sum((bess_beta-beta.star)^2)
    # Err0normal[i] =sum((normalized(bess_beta)-normalized(beta.star))^2)
    # Err0scale[i] = sqrt(sum((scale(bess_beta,beta.star)-beta.star)^2))
    # 
    # l1 svm
  #   t1 = Sys.time()
  #   best.cost = cv.method(data$x, data$y, nfolds=5, method ="l1")$best.cost
  #   beta_lasso = svm.l1(data$x, data$y, lambda = best.cost)$w
  #   t2 = Sys.time()
  #   
  #   supp_size = sum(beta_lasso != 0)
  #   active_set = which(beta_lasso != 0)
  #   
  #   if(supp_size==q & setequal(active_set, 1:q))
  #     Correct1[i] = 1
  #   else if(supp_size > q)
  #     Overfit1[i] = 1
  #   else if(supp_size < q)
  #     Underfit1[i] = 1
  #   
  #   beta.l1[,i] = beta_lasso
  #   y_hat1 = drop(test_data$x %*% beta_lasso)
  #   Y.l1[,i] = y_hat1
  #   acc1[i] = accuracy(y_hat1, test_data$y)
  #   Auc1[i] = auc(test_data$y, y_hat1)
  #   Pre1[i] = precision_comp(test_data$y,y_hat1)
  #   Rec1[i] = recall_comp(test_data$y,y_hat1)
  #   F1S1[i] = 2*Pre1[i]*Rec1[i]/(Pre1[i] + Rec1[i])
  #   
  #   TP1[i]=sum(abs(beta_lasso[1:5])>0)
  #   FP1[i]=sum(abs(beta_lasso[6:p])>0)
  #   time1[i] = as.double(t2-t1, units = "secs")
  #   # error.l1[i] = sqrt(sum((beta_lasso-beta.star)^2))
  #   # error_normalized.l1[i] = sqrt(sum((normalized(beta_lasso)-normalized(beta.star))^2))
  #   # error_scale.l1[i] = sqrt(sum((scale(beta_lasso,beta.star)-beta.star)^2))
  #   
  #   # scad svm
  #   t1 = Sys.time()
  #   best.cost = cv.method(data$x, data$y, nfolds=5 ,method = "scad")$best.cost
  #   beta_scad = svm.scad(data$x, data$y, lambda = best.cost)$w
  #   t2 = Sys.time()
  #   
  #   supp_size = sum(beta_scad != 0)
  #   active_set = which(beta_scad != 0)
  #   
  #   if(supp_size==q & setequal(active_set, 1:q))
  #     Corrects[i] = 1
  #   else if(supp_size > q)
  #     Overfits[i] = 1
  #   else if(supp_size < q)
  #     Underfits[i] = 1
  #   
  #   beta.scad[,i] = beta_scad
  #   y_hats = drop(test_data$x %*% beta_scad)
  #   Y.scad[,i] = y_hats
  #   accs[i] = accuracy(y_hats, test_data$y)
  #   Aucs[i] = auc(test_data$y, y_hats)
  #   Pres[i] = precision_comp(test_data$y,y_hats)
  #   Recs[i] = recall_comp(test_data$y, y_hats)
  #   F1Ss[i] = 2*Pres[i]*Recs[i]/(Pres[i] + Recs[i])
  #   
  #   TPs[i]=sum(abs(beta_scad[1:5])>0)
  #   FPs[i]=sum(abs(beta_scad[6:p])>0)
  #   times[i] = as.double(t2-t1, units = "secs")
  #   # error.scad[i] = sqrt(sum((beta_scad-beta.star)^2))
  #   # error_normalized.scad[i] = sqrt(sum((normalized(beta_scad)-normalized(beta.star))^2))
  #   # error_scale.scad[i] = sqrt(sum((scale(beta_scad,beta.star)-beta.star)^2))
  #   
  #   # mcp svm
  #   t1 = Sys.time()
  #   best.cost = cv.method(data$x, data$y, nfolds=5 ,method = "mcp")$best.cost
  #   beta_mcp = svm.scad(data$x, data$y, lambda = best.cost)$w
  #   t2 = Sys.time()
  #   
  #   supp_size = sum(beta_mcp != 0)
  #   active_set = which(beta_mcp != 0)
  #   
  #   if(supp_size==q & setequal(active_set, 1:q))
  #     Correctm[i] = 1
  #   else if(supp_size > q)
  #     Overfitm[i] = 1
  #   else if(supp_size < q)
  #     Underfitm[i] = 1
  #   
  #   beta.mcp[,i] = beta_mcp
  #   y_hatm = drop(test_data$x %*% beta_mcp)
  #   Y.mcp[,i] = y_hatm
  #   accm[i] = accuracy(y_hatm, test_data$y)
  #   Aucm[i] = auc(test_data$y, y_hatm)
  #   Prem[i] = precision_comp(test_data$y,y_hatm)
  #   Recm[i] = recall_comp(test_data$y,y_hatm)
  #   F1Sm[i] = 2*Prem[i]*Recm[i]/(Prem[i] + Recm[i])
  #   
  #   TPm[i]=sum(abs(beta_mcp[1:5])>0)
  #   FPm[i]=sum(abs(beta_mcp[6:p])>0)
  #   timem[i] = as.double(t2-t1, units = "secs")
  #   
  #   # l2 svm
  #   t1 = Sys.time()
  #   beta.cost = cv.method(data$x, data$y, nfolds=5, method= "l2")
  #   beta_l2 <- drop(LiblineaR(data$x, data$y, type = 3, bias = 0, cost = best.cost)$W)
  #   t2 = Sys.time()
  #   
  #   supp_size = sum(beta_l2 != 0)
  #   active_set = which(beta_l2 != 0)
  #   
  #   if(supp_size==q & setequal(active_set, 1:q))
  #     Correct2[i] = 1
  #   else if(supp_size > q)
  #     Overfit2[i] = 1
  #   else if(supp_size < q)
  #     Underfit2[i] = 1
  #   
  #   beta.l2[,i] = beta_l2
  #   y_hat2 = drop(test_data$x %*% beta_l2)
  #   Y.l2[,i] = y_hat2
  #   acc2[i] = accuracy(y_hat2, test_data$y)
  #   Auc2[i] = auc(test_data$y, y_hat2)
  #   Pre2[i] = precision_comp(test_data$y,y_hat2)
  #   Rec2[i] = recall_comp(test_data$y,y_hat2)
  #   F1S2[i] = 2*Pre2[i]*Rec2[i]/(Pre2[i] + Rec2[i])
  #   
  #   TP2[i]=sum(abs(beta_l2[1:5])>0)
  #   FP2[i]=sum(abs(beta_l2[6:p])>0)
  #   time2[i] = as.double(t2-t1, units = "secs")
  #   # error.l2[i] = sqrt(sum((beta_l2-beta.star)^2))
  #   # error_normalized.l2[i] = sqrt(sum((normalized(beta_l2)-normalized(beta.star))^2))
  #   # error_scale.l2[i] = sqrt(sum((scale(beta_l2, beta.star)-beta.star)^2))
  #   
  #   
  #   
    #l0learn
    # t1 = Sys.time()
    # fitl0learn <- L0Learn.cvfit(data$x, data$y, loss = "SquaredHinge", penalty = "L0L2",algorithm = "CDPSI",
    #                             nGamma = 1, gammaMin = lambda.2, gammaMax = lambda.2,maxSwaps = 1,
    #                             intercept = FALSE, nFolds = 5)
    # # fitl0learn <- cv.method(data$x, data$y, nfolds=5, method= "l0learn")
    # t2 = Sys.time()
    # 
    # # fitl0learn$fit$beta[[1]][,index]
    # 
    # index <- which.min(fitl0learn$cvMeans[[1]])
    # lamb <- fitl0learn$fit$lambda[[1]][index]
    # betal0learn = fitl0learn$fit$beta[[1]][,index]
    # beta.l0learn[,i] = betal0learn
    # 
    # supp_size = sum(betal0learn != 0)
    # active_set = which(betal0learn != 0)
    # 
    # if(supp_size==q & setequal(active_set, 1:q))
    #   Correctl0learn[i] = 1
    # else if(supp_size > q)
    #   Overfitl0learn[i] = 1
    # else if(supp_size < q)
    #   Underfitl0learn[i] = 1
    # 
    # y_hatl0learn = drop(predict(fitl0learn, newx = test_data$x, lambda = lamb))
    # Y.l0learn[,i] = y_hatl0learn
    # 
    # accl0learn[i] <- accuracy(y_hatl0learn, test_data$y)
    # Aucl0learn[i] <- auc(test_data$y, y_hatl0learn)
    # Prel0learn[i] = precision_comp(test_data$y,y_hatl0learn)
    # Recl0learn[i] = recall_comp(test_data$y, y_hatl0learn)
    # F1Sl0learn[i] = 2*Prel0learn[i]*Recl0learn[i]/(Prel0learn[i] + Recl0learn[i])
    # 
    # TPl0learn[i]=sum(abs(betal0learn[1:5])>0)
    # FPl0learn[i]=sum(abs(betal0learn[6:p])>0)
    # timel0learn[i] = as.double(t2-t1, units = "secs")
    # Errl0learn[i] = sum((betal0learn-beta.star)^2)
    # Errl0learnnormal[i] =sum((normalized(betal0learn)-normalized(beta.star))^2)
    # Errl0learnscale[i] = sqrt(sum((scale(betal0learn,beta.star)-beta.star)^2))
    
    
    t1 = Sys.time()
    fit = cv.glmnet(data$x, data$y, family = "binomial", intercept=FALSE, type.measure = "class")
    
    t2 = Sys.time()
    
    index = fit$index[1]
    
    beta.l1[,i] = fit$glmnet.fit$beta[,index]
    supp_size = sum(fit$beta[,1]!=0)
    active_set = which(fit$beta[,1]!=0 )
    
    if(supp_size==q & setequal(active_set, 1:q))
      Correct1[i] = 1
    else if(supp_size > q)
      Overfit1[i] = 1
    else if(supp_size < q)
      Underfit1[i] = 1
    
    TP1[i]=sum(abs(beta.l1[,i][1:5])>0)
    FP1[i]=sum(abs(beta.l1[,i][6:p])>0)
    
    y_hat1 = predict(fit, newx = test_data$x, s = "lambda.min")
    acc1[i] = accuracy(test_data$y, y_hat1)
    Auc1[i] = auc(test_data$y, y_hat1)
    Pre1[i] = precision_comp(test_data$y, y_hat1)
    Rec1[i] = recall_comp(test_data$y,y_hat1)
    F1S1[i] = 2*Pre1[i]*Rec1[i]/(Pre1[i] + Rec1[i])
    
    time1[i] = as.double(t2-t1, units = "secs")
    
    ## MCP logistic
    t1 = Sys.time()
    
    fit.mcp = cv.ncvreg(data$x, data$y, family = "binomial",penalty = "MCP", nfolds=5)
    
    t2 = Sys.time()
    
    index = fit.mcp$min
    
    beta.mcp[,i] = coef(fit.mcp, lambda = fit.mcp$lambda.min)[-1]
    supp_size = sum(beta.mcp[,i]!=0)
    active_set = which(beta.mcp[,i]!=0)
    if(supp_size==q & setequal(active_set, 1:q))
      Correctm[i] = 1
    else if(supp_size > q)
      Overfitm[i] = 1
    else if(supp_size < q)
      Underfitm[i] = 1
    
    TPm[i]=sum(abs(beta.mcp[,i][1:5])>0)
    FPm[i]=sum(abs(beta.mcp[,i][6:p])>0)
    
    y_hatm = as.vector(predict(fit.mcp, X=test_data$x))
    accm[i] = accuracy(test_data$y, y_hatm)
    Aucm[i] = auc(test_data$y, y_hatm)
    Prem[i] = precision_comp(test_data$y, y_hatm)
    Recm[i] = recall_comp(test_data$y,y_hatm)
    F1Sm[i] = 2*Prem[i]*Recm[i]/(Prem[i] + Recm[i])
    
    timem[i] = as.double(t2-t1, units = "secs")
    
    ## L0 logistic
    cost_list = 1:20
    
    t1 = Sys.time()
    fit.l0 = abess(data$x, data$y, family = "binomial", tune.type = "cv", fit.intercept = FALSE, normalize=0, nfolds = 5)
    
    t2 = Sys.time()
    
    index = which.min(fit.l0$tune.value)
    beta.l0[,i] = fit.l0$beta[,index]
    
    supp_size = sum(beta.l0[,i]!=0)
    active_set = which(beta.l0[,i]!=0)
    if(supp_size==q & setequal(active_set, 1:q))
      Correctm[i] = 1
    else if(supp_size > q)
      Overfitm[i] = 1
    else if(supp_size < q)
      Underfitm[i] = 1
    
    TP0[i]=sum(abs(beta.l0[,i][1:5])>0)
    FP0[i]=sum(abs(beta.l0[,i][6:p])>0)
    
    y_hatl0 = predict(fit.l0, newx = test_data$x, support.size = NULL)
    acc0[i] = accuracy(test_data$y, y_hatl0)
    Auc0[i] = auc(test_data$y, y_hatl0)
    Pre0[i] = precision_comp(test_data$y, y_hatl0)
    Rec0[i] = recall_comp(test_data$y,y_hatl0)
    F1S0[i] = 2*Pre0[i]*Rec0[i]/(Pre0[i] + Rec0[i])
    
    time0[i] = as.double(t2-t1, units = "secs")
  }
  
  # cat('l0 error:')
  # cat(mean(error), sd(error),'\n')
  # cat(mean(error_normalized), sd(error_normalized),'\n')
  # 
  # cat('l1 error:')
  # cat(mean(error.l1), sd(error.l1),'\n')
  # cat(mean(error_normalized.l1), sd(error_normalized.l1),'\n')
  
  # cat('scad error:')
  # cat(mean(error.scad), sd(error.scad),'\n')
  # cat(mean(error_normalized.scad), sd(error_normalized.scad),'\n')
  
  # cat('l2 error:')
  # cat(mean(error.l2), sd(error.l2),'\n')
  # cat(mean(error_normalized.l2), sd(error_normalized.l2),'\n')
  
  cat('bess acc:',mean(acc0),'(',sd(acc0),')',  'l1 acc:',mean(acc1),'(',sd(acc1),')',
      'scad acc:',mean(accs),'(',sd(accs),')', 'mcp acc:',mean(accm),'(',sd(accm),')',
      'l2 acc:',mean(acc2),'(',sd(acc2),')', 'l0learn acc:', mean(accl0learn),'(',sd(accl0learn),')',
      'Bayes acc:',mean(acc),'(',sd(acc),')\n',file = filename, append = TRUE,sep = '')
      
  cat('bess auc:',mean(Auc0),'(',sd(Auc0),')',  'l1 auc:',mean(Auc1),'(',sd(Auc1),')',
      'scad auc:',mean(Aucs),'(',sd(Aucs),')', 'mcp auc:',mean(Aucm),'(',sd(Aucm),')',
      'l2 auc:',mean(Auc2),'(',sd(Auc2),')',  'l0learn auc:', mean(Aucl0learn),'(',sd(Aucl0learn),')',
      'Bayes acc:',mean(Auc),'(',sd(Auc),')\n',file = filename, append = TRUE,sep = '')
  
  cat('bess Pre:',mean(Pre0),'(',sd(Pre0),')',  'l1 Pre:',mean(Pre1),'(',sd(Pre1),')',
      'scad Pre:',mean(Pres),'(',sd(Pres),')', 'mcp Pre:',mean(Prem),'(',sd(Prem),')',
      'l2 Pre:',mean(Pre2),'(',sd(Pre2),')',  'l0learn Pre:', mean(Prel0learn),'(',sd(Prel0learn),')',
      'Bayes acc:',mean(Pre),'(',sd(Pre),')\n',file = filename, append = TRUE,sep = '')
  
  cat('bess Rec:',mean(Rec0),'(',sd(Rec0),')',  'l1 Rec:',mean(Rec1),'(',sd(Rec1),')',
      'scad Rec:',mean(Recs),'(',sd(Recs),')', 'mcp Rec:',mean(Recm),'(',sd(Recm),')',
      'l2 Rec:',mean(Rec2),'(',sd(Rec2),')',  'l0learn Rec:', mean(Recl0learn),'(',sd(Recl0learn),')',
      'Bayes acc:',mean(Rec),'(',sd(Rec),')\n',file = filename, append = TRUE,sep = '')
  
  cat('bess F1S:',mean(F1S0),'(',sd(F1S0),')',  'l1 F1S:',mean(F1S1),'(',sd(F1S1),')',
      'scad F1S:',mean(F1Ss),'(',sd(F1Ss),')', 'mcp F1S:',mean(F1Sm),'(',sd(F1Sm),')',
      'l2 F1S:',mean(F1S2),'(',sd(F1S2),')',  'l0learn F1S:', mean(F1Sl0learn),'(',sd(F1Sl0learn),')',
      'Bayes acc:',mean(F1S),'(',sd(F1S),')\n',file = filename, append = TRUE,sep = '')
  
  cat('bess TP:',mean(TP0),'(',sd(TP0),')', 'l1 TP:',mean(TP1),'(',sd(TP1),')',
      'scad TP:',mean(TPs),'(',sd(TPs),')', 'mcp TP:',mean(TPm),'(',sd(TPm),')',
      'l2 TP:',mean(TP2),'(',sd(TP2),')','l0learn TP:',mean(TPl0learn),'(',sd(TPl0learn),')\n',
      file = filename, append = TRUE,sep = '')
  
  cat('bess FP:',mean(FP0),'(',sd(FP0),')', 'l1 FP:',mean(FP1),'(',sd(FP1),')',
      'scad FP:',mean(FPs),'(',sd(FPs),')', 'mcp FP:',mean(FPm),'(',sd(FPm),')',
      'l2 FP:',mean(FP2),'(',sd(FP2),')', 'l0learn FP:',mean(FPl0learn),'(',sd(FPl0learn),')\n',
      file = filename, append = TRUE,sep = '')
  
  cat('bess time:',mean(time0),'(',sd(time0),')', 'l1 time:',mean(time1),'(',sd(time1),')',
      'scad time:',mean(times),'(',sd(times),')', 'mcp time:',mean(timem),'(',sd(timem),')',
      'l2 time:',mean(time2),'(',sd(time2),')', 'l0learn time:',mean(timel0learn),'(',sd(timel0learn),')\n',
      file = filename, append = TRUE,sep = '')
  
  cat('bess Correct:',sum(Correct0), 'l1 Correct:',sum(Correct1),
      'scad Correct:',sum(Corrects), 'mcp Correct:',sum(Correctm),
      'l2 Correct:',sum(Correct2), 'l0learn Correct:',sum(Correctl0learn),')\n',
      file = filename, append = TRUE,sep = '')
  
  cat('bess Overfit:',sum(Overfit0), 'l1 Overfit:',sum(Overfit1),
      'scad Overfit:',sum(Overfits), 'mcp Overfit:',sum(Overfitm),
      'l2 Overfit:',sum(Overfit2), 'l0learn Overfit:',sum(Overfitl0learn),')\n',
      file = filename, append = TRUE,sep = '')
  
  cat('bess Underfit:',sum(Underfit0), 'l1 Underfit:',sum(Underfit1),
      'scad Underfit:',sum(Underfits), 'mcp Underfit:',sum(Underfitm),
      'l2 Underfit:',sum(Underfit2), 'l0learn Underfit:',sum(Underfitl0learn),')\n',
      file = filename, append = TRUE,sep = '')

  
  cat("\n", file = filename, append = TRUE)
  
  
  data <- list(acc = acc, acc.orcal = acco, acc.l0 = acc0, acc.l1 = acc1, acc.scad = accs, acc.mcp = accm, acc.l2 = acc2, acc.l0learn = accl0learn,
               Auc = Auc, Auc.orcal = Auco, Auc.l0 = Auc0, Auc.l1 = Auc1, Auc.scad = Aucs, Auc.mcp = Aucm, Auc.l2 = Auc2, Auc.l0learn = Aucl0learn,
               Pre = Pre, Pre.orcal = Preo, Pre.l0=Pre0, Pre.l1=Pre1, Pre.scad=Pres, Pre.mcp=Prem, Pre.l2=Pre2, Pre.l0learn=Prel0learn,
               Rec = Rec, Rec.orcal = Reco, Rec.l0=Rec0, Rec.l1=Rec1, Rec.scad=Recs, Rec.mcp=Recm, Rec.l2=Rec2, Rec.l0learn=Recl0learn,
               F1S = F1S, F1S.orcal = F1So, F1S.l0=F1S0, F1S.l1=F1S1, F1S.scad=F1Ss, F1S.mcp=F1Sm, F1S.l2=F1S2, F1S.l0learn=F1Sl0learn,
               TP.l0 = TP0, TP.l1 = TP1, TP.scad = TPs, TP.mcp = TPm, TP.l2 = TP2, TP.l0learn = TPl0learn,
               FP.l0 = FP0, FP.l1 = FP1, FP.scad = FPs, FP.mcp = FPm, FP.l2 = TP2, FP.l0learn = FPl0learn,
               time.l0=time0, time.l1=time1, time.scad=times, time.mcp=timem,time.l2=time2, time.l0learn=timel0learn,
               Correct.l0=Correct0, Correct.l1=Correct1, Correct.scad=Corrects, Correct.mcp=Correctm, Correct.l2=Correct2, Correct.l0learn=Correctl0learn,
               Overfit.l0=Overfit0, Overfit.l1=Overfit1, Overfit.scad=Overfits, Overfit.mcp=Overfitm, Overfit.l2=Overfit2, Overfit.l0learn=Overfitl0learn,
               Underfit.l0=Underfit0, Underfit.l1=Underfit1, Underfit.scad=Underfits, Underfit.mcp=Underfitm, Underfit.l2=Underfit2, Underfit.l0learn=Underfitl0learn
               )
  
  Accuracy = paste0( c( mean(acc0), mean(acc1), mean(accs), mean(accm), mean(acc2), mean(accl0learn), mean(acc), mean(acco)),
                     '(',
                     c( sd(acc0), sd(acc1), sd(accs), sd(accm), sd(acc2), sd(accl0learn), sd(acc), sd(acco)),
                     ')'
                     )
  AUC = paste0( c( mean(Auc0), mean(Auc1), mean(Aucs), mean(Aucm), mean(Auc2), mean(Aucl0learn), mean(Auc), mean(Auco)),
                '(',
                c( sd(Auc0), sd(Auc1), sd(Aucs), sd(Aucm), sd(Auc2), sd(Aucl0learn), sd(Auc), sd(Auco)),
                ')'
                )
  
  Precision = paste0( c( mean(Pre0,na.rm = TRUE), mean(Pre1,na.rm = TRUE), mean(Pres,na.rm = TRUE), mean(Prem,na.rm = TRUE), mean(Pre2,na.rm = TRUE), mean(Prel0learn,na.rm = TRUE), mean(Pre,na.rm = TRUE), mean(Preo,na.rm = TRUE)),
                     '(',
                     c( sd(Pre0,na.rm = TRUE), sd(Pre1,na.rm = TRUE), sd(Pres,na.rm = TRUE), sd(Prem,na.rm = TRUE), sd(Pre2,na.rm = TRUE), sd(Prel0learn,na.rm = TRUE), sd(Pre,na.rm = TRUE), sd(Preo,na.rm = TRUE)),
                     ')')
  Recall = paste0( c( mean(Rec0,na.rm = TRUE), mean(Rec1,na.rm = TRUE), mean(Recs,na.rm = TRUE), mean(Recm,na.rm = TRUE), mean(Rec2,na.rm = TRUE), mean(Recl0learn,na.rm = TRUE), mean(Rec,na.rm = TRUE), mean(Reco,na.rm = TRUE)),
                '(',
                c( sd(Rec0,na.rm = TRUE), sd(Rec1,na.rm = TRUE), sd(Recs,na.rm = TRUE), sd(Recm,na.rm = TRUE), sd(Rec2,na.rm = TRUE), sd(Recl0learn,na.rm = TRUE), sd(Rec,na.rm = TRUE), sd(Reco,na.rm = TRUE)),
                ')')
  F1Score = paste0( c( mean(F1S0,na.rm = TRUE), mean(F1S1,na.rm = TRUE), mean(F1Ss,na.rm = TRUE), mean(F1Sm,na.rm = TRUE), mean(F1S2,na.rm = TRUE), mean(F1Sl0learn,na.rm = TRUE),mean(F1S,na.rm = TRUE),mean(F1So,na.rm = TRUE)),
                '(',
                c( sd(F1S0,na.rm = TRUE), sd(F1S1,na.rm = TRUE), sd(F1Ss,na.rm = TRUE), sd(F1Sm,na.rm = TRUE), sd(F1S2,na.rm = TRUE), sd(F1Sl0learn,na.rm = TRUE),sd(F1S,na.rm = TRUE),sd(F1So,na.rm = TRUE)),
                ')')
                
  TP = paste0( c( mean(TP0), mean(TP1), mean(TPs), mean(TPm), mean(TP2), mean(TPl0learn),"",""),
               '(',
               c( sd(TP0), sd(TP1), sd(TPs), sd(TPm), sd(TP2), sd(TPl0learn),"",""),
               ')'
               )
  
  FP = paste0( c( mean(FP0), mean(FP1), mean(FPs), mean(FPm), mean(FP2), mean(FPl0learn),"",""),
               '(',
               c( sd(FP0), sd(FP1), sd(FPs), sd(FPm), sd(FP2), sd(FPl0learn),"",""),
               ')'
               )
  
  time = paste0( c( mean(time0), mean(time1), mean(times), mean(timem), mean(time2), mean(timel0learn), "",""),
                 '(',
                 c( sd(time0), sd(time1), sd(times), sd(timem), sd(time2), sd(timel0learn), "",""),
                 ')'
                 )
  Err = paste0(c( mean(Err0), mean(Err1), mean(Errs), mean(Errm), mean(Err2), mean(Errl0learn), "",""),
          '(',
          c( sd(Err0), sd(Err1), sd(Errs), sd(Errm), sd(Err2), sd(Errl0learn), "",""),
          ')')
  
  Errnormal = paste0(c( mean(Err0normal), mean(Err1normal), mean(Errsnormal), mean(Errmnormal), mean(Err2normal), mean(Errl0learnnormal), "",""),
          '(',
          c( sd(Err0normal), sd(Err1normal), sd(Errsnormal), sd(Errmnormal), sd(Err2normal), sd(Errl0learnnormal), "",""),
          ')')
          
  Errscale = paste0(c( mean(Err0scale), mean(Err1scale), mean(Errsscale), mean(Errmscale), mean(Err2scale), mean(Errl0learnscale), "",""),
          '(',
          c( sd(Err0scale), sd(Err1scale), sd(Errsscale), sd(Errmscale), sd(Err2scale), sd(Errl0learnscale), "",""),
          ')')
  
  Correct = c(sum(Correct0), sum(Correct1), sum(Corrects), sum(Correctm), sum(Correct2), sum(Correctl0learn), "","")
  
  Overfit = c(sum(Overfit0), sum(Overfit1), sum(Overfits), sum(Overfitm), sum(Overfit2), sum(Overfitl0learn), "","")
  
  Underfit = c(sum(Underfit0), sum(Underfit1), sum(Underfits), sum(Underfitm), sum(Underfit2), sum(Underfitl0learn), "","")
  
  Method = c("bess svm", "l1 svm", "scad svm", "mcp svm", "l2 svm", " l0learn", "Bayes", "orcal")
  
  simu_result <- data.frame(Method, Accuracy, AUC, Precision, Recall, F1Score, TP, FP, time, Err, Errnormal, Errscale, Correct, Overfit, Underfit)
  
  
  # save(data, file = paste0("12month_simu_",n,"_",p,".Rdata"))
  
  write.csv(simu_result, file = paste0("logistic_simu_",n,"_", p, "_result.csv"))
  
  return(simu_result)
}
# 

simulation(100,100)
# save(simu_100_100, file = "simu_100_100.Rdata")

# write.csv(simu_100_100_result, "simu_100_100_result.csv")
# 
simulation(100,1000)
# save(simu_100_1000, file = "simu_100_1000.Rdata")
# write.csv(simu_100_1000_result, "simu_100_1000_result.csv")

simulation(100,1500)
# save(simu_100_1500, file = "simu_100_1500.Rdata")
# write.csv(simu_100_1500_result, "simu_100_1500_result.csv")

simulation(200,1500)
# save(simu_200_1500, file = "simu_200_1500.Rdata")
# write.csv(simu_200_1500_result, "simu_200_1500_result.csv")

simulation(200,2000)
# save(simu_200_2000, file = "simu_200_2000.Rdata")
# write.csv(simu_200_2000_result, "simu_200_2000_result.csv")

