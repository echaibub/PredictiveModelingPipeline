
library(glmnet)
library(nnet)
library(randomForest)
library(gbm)
library(gam)
library(nnls)
library(FNN)


TransformResp <- function(x, n.questions) {
  max.score <- n.questions * 4
  floor(x * 100/max.score)
}



SplitData <- function(dat, n.splits) {
  split.id <- sample(rep(seq(n.splits), length = nrow(dat)))
  test.id <- which(split.id == 1)
  train.id <- which(split.id != 1)
  list(train.dat = dat[train.id,],
       test.dat = dat[test.id,],
       train.id = train.id,
       test.id = test.id)
}


## rbnit stands for "rank-based normal inverse transformation"
TransformDataMatrix <- function(dat, trans.type = c("none", "scale", "rbnit"), 
                                trans.resp = TRUE) {
  if (trans.type != "none") {
    if (trans.type == "scale") {
      if (trans.resp) {
        tdat <- scale(dat)
      }
      else {
        resp <- dat[, 1]
        aux <- scale(dat[, -1])
        tdat <- cbind(resp, aux)
      }
    }
    if (trans.type == "rbnit") {
      if (trans.resp) {      
        tdat <- apply(as.matrix(dat), 2, NormalTrans)
      }
      else {
        resp <- dat[, 1]
        aux <- apply(as.matrix(dat[, -1]), 2, NormalTrans)
        tdat <- cbind(resp, aux)
      }
    }
  }
  else {
    tdat <- dat
  }
  tdat
}



NormalTrans <- function(x) {
  n <- sum(!is.na(x))
  r <- rank(x, na.last = "keep")
  qnorm((r - 0.5)/n)
}


BaselineFitRuns <- function(dat, 
                  split.seeds,
                  n.runs = 100,
                  n.splits,
                  trans.type = "none",
                  error.type = c("mse", "mae", "wmse", "wmae"),
                  subject.ids,
                  trans.resp = TRUE) {
  Pred <- vector(mode = "list", length = n.runs)
  Pred.train <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    fit <- BaselineFit(aux$train.dat, aux$test.dat, trans.type, error.type, 
                       subject.ids[aux$test.id], trans.resp)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test)
}


BaselineFit <- function(train.dat,
                         test.dat,
                         trans.type = "none",
                         error.type = c("mse", "mae", "wmse", "wmae"),
                         subject.ids,
                         trans.resp = TRUE) {
  train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
  test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
  X.test <- test.dat[, -1, drop = FALSE]
  n.test <- nrow(X.test)
  Y.test <- test.dat[, "resp"]
  Pred <- rep(mean(train.dat[, "resp"]), n.test)
  pred.weights <- 
    switch(error.type, 
           mse = rep(1/n.test, n.test),
           mae = rep(1/n.test, n.test),
           wmse = ComputeWeights(subject.ids),
           wmae = ComputeWeights(subject.ids))
  error <- 
    switch(error.type,
           mse = sum(pred.weights * (Y.test - Pred)^2),
           mae = sum(pred.weights * abs(Y.test - Pred)),
           wmse = sum(pred.weights * ((Y.test - Pred)^2)),
           wmae = sum(pred.weights * abs(Y.test - Pred)))
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test)
}



## simple linear model fit
##
LmFitRuns <- function(dat, 
                      split.seeds,
                      n.runs = 100,
                      n.splits = 5,
                      trans.type = "none",
                      error.type = c("mse", "mae", "wmse", "wmae"),
                      subject.ids,
                      trans.resp = FALSE,
                      stacking.output = FALSE,
                      n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    fit <- LmFit(aux$train.dat, aux$test.dat, trans.type, error.type, 
                 subject.ids[aux$test.id], trans.resp, stacking.output)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    stacking.pred[[i]] <- fit$stacking.pred
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.pred = stacking.pred)
}



LmFit <- function(train.dat, 
                   test.dat,
                   trans.type = "none",
                   error.type = c("mse", "mae", "wmse", "wmae"),
                   subject.ids,
                   trans.resp = FALSE,
                   stacking.output = FALSE,
                   n.folds.stacking = 5,
                   stacking.seed = NULL) {
  if (!stacking.output) {
    stacking.pred <- NULL   
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, "resp"]
    X.test <- test.dat[, -1, drop = FALSE]
    fit <- lm(resp ~ ., train.dat)
    Pred <- predict(fit, newdata = X.test)
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(subject.ids),
             wmae = ComputeWeights(subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))  
  }
  else {   
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, "resp"]
    X.test <- test.dat[, -1, drop = FALSE]
    fit <- lm(resp ~ ., train.dat)
    Pred <- predict(fit, newdata = X.test)
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(subject.ids),
             wmae = ComputeWeights(subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      sfit <- lm(resp ~ ., train.dat[!sid, ])
      stacking.pred[sid] <- predict(sfit, newdata = train.dat[sid, -1])       
    }
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.pred = stacking.pred)
}



## simple linear model fit with step-wise variable selection
##
StepLmFitRuns <- function(dat, 
                          split.seeds,
                          n.runs = 100,
                          n.splits = 5,
                          trans.type = "none",
                          error.type = c("mse", "mae", "wmse", "wmae"),
                          subject.ids,
                          trans.resp = FALSE,                  
                          stacking.output = FALSE,
                          n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  sel.features <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    fit <- StepLmFit(aux$train.dat, aux$test.dat, trans.type, error.type, 
                     subject.ids[aux$test.id], trans.resp, stacking.output, 
                     n.folds.stacking)
    Y.test[[i]] <- fit$Y.test
    sel.features[[i]] <- fit$sel.features
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    stacking.pred[[i]] <- fit$stacking.pred
  } 
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)
}



StepLmFit <- function(train.dat,
                       test.dat,
                       trans.type = "none",
                       error.type = c("mse", "mae", "wmse", "wmae"),
                       subject.ids,
                       trans.resp = FALSE,                  
                       stacking.output = FALSE,
                       n.folds.stacking = 5,
                       stacking.seed = NULL) {
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, "resp"]
    X.test <- test.dat[, -1]
    fit <- lm(resp ~ ., train.dat)
    fit <- step(fit, k = log(nrow(train.dat)), trace = 0)
    sel.features <- attributes(fit$terms)$term.labels
    Pred <- predict(fit, newdata = X.test)
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(subject.ids),
             wmae = ComputeWeights(subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
  }
  else {
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, "resp"]
    X.test <- test.dat[, -1]
    fit <- lm(resp ~ ., train.dat)
    fit <- step(fit, k = log(nrow(train.dat)), trace = 0)
    sel.features <- attributes(fit$terms)$term.labels
    Pred <- predict(fit, newdata = X.test)
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(subject.ids),
             wmae = ComputeWeights(subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      sfit <- lm(resp ~ ., train.dat[!sid, ])
      sfit <- step(sfit, k = log(nrow(train.dat[!sid, ])), trace = 0)
      stacking.pred[sid] <- predict(sfit, newdata = train.dat[sid, -1])       
    }                
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)
}



GlmnetFitRuns <- function(dat, 
                          split.seeds,
                          n.runs = 100,
                          n.splits = 5,
                          n.folds = 10,
                          alpha = 1,
                          trans.type = "none",
                          error.type = c("mse", "mae", "wmse", "wmae"),
                          subject.ids,
                          trans.resp = FALSE,
                          stacking.output = FALSE,
                          n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  sel.features <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  feat.names <- names(dat)[-1]
  
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)   
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- GlmnetFit(aux$train.dat, aux$test.dat, n.folds, alpha, trans.type, error.type,
                     train.subject.ids, test.subject.ids, trans.resp, stacking.output, 
                     n.folds.stacking)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    sel.features[[i]] <- fit$sel.features
    error[i] <- fit$ME
    stacking.pred[[i]] <- fit$stacking.pred         
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       Pred = Pred, 
       Y.test = Y.test,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)  
}



GlmnetFit <- function(train.dat, 
                       test.dat,
                       n.folds = 10,
                       alpha = 1,
                       trans.type = "none",
                       error.type = c("mse", "mae", "wmse", "wmae"),
                       train.subject.ids,
                       test.subject.ids,
                       trans.resp = FALSE,
                       stacking.output = FALSE,
                       n.folds.stacking = 5,
                       stacking.seed = NULL,
                       cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }
  feat.names <- names(train.dat)[-1]
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    y.train <- train.dat[, "resp"]
    X.train <- as.matrix(train.dat[, -1])
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    X.test <- as.matrix(test.dat[, -1])    
    n.train <- nrow(X.train)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids)) 
    type.measure <- 
      switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae")   
    cv.fit <- cv.glmnet(X.train, y.train, nfolds = n.folds, alpha = alpha, 
                        weights = train.weights, type.measure = type.measure)
    Pred <- predict(cv.fit, newx = X.test, s = cv.fit$lambda.min)
    j <- which(cv.fit$lambda == cv.fit$lambda.min)
    betas <- cv.fit$glmnet.fit$beta[, j]
    sel.features <- feat.names[betas != 0]
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))      
  }
  else {
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    y.train <- train.dat[, "resp"]
    X.train <- as.matrix(train.dat[, -1])
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    X.test <- as.matrix(test.dat[, -1])    
    n.train <- nrow(X.train)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids)) 
    type.measure <- 
      switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae")   
    cv.fit <- cv.glmnet(X.train, y.train, nfolds = n.folds, alpha = alpha, 
                        weights = train.weights, type.measure = type.measure)
    Pred <- predict(cv.fit, newx = X.test, s = cv.fit$lambda.min)
    j <- which(cv.fit$lambda == cv.fit$lambda.min)
    betas <- cv.fit$glmnet.fit$beta[, j]
    sel.features <- feat.names[betas != 0]
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      s.n.train <- nrow(X.train[!sid,])
      s.train.weights <- 
        switch(error.type, 
               mse = rep(1/s.n.train, s.n.train),
               mae = rep(1/s.n.train, s.n.train),
               wmse = ComputeWeights(train.subject.ids[!sid]),
               wmae = ComputeWeights(train.subject.ids[!sid])) 
      s.cv.fit <- cv.glmnet(X.train[!sid, ], y.train[!sid], nfolds = n.folds, 
                            alpha = alpha, weights = s.train.weights, 
                            type.measure = type.measure)
      stacking.pred[sid] <- predict(s.cv.fit, newx = X.train[sid,], 
                                    s = s.cv.fit$lambda.min) 
    }                       
  }
  list(ME = error, 
       Pred = Pred, 
       Y.test = Y.test,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)  
}



EnetFitRuns <- function(dat, 
                        split.seeds,
                        n.runs = 100,
                        n.splits,
                        n.folds = 10,
                        alpha.grid,
                        trans.type = "none",
                        error.type = c("mse", "mae", "wmse", "wmae"),
                        subject.ids,
                        trans.resp = FALSE,
                        stacking.output = FALSE,
                        n.folds.stacking = 5) {
  n.alpha <- length(alpha.grid)
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  sel.features <- vector(mode = "list", length = n.runs)
  best.alpha <- rep(NA, n.runs)
  error <- rep(NA, n.runs)
  feat.names <- names(dat)[-1]
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- EnetFit(aux$train.dat, aux$test.dat, n.folds, alpha.grid, trans.type, error.type,
                   train.subject.ids, test.subject.ids, trans.resp, stacking.output, 
                   n.folds.stacking)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    sel.features[[i]] <- fit$sel.features
    stacking.pred[[i]] <- fit$stacking.pred
    best.alpha[i] <- fit$best.alpha 
  } 
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       Pred = Pred, 
       Y.test = Y.test, 
       best.alpha = best.alpha,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)  
}



EnetFit <- function(train.dat,
                     test.dat,
                     n.folds = 10,
                     alpha.grid,
                     trans.type = "none",
                     error.type = c("mse", "mae", "wmse", "wmae"),
                     train.subject.ids,
                     test.subject.ids,
                     trans.resp = FALSE,
                     stacking.output = FALSE,
                     n.folds.stacking = 5,
                     stacking.seed = NULL,
                     cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }
  n.alpha <- length(alpha.grid)
  feat.names <- names(train.dat)[-1]
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    y.train <- train.dat[, "resp"]
    X.train <- as.matrix(train.dat[, -1])
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    X.test <- as.matrix(test.dat[, -1])    
    n.train <- nrow(X.train)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids))
    type.measure <- 
      switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae")    
    lambs <- rep(NA, n.alpha)
    alpha.error <- rep(NA, n.alpha)
    for (j in seq(n.alpha)) {
      cv.fit <- cv.glmnet(X.train, y.train, nfolds = n.folds, alpha = alpha.grid[j], 
                          weights = train.weights, type.measure = type.measure)
      lambs[j] <- cv.fit$lambda.min
      alpha.error[j] <- cv.fit$cvm[which(cv.fit$lambda == cv.fit$lambda.min)]
    }
    best <- which.min(alpha.error)
    best.alpha <- alpha.grid[best]
    best.lambda <- lambs[best]
    best.fit <- glmnet(X.train, y.train, alpha = best.alpha, lambda = best.lambda) 
    Pred <- predict(best.fit, newx = X.test, s = best.lambda)
    betas <- best.fit$beta[, 1]
    sel.features <- feat.names[betas != 0]
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))    
  }
  else {
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    y.train <- train.dat[, "resp"]
    X.train <- as.matrix(train.dat[, -1])
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    X.test <- as.matrix(test.dat[, -1])    
    n.train <- nrow(X.train)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids))
    type.measure <- 
      switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae")    
    lambs <- rep(NA, n.alpha)
    alpha.error <- rep(NA, n.alpha)
    for (j in seq(n.alpha)) {
      cv.fit <- cv.glmnet(X.train, y.train, nfolds = n.folds, alpha = alpha.grid[j], 
                          weights = train.weights, type.measure = type.measure)
      lambs[j] <- cv.fit$lambda.min
      alpha.error[j] <- cv.fit$cvm[which(cv.fit$lambda == cv.fit$lambda.min)]
    }
    best <- which.min(alpha.error)
    best.alpha <- alpha.grid[best]
    best.lambda <- lambs[best]
    best.fit <- glmnet(X.train, y.train, alpha = best.alpha, lambda = best.lambda) 
    Pred <- predict(best.fit, newx = X.test, s = best.lambda)
    betas <- best.fit$beta[, 1]
    sel.features <- feat.names[betas != 0]
    n.test <- nrow(X.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }   
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      s.n.train <- nrow(X.train[!sid,])
      s.train.weights <- 
        switch(error.type, 
               mse = rep(1/s.n.train, s.n.train),
               mae = rep(1/s.n.train, s.n.train),
               wmse = ComputeWeights(train.subject.ids[!sid]),
               wmae = ComputeWeights(train.subject.ids[!sid]))        
      lambs <- rep(NA, n.alpha)
      alpha.error <- rep(NA, n.alpha)
      for (k in seq(n.alpha)) {
        cv.fit <- cv.glmnet(X.train[!sid,], y.train[!sid], nfolds = n.folds, 
                            alpha = alpha.grid[j], weights = s.train.weights, 
                            type.measure = type.measure)
        lambs[k] <- cv.fit$lambda.min
        alpha.error[k] <- cv.fit$cvm[which(cv.fit$lambda == cv.fit$lambda.min)]
      }
      best <- which.min(alpha.error)
      best.lambda <- lambs[best]
      best.fit <- glmnet(X.train[!sid,], y.train[!sid], alpha = alpha.grid[best], 
                         lambda = best.lambda) 
      stacking.pred[sid] <- predict(best.fit, newx = X.train[sid,], s = best.lambda)
    }                    
  }
  list(ME = error, 
       Pred = Pred, 
       Y.test = Y.test, 
       best.alpha = best.alpha,
       sel.features = sel.features, 
       stacking.pred = stacking.pred)  
}





CvKnnRegr <- function(dat, 
                  n.folds, 
                  k.grid, 
                  cv.measure = c("mse", "mae", "wmse", "wmae"),
                  subject.ids) {
  foldid <- sample(rep(seq(n.folds), length = nrow(dat)))
  n.pars <- length(k.grid)
  Error <- matrix(NA, n.folds, n.pars)
  for (i in seq(n.folds)) {
    #cat("cv fold ", i, "\n")
    idx <- foldid == i
    train.dat <- dat[!idx,]
    val.dat <- dat[idx,]
    n.val <- length(idx)
    pred.weights <- 
      switch(cv.measure, 
             mse = rep(1/n.val, n.val),
             mae = rep(1/n.val, n.val),
             wmse = ComputeWeights(subject.ids[idx]),
             wmae = ComputeWeights(subject.ids[idx]))    
    for (k in seq(n.pars)) {
      fit <- knn.reg(train = train.dat[, -1], test = val.dat[, -1], y = train.dat[, 1],
                     k = k.grid[k])
      pred <- fit$pred
      Error[i, k] <- 
        switch(cv.measure,
               mse = sum(pred.weights * (val.dat[, 1] - pred)^2),
               mae = sum(pred.weights * abs(val.dat[, 1] - pred)),
               wmse = sum(pred.weights * (val.dat[, 1] - pred)^2),
               wmae = sum(pred.weights * abs(val.dat[, 1] - pred)))
    }
  }
  error <- apply(Error, 2, mean)
  out <- cbind(k.grid, error, apply(Error, 2, sd))
  colnames(out) <- c("k", "ME mean", "ME sd")
  best <- which.min(error)
  list(out = out, k = k.grid[best], Error = Error)
}



KnnRegrFitRuns <- function(dat, 
                           split.seeds,
                           n.runs = 100,
                           n.splits,
                           n.folds = 10,
                           k.grid,
                           trans.type = "none",
                           error.type = c("mse", "mae", "wmse", "wmae"),
                           subject.ids,
                           trans.resp = FALSE,
                           stacking.output = FALSE,
                           n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  sel.features <- vector(mode = "list", length = n.runs)
  best.k <- rep(NA, n.runs)
  error <- rep(NA, n.runs)
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- KnnRegrFit(aux$train.dat, aux$test.dat, n.folds, k.grid, trans.type, error.type,
                      train.subject.ids, test.subject.ids, trans.resp, stacking.output, 
                      n.folds.stacking)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    best.k[i] <- fit$best.k
    stacking.pred[[i]] <- fit$stacking.pred  
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       best.k = best.k, 
       stacking.pred = stacking.pred)  
}



KnnRegrFit <- function(train.dat, 
                        test.dat,
                        n.folds = 10,
                        k.grid,
                        trans.type = "none",
                        error.type = c("mse", "mae", "wmse", "wmae"),
                        train.subject.ids,
                        test.subject.ids,
                        trans.resp = FALSE,
                        stacking.output = FALSE,
                        n.folds.stacking = 5,
                        stacking.seed = NULL,
                        cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    cv.fit <- CvKnnRegr(train.dat, n.folds, k.grid, error.type, train.subject.ids)
    fit <- knn.reg(train = train.dat[, -1], test = test.dat[, -1], y = train.dat[, 1],
                   k = cv.fit$k)
    Pred <- fit$pred
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    best.k <- cv.fit$k    
  }
  else {
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]  
    cv.fit <- CvKnnRegr(train.dat, n.folds, k.grid, error.type, train.subject.ids)
    fit <- knn.reg(train = train.dat[, -1], test = test.dat[, -1], y = train.dat[, 1],
                   k = cv.fit$k)
    Pred <- fit$pred
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    best.k <- cv.fit$k
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }  
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      s.cv.fit <- CvKnnRegr(train.dat[!sid,], n.folds, k.grid, error.type, 
                        train.subject.ids[!sid])
      s.fit <- knn.reg(train = train.dat[!sid, -1], test = train.dat[sid, -1], 
                       y = train.dat[!sid, 1], k = s.cv.fit$k)
      stacking.pred[sid] <- s.fit$pred      
    }                         
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       best.k = best.k, 
       stacking.pred = stacking.pred)  
}



CvRandomForest <- function(dat, 
                           n.folds,
                           n.trees,
                           m.grid,
                           cv.measure = c("mse", "mae", "wmse", "wmae"),
                           subject.ids) {
  foldid <- sample(rep(seq(n.folds), length = nrow(dat)))
  n.pars <- length(m.grid)
  Error <- matrix(NA, n.folds, n.pars)
  for (i in seq(n.folds)) {
    cat("cv fold ", i, "\n")
    idx <- foldid == i
    train.dat <- dat[!idx,]
    val.dat <- dat[idx,]
    n.val <- length(idx)
    pred.weights <- 
      switch(cv.measure, 
             mse = rep(1/n.val, n.val),
             mae = rep(1/n.val, n.val),
             wmse = ComputeWeights(subject.ids[idx]),
             wmae = ComputeWeights(subject.ids[idx]))   
    for (k in seq(n.pars)) {
      cat("m = ", k, "\n")
      fit <- randomForest(resp ~ ., data = train.dat, ntree = n.trees, 
                          mtry = m.grid[k])
      pred <- predict(fit, newdata = val.dat[, -1])
      Error[i, k] <- 
        switch(cv.measure,
               mse = sum(pred.weights * (val.dat[, 1] - pred)^2),
               mae = sum(pred.weights * abs(val.dat[, 1] - pred)),
               wmse = sum(pred.weights * (val.dat[, 1] - pred)^2),
               wmae = sum(pred.weights * abs(val.dat[, 1] - pred)))
    }
  }
  error <- apply(Error, 2, mean)
  out <- cbind(m.grid, error, apply(Error, 2, sd))
  colnames(out) <- c("m", "ME mean", "ME sd")
  best <- which.min(error)
  list(out = out, m = m.grid[best], Error = Error)  
}



RandomForestRegrFitRuns <- function(dat, 
                                    split.seeds,
                                    n.runs = 100,
                                    n.splits,
                                    n.folds = 10,
                                    n.trees = 500,
                                    m.grid = seq(1, ncol(dat) - 1, by = 1),
                                    trans.type = "none",
                                    error.type = c("mse", "mae", "wmse", "wmae"),
                                    subject.ids,
                                    trans.resp = FALSE,
                                    stacking.output = FALSE,
                                    n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  best.m <- rep(NA, n.runs)
  Importance <- matrix(NA, ncol(dat) - 1, n.runs)
  dimnames(Importance) <- list(colnames(dat)[-1], 
                               paste("run", seq(n.runs), sep = ""))
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- RandomForestRegrFit(aux$train.dat, aux$test.dat, n.folds, n.trees, m.grid,
                               trans.type, error.type, train.subject.ids,
                               test.subject.ids, trans.resp, stacking.output,
                               n.folds.stacking)     
    Y.test[[i]] <- fit$Y.test
    best.m[i] <- fit$best.m
    Importance[, i] <- as.vector(fit$importance[, 1])
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    stacking.pred[[i]] <- fit$stacking.pred        
  }   
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, Y.test = Y.test, 
       Importance = Importance, 
       best.m = best.m, 
       stacking.pred = stacking.pred)  
}



RandomForestRegrFit <- function(train.dat,
                                 test.dat,
                                 n.folds = 10,
                                 n.trees = 500,
                                 m.grid = seq(1, ncol(train.dat) - 1, by = 1),
                                 trans.type = "none",
                                 error.type = c("mse", "mae", "wmse", "wmae"),
                                 train.subject.ids,
                                 test.subject.ids,
                                 trans.resp = FALSE,
                                 stacking.output = FALSE,
                                 n.folds.stacking = 5,
                                 stacking.seed = NULL,
                                 cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    cv.m <- CvRandomForest(train.dat, n.folds, n.trees, m.grid, error.type,
                           train.subject.ids)$m
    fit <- randomForest(resp ~ ., data = train.dat, ntree = n.trees, mtry = cv.m)
    Importance <- fit$importance[, 1]
    Pred <- predict(fit, newdata = test.dat[, -1])
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))   
  }
  else {
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    cv.m <- CvRandomForest(train.dat, n.folds, n.trees, m.grid, error.type,
                           train.subject.ids)$m
    fit <- randomForest(resp ~ ., data = train.dat, ntree = n.trees, mtry = cv.m)
    Importance <- as.vector(fit$importance[, 1])
    Pred <- predict(fit, newdata = test.dat[, -1])
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Prec)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }       
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      cvm <- CvRandomForest(train.dat, n.folds, n.trees, m.grid, error.type,
                            train.subject.ids[!sid])$m       
      sfit <- randomForest(resp ~ ., data = train.dat[!sid,], ntree = n.trees, mtry = cvm)
      stacking.pred[sid] <- predict(sfit, newdata = train.dat[sid, -1])       
    }                            
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       Importance = Importance, 
       best.m = cv.m, 
       stacking.pred = stacking.pred)  
}



DefaultRandomForestRegrFitRuns <- function(dat, 
                                           split.seeds,
                                           n.runs = 100,
                                           n.splits,
                                           n.trees = 500,
                                           trans.type = "none",
                                           error.type = c("mse", "mae", "wmse", "wmae"),
                                           subject.ids,
                                           trans.resp = FALSE,
                                           stacking.output = FALSE,
                                           n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  Importance <- matrix(NA, ncol(dat) - 1, n.runs)
  dimnames(Importance) <- list(colnames(dat)[-1], 
                               paste("run", seq(n.runs), sep = ""))
  stacking.pred <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- DefaultRandomForestRegrFit(aux$train.dat, aux$test.dat, n.trees,
                               trans.type, error.type, train.subject.ids,
                               test.subject.ids, trans.resp, stacking.output,
                               n.folds.stacking)     
    Y.test[[i]] <- fit$Y.test
    Importance[, i] <- fit$Importance
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    stacking.pred[[i]] <- fit$stacking.pred        
  } 
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       Importance = Importance, 
       stacking.pred = stacking.pred)  
}



DefaultRandomForestRegrFit <- function(train.dat,
                                 test.dat,
                                 n.trees = 500,
                                 trans.type = "none",
                                 error.type = c("mse", "mae", "wmse", "wmae"),
                                 train.subject.ids,
                                 test.subject.ids,
                                 trans.resp = FALSE,
                                 stacking.output = FALSE,
                                 n.folds.stacking = 5,
                                 stacking.seed = NULL,
                                 cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    fit <- randomForest(resp ~ ., data = train.dat, ntree = n.trees)
    Importance <- fit$importance[, 1]
    Pred <- predict(fit, newdata = test.dat[, -1])
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))    
  }
  else {
    train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
    test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
    Y.test <- test.dat[, "resp"]
    fit <- randomForest(resp ~ ., data = train.dat, ntree = n.trees)
    Importance <- fit$importance[, 1]
    Pred <- predict(fit, newdata = test.dat[, -1])
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }  
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      sfit <- randomForest(resp ~ ., data = train.dat[!sid,], ntree = n.trees)
      stacking.pred[sid] <- predict(sfit, newdata = train.dat[sid, -1])       
    }                        
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       Importance = Importance, 
       stacking.pred = stacking.pred)  
}



GbmFitRuns <- function(dat, 
                       split.seeds,
                       n.runs = 100,
                       n.splits,
                       n.trees = 1000,
                       n.folds = 10,
                       shrinkage = 0.001,
                       distribution = NULL,
                       depth = 1,
                       trans.type = "none",
                       error.type = c("mse", "mae", "wmse", "wmae"),
                       subject.ids,
                       trans.resp = FALSE,
                       stacking.output = FALSE,
                       n.folds.stacking = 5) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  Importance <- matrix(NA, ncol(dat) - 1, n.runs)
  Ntrees <- rep(NA, n.runs)
  dimnames(Importance) <- list(colnames(dat)[-1], 
                               paste("run", seq(n.runs), sep = ""))  
  stacking.pred <- vector(mode = "list", length = n.runs)    
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- GbmFit(aux$train.dat, aux$test.dat, n.trees, n.folds, shrinkage, distribution, 
                  depth, trans.type, error.type, train.subject.ids, test.subject.ids, 
                  trans.resp, stacking.output, n.folds.stacking)     
    Y.test[[i]] <- fit$Y.test
    Ntrees[i] <- fit$Ntrees
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
    Importance[, i] <- fit$Importance[, 2]
    stacking.pred[[i]] <- fit$stacking.pred
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       Importance = Importance, 
       Ntrees = Ntrees, 
       stacking.pred = stacking.pred)
}



GbmFit <- function(train.dat, 
                    test.dat,
                    n.trees = 1000,
                    n.folds = 10,
                    shrinkage = 0.001,
                    distribution = NULL,
                    depth = 1,
                    trans.type = "none",
                    error.type = c("mse", "mae", "wmse", "wmae"),
                    train.subject.ids,
                    test.subject.ids,
                    trans.resp = FALSE,
                    stacking.output = FALSE,
                    n.folds.stacking = 5,
                    stacking.seed = NULL,
                    cv.seed = NULL) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }  
  if (!stacking.output) {
    stacking.pred <- NULL
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, 1]
    n.train <- nrow(train.dat)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids))  
    fit <- gbm(resp ~ ., data = train.dat, n.trees = n.trees, shrinkage = shrinkage,
               cv.folds = n.folds, distribution = distribution, interaction.depth = depth,
               weights = train.weights)
    Ntrees <- gbm.perf(fit, method = "cv", plot.it = FALSE)
    Pred <- predict(fit, newdata = test.dat[, -1], Ntrees)
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    Importance <- summary(fit, plotit = FALSE, order = FALSE)
  }
  else {
    train.dat <- data.frame(TransformDataMatrix(train.dat, trans.type, trans.resp))
    test.dat <- data.frame(TransformDataMatrix(test.dat, trans.type, trans.resp))
    Y.test <- test.dat[, "resp"]
    n.train <- nrow(train.dat)
    train.weights <- 
      switch(error.type, 
             mse = rep(1/n.train, n.train),
             mae = rep(1/n.train, n.train),
             wmse = ComputeWeights(train.subject.ids),
             wmae = ComputeWeights(train.subject.ids))  
    fit <- gbm(resp ~ ., data = train.dat, n.trees = n.trees, shrinkage = shrinkage,
               cv.folds = n.folds, distribution = distribution, interaction.depth = depth,
               weights = train.weights)
    Ntrees <- gbm.perf(fit, method = "cv", plot.it = FALSE)
    Pred <- predict(fit, newdata = test.dat[, -1], Ntrees)
    n.test <- length(Y.test)
    pred.weights <- 
      switch(error.type, 
             mse = rep(1/n.test, n.test),
             mae = rep(1/n.test, n.test),
             wmse = ComputeWeights(test.subject.ids),
             wmae = ComputeWeights(test.subject.ids))
    error <- 
      switch(error.type,
             mse = sum(pred.weights * (Y.test - Pred)^2),
             mae = sum(pred.weights * abs(Y.test - Pred)),
             wmse = sum(pred.weights * ((Y.test - Pred)^2)),
             wmae = sum(pred.weights * abs(Y.test - Pred)))
    Importance <- summary(fit, plotit = FALSE, order = FALSE)
    if (!is.null(stacking.seed)) {
      set.seed(stacking.seed)
    }  
    n.train <- nrow(train.dat)
    sfold <- sample(rep(seq(n.folds.stacking), length = n.train))
    stacking.pred <- rep(NA, n.train)
    for (j in seq(n.folds.stacking)) {
      sid <- sfold == j
      s.n.train <- nrow(train.dat[!sid,])
      s.train.weights <- 
        switch(error.type, 
               mse = rep(1/s.n.train, s.n.train),
               mae = rep(1/s.n.train, s.n.train),
               wmse = ComputeWeights(train.subject.ids[!sid]),
               wmae = ComputeWeights(train.subject.ids[!sid])) 
      sfit <- gbm(resp ~ ., data = train.dat[!sid,], n.trees = n.trees, shrinkage = shrinkage,
                  cv.folds = n.folds, distribution = distribution, interaction.depth = depth,
                  weights = s.train.weights)
      ntrees <- gbm.perf(sfit, method = "cv", plot.it = FALSE)
      stacking.pred[sid] <- predict(sfit, newdata = train.dat[sid, -1], ntrees)         
    }                   
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       Importance = Importance, 
       Ntrees = Ntrees, 
       stacking.pred = stacking.pred)
}





## x must be a list of lists containing the predictions on train 
## set for all methods
## 
ShapePredMatrix <- function(x) {
  n.methods <- length(x)
  n.runs <- length(x[[1]])
  Pred.M <- vector(mode = "list", length = n.runs)
  for (i in seq(n.runs)) {
    n.samples <- length(x[[1]][[i]])
    Pred <- matrix(NA, n.samples, n.methods)
    for (j in seq(n.methods)) {
      Pred[, j] <- x[[j]][[i]]
    }
    Pred.M[[i]] <- Pred
  }
  Pred.M
} 


## Pred.M list of methods' prediction (on training data) matrices
##
StackingNnlsFitRuns <- function(dat, 
                            split.seeds,
                            n.runs,
                            n.splits,
                            Pred.train.M, 
                            Pred.test.M,
                            trans.type,
                            error.type = c("mse", "mae", "wmse", "wmae"),
                            subject.ids,
                            trans.resp = FALSE) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  stacking.weights <- matrix(NA, n.runs, ncol(Pred.train.M[[1]]))
  dimnames(stacking.weights) <- 
    list(as.character(seq(n.runs)), colnames(Pred.train.M[[1]]))
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    test.subject.ids <- subject.ids[aux$test.ids]
    fit <- StackingNnlsFit(aux$train.dat, aux$test.dat, Pred.train.M, Pred.test.M,
                           trans.type, error.type, test.subject.ids, trans.resp)
    stacking.weights[i,] <- fit$stacking.weights
    Pred[[i]] <- fit$Pred
    error[i] <- fit$ME
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights)
}



StackingNnlsFit <- function(train.dat,
                             test.dat,
                             Pred.train.M, 
                             Pred.test.M,
                             trans.type,
                             error.type = c("mse", "mae", "wmse", "wmae"),
                             test.subject.ids,
                             trans.resp = FALSE) {
  train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
  test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
  Y.train <- train.dat[, "resp"]
  Y.test <- test.dat[, "resp"]
  W <- Pred.train.M
  WtW <- crossprod(W)
  Wty <- crossprod(W, Y.train)
  fit <- nnls(WtW, Wty)
  stacking.weights <- fit$x
  Pred <- Pred.test.M %*% fit$x   
  n.test <- length(Y.test)
  pred.weights <- 
    switch(error.type, 
           mse = rep(1/n.test, n.test),
           mae = rep(1/n.test, n.test),
           wmse = ComputeWeights(test.subject.ids),
           wmae = ComputeWeights(test.subject.ids))
  error <- 
    switch(error.type,
           mse = sum(pred.weights * (Y.test - Pred)^2),
           mae = sum(pred.weights * abs(Y.test - Pred)),
           wmse = sum(pred.weights * ((Y.test - Pred)^2)),
           wmae = sum(pred.weights * abs(Y.test - Pred)))
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights)
}



StackingGlmnetFitRuns <- function(dat, 
                              split.seeds,
                              n.runs,
                              n.splits,
                              Pred.train.M, 
                              Pred.test.M,
                              trans.type,
                              error.type = c("mse", "mae", "wmse", "wmae"),
                              subject.ids,
                              alpha = 1,
                              n.folds = 10,
                              trans.resp = FALSE) {
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  stacking.weights <- matrix(NA, n.runs, ncol(Pred.train.M[[1]]))
  dimnames(stacking.weights) <- 
    list(as.character(seq(n.runs)), colnames(Pred.train.M[[1]]))
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- StackingGlmnetFit(aux$train.dat, aux$test.dat, Pred.train.M, Pred.test.M,
                             trans.type, error.type, train.subject.ids, 
                             test.subject.ids, alpha, n.folds, trans.resp)
    Y.test[[i]] <- fit$Y.test
    Pred[[i]] <- fit$Pred
    stacking.weights[i,] <- fit$stacking.weights
    error[i] <- fit$ME
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights)
}



StackingGlmnetFit <- function(train.dat, 
                               test.dat,
                               Pred.train.M, 
                               Pred.test.M,
                               trans.type,
                               error.type = c("mse", "mae", "wmse", "wmae"),
                               train.subject.ids,
                               test.subject.ids,
                               alpha = 1,
                               n.folds = 10,
                               trans.resp = FALSE,
                               cv.seed = 12345) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }  
  train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
  test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
  Y.train <- train.dat[, "resp"]
  Y.test <- test.dat[, "resp"]   
  n.train <- length(Y.train)
  train.weights <- 
    switch(error.type, 
           mse = rep(1/n.train, n.train),
           mae = rep(1/n.train, n.train),
           wmse = ComputeWeights(train.subject.ids),
           wmae = ComputeWeights(train.subject.ids))
  type.measure <- 
    switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae")  
  cv.fit <- cv.glmnet(x = Pred.train.M, y = Y.train, alpha = alpha, 
                      type.measure = type.measure, nfolds = n.folds)
  Pred <- predict(cv.fit, newx = Pred.test.M, s = cv.fit$lambda.min)
  stacking.weights <- 
    cv.fit$glmnet.fit$beta[, which(cv.fit$lambda == cv.fit$lambda.min)]
  n.test <- length(Y.test)
  pred.weights <- 
    switch(error.type, 
           mse = rep(1/n.test, n.test),
           mae = rep(1/n.test, n.test),
           wmse = ComputeWeights(test.subject.ids),
           wmae = ComputeWeights(test.subject.ids))
  error <- 
    switch(error.type,
           mse = sum(pred.weights * (Y.test - Pred)^2),
           mae = sum(pred.weights * abs(Y.test - Pred)),
           wmse = sum(pred.weights * ((Y.test - Pred)^2)),
           wmae = sum(pred.weights * abs(Y.test - Pred)))
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights)
}



StackingEnetFitRuns <- function(dat, 
                            split.seeds,
                            n.runs,
                            n.splits,
                            Pred.train.M, 
                            Pred.test.M,
                            trans.type,
                            error.type = c("mse", "mae", "wmse", "wmae"),
                            subject.ids,
                            alpha.grid,
                            n.folds = 10,
                            trans.resp = FALSE) {
  n.alpha <- length(alpha.grid)
  Pred <- vector(mode = "list", length = n.runs)
  Y.test <- vector(mode = "list", length = n.runs)
  error <- rep(NA, n.runs)
  stacking.weights <- matrix(NA, n.runs, ncol(Pred.train.M[[1]]))
  dimnames(stacking.weights) <- 
    list(as.character(seq(n.runs)), colnames(Pred.train.M[[1]]))
  best.alpha <- rep(NA, n.runs)
  for (i in seq(n.runs)) {
    cat("run ", i, "\n")
    set.seed(split.seeds[i])
    aux <- SplitData(dat, n.splits)
    train.subject.ids <- subject.ids[aux$train.id]
    test.subject.ids <- subject.ids[aux$test.id]
    fit <- StackingEnetFit(aux$train.dat, aux$test.dat, Pred.train.M, Pred.test.M, 
                           trans.type, error.type, train.subject.ids, test.subject.ids, 
                           alpha.grid, n.folds, trans.resp)
    Y.test[[i]] <- fit$Y.test
    best.alpha[i] <- fit$best.alpha
    Pred[[i]] <- fit$Pred
    stacking.weights[i,] <- fit$stacking.weights
    error[i] <- fit$ME
  }
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights,
       best.alpha = best.alpha)
}



StackingEnetFit <- function(train.dat, 
                             test.dat,
                             Pred.train.M, 
                             Pred.test.M,
                             trans.type,
                             error.type = c("mse", "mae", "wmse", "wmae"),
                             train.subject.ids,
                             test.subject.ids,
                             alpha.grid,
                             n.folds = 10,
                             trans.resp = FALSE,
                             cv.seed = 12345) {
  if (!is.null(cv.seed)) {
    set.seed(cv.seed)
  }  
  n.alpha <- length(alpha.grid)
  train.dat <- TransformDataMatrix(train.dat, trans.type, trans.resp)
  test.dat <- TransformDataMatrix(test.dat, trans.type, trans.resp)
  Y.train <- train.dat[, "resp"]
  Y.test <- test.dat[, "resp"]   
  n.train <- length(Y.train)
  train.weights <- 
    switch(error.type, 
           mse = rep(1/n.train, n.train),
           mae = rep(1/n.train, n.train),
           wmse = ComputeWeights(train.subject.ids),
           wmae = ComputeWeights(train.subject.ids))
  type.measure <- 
    switch(error.type, mse = "mse", mae = "mae", wmse = "mse", wmae = "mae") 
  lambs <- rep(NA, n.alpha)
  alpha.error <- rep(NA, n.alpha)
  for (j in seq(n.alpha)) {
    cv.fit <- cv.glmnet(Pred.train.M, Y.train, nfolds = n.folds, 
                        alpha = alpha.grid[j], weights = train.weights, 
                        type.measure = type.measure)
    lambs[j] <- cv.fit$lambda.min
    alpha.error[j] <- cv.fit$cvm[which(cv.fit$lambda == cv.fit$lambda.min)]
  }
  best <- which.min(alpha.error)
  best.alpha <- alpha.grid[best]
  best.lambda <- lambs[best]
  best.fit <- glmnet(Pred.train.M, Y.train, alpha = best.alpha, 
                     lambda = best.lambda) 
  Pred <- predict(best.fit, newx = Pred.test.M, s = best.lambda)    
  stacking.weights <- best.fit$beta[, 1]
  n.test <- length(Y.test)
  pred.weights <- 
    switch(error.type, 
           mse = rep(1/n.test, n.test),
           mae = rep(1/n.test, n.test),
           wmse = ComputeWeights(test.subject.ids),
           wmae = ComputeWeights(test.subject.ids))
  error <- 
    switch(error.type,
           mse = sum(pred.weights * (Y.test - Pred)^2),
           mae = sum(pred.weights * abs(Y.test - Pred)),
           wmse = sum(pred.weights * ((Y.test - Pred)^2)),
           wmae = sum(pred.weights * abs(Y.test - Pred)))
  
  list(ME = error, 
       error.type = error.type, 
       Pred = Pred, 
       Y.test = Y.test, 
       stacking.weights = stacking.weights,
       best.alpha = best.alpha)
}



FreqSelFeatures <- function(feat.names, sel.features) {
  n.runs <- length(sel.features)
  M <- matrix(NA, n.runs, length(feat.names))
  dimnames(M) <- list(NULL, feat.names)
  for (i in seq(n.runs)) {
    M[i, match(sel.features[[i]], feat.names)] <- 1
  }
  freq <- apply(M, 2, function(x) sum(!is.na(x)))/n.runs
  list(freq = freq, M = M)
}


CountDuplicates <- function(x) {
  aux <- duplicated(x)
  aux <- unique(x[aux])
  counts <- rep(1, length(x))
  n <- length(aux)
  for (i in seq(n)) {
    pos <- which(x == aux[i])
    counts[pos] <- length(pos)
  }
  counts
}


## x: subject ids
ComputeWeights <- function(x) {
  counts <- CountDuplicates(x)
  counts/sum(counts)
}


CreatePredsList <- function(x) {
  n <- length(x)
  model.nms <- names(x)[-c(1, n)] ## discard baseline model and output
  n.models <- length(model.nms)
  stacking.pred <- vector(mode = "list", length = n.models)
  names(stacking.pred) <- model.nms
  test.pred <- vector(mode = "list", length = n.models)
  names(test.pred) <- model.nms  
  for (i in seq(n.models)) {
    stacking.pred[[i]] <- x[[i+1]]$stacking.pred
    test.pred[[i]] <- x[[i+1]]$Pred
  }
  list(stacking.pred = stacking.pred, test.pred = test.pred)
}



ShapePredMatrix2 <- function(x) {
  n.methods <- length(x)
  Pred.M <- matrix(NA, length(x[[1]]), n.methods)
  for (i in seq(n.methods)) {
    Pred.M[, i] <- x[[i]]
  }
  Pred.M
}


