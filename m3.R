library(randomForest)
library(xgboost)
library(mlr)
library(dplyr)
library(ts)
library(alr4)
library(caret)
library(plyr)
library(recipes)
library(RANN)
library(DMwR)
library(class)
library(MASS)
library(BART)
library(mice)
library(gbm)

rm(list=ls())
setwd("~/Desktop/module3")
Xtrain = read.csv("Xtrain.txt", sep=' ', row.names = NULL, header = TRUE)
Ytrain = read.table("Ytrain.txt", sep=',', row.names = NULL, header = TRUE)
Xtest = read.csv("xtest.txt", sep=' ', row.names = NULL, header = TRUE)

###extracting part of data from Xtrain
set.seed(1081)
index=1:153287
Sindex=sample(index,80000)
Xtrain1=Xtrain[Sindex,]
Ytrain1=Ytrain[Sindex,]


###missing value
str(train1)
Xtrain1[,"X.B17"]=as.numeric(Xtrain1[,"X.B17"])

Xtrain1 = na.roughfix(Xtrain1)
Xtrain1 = knnImputation(Xtrain1)

train1=cbind(Ytrain1,Xtrain1)
train1=train1[,-c(1,3,59:78)]


Xtest[,"X.B17"]=as.numeric(Xtest[,"X.B17"])

Xtest1 = na.roughfix(Xtest)
Xtest1 = Xtest1[,-c(1,67:76)] 

###random forest model
randf_model=randomForest(Value~., data= train1)
solution2=predict(randf_model,Xtest1)



###gbm param
tsk = makeRegrTask(data = train1, target = "Value")
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

tc = makeTuneControlRandom(maxit = 2)

gbm_lrn = makeLearner(cl = "regr.gbm", par.vals = list())
gbm_ps = makeParamSet( makeNumericParam("shrinkage",lower = 0.0001, upper= 0.01),makeNumericParam("bag.fraction",lower = 0.5,upper = 1),
                       makeIntegerParam("n.trees",lower = 4500,upper = 5000), makeIntegerParam("interaction.depth",lower = 1,upper = 40),
                       makeIntegerParam("n.minobsinnode",lower = 5,upper = 30))
gbm_tr = tuneParams(gbm_lrn,tsk.train,cv5,rmse,gbm_ps,tc)
gbm_lrn = setHyperPars(gbm_lrn,par.vals = gbm_tr$x)

gbm_mod = train(gbm_lrn, tsk.train)
gbm_pred = predict(gbm_mod, tsk.test)

####gbm model
GBM_model=gbm( Value~.,data =train1 ,distribution = "gaussian",n.trees = 5000,
               shrinkage = 0.00294 , interaction.depth = 10, bag.fraction=0.867, n.minobsinnode=20 )
solution=predict(GBM_model,Xtest1,n.trees = 5000)

####essenble
gbm_pred = predict(GBM_model,train1[,-1],n.trees = 5000)
xgboost_pred = predict(xgb1,train1[,-1])

essem_data=cbind(train1$value,gbm_pred,xgboost_pred)

essem_model = lm(essem_data[,1]~.,data=essem_data)

gbm_pred = predict(GBM_model,Xtest1,n.trees = 5000)
xgboost_pred = predict(xgb1,Xtest1)

essem_data_test = cbind(gbm_pred,xgboost_pred)
solution = predict(essem_model,essem_data_test)

###csv
solution=data.frame(solution)
sol=cbind(Xtest$Id,solution)
colnames(sol)=c("Id","Value")

write.csv(sol, file = "sol.csv",row.names=FALSE)


####CV


# further split train set into train and test set
index = sample(nrow(Xtrain),floor(0.75*nrow(Xtrain)))
train = Xtrain[index,]
test = Xtrain[-index,]

train=cbind(train,Ytrain[index,])

###extracting part of data from Xtrain
set.seed(1002)
index=1:count(index)
Sindex=sample(index,5000)
Xtrain1=Xtrain[Sindex,]
Ytrain1=Ytrain[Sindex,]






