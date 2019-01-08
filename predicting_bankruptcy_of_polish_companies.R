library(caret)
library(pROC)
library(doParallel)
library(DMwR)
library(ROSE)

#Set up parallel computing (optional) using 10 cores
cl <- makeCluster(10)
registerDoParallel(cl)

#Set working directory
set.seed(70)
setwd()

#import data which was converted from .arff to .csv
#original data can be found at this link https://archive.ics.uci.edu/ml/machine-learning-databases/00365/
#the dataset used is labelled 3year.arff
year3 <- read.csv("3year.csv", header = FALSE)

#convert all independent variables to numerical and replace all "?" with NA
ind <- sapply(year3, is.factor)
year3[ind] <- lapply(year3[ind], as.character)
year3[ year3 == "?" ] = NA
year3[ind] <- lapply(year3[ind], as.numeric)

#convert the classification column to factor
year3$V65 <- as.factor(year3$V65)

#double check that all independent variables are of type numeric and classification is a factor
sapply(year3, class)

#removing V37 was used to test results we will leave this out for now
#year3$V37 <- NULL

#the code below will make V37 the new variable that counts na per observation
# year3$V37 <- apply(year3, 1, function(x) sum(is.na(x)))
# year3$V37 <- as.numeric(year3$V37)
# year3$V65 <- as.factor(year3$V65)
# year3$V37

#KNN imputation
#year3 <- knnImputation(year3, k = 10, scale = T, meth = "weighAvg", distData = NULL)

#Check how many cases contain at least one NA
table(complete.cases(year3))

#Remove all columns which have an NA value
year3 <- year3[complete.cases(year3), ]

#Split the data into a train and test split
splitIndex <- createDataPartition(year3$V65, p = .80, list = FALSE,times = 1)
trainSplit <- year3[ splitIndex,]
testSplit <- year3[-splitIndex,]

#check the imbalance of the train and test sets before applying smote
prop.table(table(trainSplit$V65))
prop.table(table(testSplit$V65))

#Setup for cross validation which will be used later by the caret package train() function
ctrl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

#Identify how many observations you should have to have a balanced data set
N_train <- table(trainSplit$V65)[2] * 2

#under sample
trainSplit <- ovun.sample(V65 ~ ., data=trainSplit, method = "under", N=N_train)$data


#gradient boosting machine
gbm.model <- train(V65 ~ ., data = trainSplit, method = "gbm", trControl = ctrl)
predictors <- names(trainSplit)[names(trainSplit) != 'V65']
pred <- predict(gbm.model, testSplit[,predictors])
pred <- as.numeric(pred)
auc <- roc(testSplit$V65, pred)
#display AUC
print(auc)

tab <- table(testSplit$V65, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

#missclassification rate
missclassification_2
#sensitivity
testSensitivity_2
#specificity
testSpecificity_2


#Bagging
mtry <- 64
tunegrid <- expand.grid( .mtry = mtry)
tbmodeltest <- train(V65 ~ ., data = trainSplit, method = "rf", trControl = ctrl, tuneGrid=tunegrid)
plot(varImp(tbmodeltest))
predictors <- names(trainSplit)[names(trainSplit) != 'V65']
pred <- predict(tbmodeltest, testSplit[,predictors])
pred <- as.numeric(pred)
auc <- roc(testSplit$V65, pred)
print(auc)

tab <- table(testSplit$V65, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

#random forest
mtry <- 18
tunegrid <- expand.grid( .mtry = mtry)
tbmodeltest <- train(V65 ~ ., data = trainSplit, method = "rf", trControl = ctrl, tuneGrid=tunegrid)
plot(varImp(tbmodeltest))
predictors <- names(trainSplit)[names(trainSplit) != 'V65']
pred <- predict(tbmodeltest, testSplit[,predictors])
pred <- as.numeric(pred)
auc <- roc(testSplit$V65, pred)
print(auc)

tab <- table(testSplit$V65, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

#knn
tbmodelknn <- train(V65 ~ ., data = trainSplit, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 45)
pred <- predict(tbmodelknn, testSplit[,predictors])
pred <- as.numeric(pred)
auc <- roc(testSplit$V65, pred)
print(auc)

tab <- table(testSplit$V65, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

#neural network
NN <- train(V65 ~ ., data = trainSplit, method = 'nnet', preProcess = c('center', 'scale'), trControl = ctrl, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))
pred <- predict(NN, testSplit[,predictors])
pred <- as.numeric(pred)
auc <- roc(testSplit$V65, pred)
print(auc)

tab <- table(testSplit$V65, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2





#class distribution
counts <- table(year3$V65)
barplot(counts, main="Class Distribution", col = "blue")


# Only NA deletion
data <- structure(list(A= c(0.5203L, 0.0417L, 0.9989L), 
                       B = c(0.5203L, 0.0417L, 0.9900L), 
                       C = c(0.5417L, 0.0833L, 1.00L), 
                       D = c(0.5217L, 0.2000L, 0.9794L)), 
                  .Names = c("RF", "Bagging", "GBM", "NN"), class = "data.frame", row.names = c(NA, -3L))
colours <- c("blue", "green", "red")
barplot(as.matrix(data), ylim=c(0,1.3), cex.lab = 2.5, cex.main = 1.4, beside=TRUE, col=colours)
legend("topleft", c("AUC","Sensitivity","Specificity"), cex=1.3, bty="n", fill=colours)



# NA delete, majority class undersampling

data <- structure(list(A= c(0.8204L, 0.9048L, 0.7361L), 
                       B = c(0.8453L, 0.9523L, 0.7382L), 
                       C = c(0.7974L, 0.9048L, 0.6900L), 
                       D = c(0.7110L, 0.9048L, 0.5173L), 
                       E = c(0.7555L, 0.0563L, 0.9941L)), 
                  .Names = c("RF", "Bagging", "GBM", "KNN", "NN"), class = "data.frame", row.names = c(NA, -3L))
colours <- c("blue", "green", "red")
barplot(as.matrix(data), ylim=c(0,1.3), cex.lab = 2.5, cex.main = 1.4, beside=TRUE, col=colours)
legend("topleft", c("AUC","Sensitivity","Specificity"), cex=1.3, bty="n", fill=colours)


# KNN IMputation, majority class undersampling, NA Column

data <- structure(list(A= c(0.8164L, 0.8687L, 0.7641L), 
                       B = c(0.8174L, 0.8687L, 0.7661L), 
                       C = c(0.8179L, 0.8687L, 0.7671L), 
                       D = c(0.6903L, 0.6465L, 0.7341L), 
                       E = c(0.7756L, 0.1347L, 0.9874L)), 
                  .Names = c("RF", "Bagging", "GBM", "KNN", "NN"), class = "data.frame", row.names = c(NA, -3L))
colours <- c("blue", "green", "red")
barplot(as.matrix(data), ylim=c(0,1.3), cex.lab = 2.5, cex.main = 1.4, beside=TRUE, col=colours)
legend("topleft", c("AUC","Sensitivity","Specificity"), cex=1.3, bty="n", fill=colours)



# simple bar chart of NA's per classification

#missing values by class
total_bankrupt <- 495
bankrupt_withNA <- 388
total_healthy <- 10008
healthy_withNA <- 5230

bankrupt_withNA_pc <- (bankrupt_withNA/total_bankrupt)
healthy_withNA_pc <- (healthy_withNA/total_healthy)

classificationStatus <- c(healthy_withNA_pc, bankrupt_withNA_pc)
barplot(classificationStatus, col="blue", ylim=c(0,0.8), xlab = "Healthy (52.25%)                 Bankrupt (78.38%)", main = "Observations with NA by class")


##################################################
#                      PCA
##################################################

year3 <- read.csv("3year.csv", header = FALSE)
year3$V65 <- as.factor(year3$V65)

ind <- sapply(year3, is.factor)
year3[ind] <- lapply(year3[ind], as.character)
year3[ year3 == "?" ] = NA
year3[ind] <- lapply(year3[ind], as.numeric)

year3$V65 <- as.factor(year3$V65)

#delete all NA
year3 <- year3[complete.cases(year3), ]
dim(year3)
class <- year3$V65
length(class)


#PCA

pr_comp <- prcomp(year3[,-65], scale. = T)
pr.var <- pr_comp$sdev^2
pve <- pr.var / sum(pr.var)

biplot(pr_comp, xlim = c(-0.2,0.2), ylim = c(-0.05,0.05))

plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

round(cumsum(pve), 2)
#first 31 principle components
length(class)

year3.pcs <- pr_comp$x[,1:22]
head(year3.pcs, 20)

year3.pcst <- year3.pcs
year3.pcst <- cbind(year3.pcs, class)


#now split data into train and test and under sample

#Split the data into a train and test split before applying smote to preserve original data

N <- nrow(year3.pcst)
rvec <- runif(N)
year3.pcst.train <- year3.pcst[rvec < 0.80,]
year3.pcst.test <- year3.pcst[rvec >= 0.80,]
summary(year3.pcst.train)


year3.pcst.test.df <- year3.pcst.test
year3.pcst.test.df <- as.data.frame(year3.pcst.test)

year3.pcst.train.df <- year3.pcst.train
year3.pcst.train.df <- as.data.frame(year3.pcst.train)



year3.pcst.train.df$class <- as.factor(year3.pcst.train.df$class)
N_train <- table(year3.pcst.train.df$class)[2] * 2


#under sample
trainTest <- ovun.sample(class ~ ., data=year3.pcst.train.df, method = "under", N=N_train)$data

table(trainTest$class)
year3.pcst.train.df <- trainTest



#use 10-fold cross validation on training
ctrl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

#gradient boosting machine
year3.pcst.train.df$class <- as.factor(year3.pcst.train.df$class)
gbm.model <- train(class ~ ., data = year3.pcst.train.df, method = "gbm", trControl = ctrl)
predictors <- names(year3.pcst.train.df)[names(year3.pcst.train.df) != 'class']
pred <- predict(gbm.model, year3.pcst.test.df[,predictors])
pred <- as.numeric(pred)
auc <- roc(year3.pcst.test.df$class, pred)
print(auc)

tab <- table(year3.pcst.test.df$class, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

dim(year3.pcst.train.df)

#Bagging
mtry <- 22
tunegrid <- expand.grid( .mtry = mtry)
tbmodeltest <- train(class ~ ., data = year3.pcst.train.df, method = "rf", trControl = ctrl, tuneGrid=tunegrid)
plot(varImp(tbmodeltest))
predictors <- names(year3.pcst.train.df)[names(year3.pcst.train.df) != 'class']
pred <- predict(tbmodeltest, year3.pcst.test.df[,predictors])
pred <- as.numeric(pred)
auc <- roc(year3.pcst.test.df$class, pred)
print(auc)

tab <- table(year3.pcst.test.df$class, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

#random forest
mtry <- 9
tunegrid <- expand.grid(.mtry=c(1:12))
tunegrid <- expand.grid( .mtry = mtry)
#tbmodeltest <- train(V65 ~ ., data = trainSplit, method = "rf", trControl = ctrl, tuneGrid=tunegrid)
tbmodeltest <- train(class ~ ., data = year3.pcst.train.df, method = "rf", trControl = ctrl, tuneGrid=tunegrid)
print(tbmodeltest)
plot(tbmodeltest)
predictors <- names(year3.pcst.train.df)[names(year3.pcst.train.df) != 'class']
pred <- predict(tbmodeltest, year3.pcst.test.df[,predictors])
pred <- as.numeric(pred)
auc <- roc(year3.pcst.test.df$class, pred)
print(auc)

tab <- table(year3.pcst.test.df$class, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

#knn
tbmodelknn <- train(class ~ ., data = year3.pcst.train.df, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 45)
pred <- predict(tbmodelknn, year3.pcst.test.df[,predictors])
pred <- as.numeric(pred)
auc <- roc(year3.pcst.test.df$class, pred)
print(auc)

tab <- table(year3.pcst.test.df$class, pred)
missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2


#neural network
summary(trainSplit)
NN <- train(class ~ ., data = year3.pcst.train.df, method = 'nnet', preProcess = c('center', 'scale'), trControl = ctrl, tuneGrid=expand.grid(size=c(10), decay=c(0.1)))
pred <- predict(NN, year3.pcst.test.df[,predictors])
pred <- as.numeric(pred)
auc <- roc(year3.pcst.test.df$class, pred)
print(auc)


tab <- table(pred, year3.pcst.test.df$class)

missclassification_2 <- (tab[2] + tab[3])/(tab[1] + tab[2] + tab[3] + tab[4])
testSensitivity_2 <- tab[4] / (tab[4] + tab[2])
testSpecificity_2 <- tab[1] / (tab[1] + tab[3])

missclassification_2
testSensitivity_2
testSpecificity_2

