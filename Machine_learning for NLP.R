# R code for Week 8 

## PART I ###

# Load related packages
library(SnowballC)

# 1. Loading free text documents and creating text Corpus 
rm(list = ls())
setwd("D:/GSPH-ST-LUKES/TA-GSPH/Medical informatics/NLP-ML")
#setwd("C:/Users/zoies/Desktop/StLuke_MicrosoftPC/Teaching/HI-2023/W8Handson")
txt.csv <- read.csv("GPSAsummary_class.csv")
#load tm package
library(tm) 

txt.csv$GPSA.summary <-  iconv(txt.csv$GPSA.summary, to ="utf-8")
txt<-Corpus(VectorSource(as.vector(txt.csv$GPSA.summary)))        

# Created a corpus, you may inspect the first 3 documents using the following code 
inspect(txt[1:3])

# or you wish to inspect a particular doc
writeLines(as.character(txt[[20]]))

# 2. Preprocessing / Data preparation - Text transformation 
getTransformations() 

txt <- tm_map(txt, content_transformer(tolower))
inspect(txt[1:3])

txt <- tm_map(txt, removeNumbers)
inspect(txt[1:3])

txt <- tm_map(txt, removePunctuation)
inspect(txt[1:3])

txt <- tm_map(txt, removeWords, stopwords("english"))
inspect(txt[1:3])

txt <- tm_map(txt, stripWhitespace) 
inspect(txt[1:3])

txt <- tm_map(txt, stemDocument, language = "english")
inspect(txt[1:3])

txtAll <- TermDocumentMatrix(txt)

inspect(txtAll[1:8, 1:3]) 
#display the first 8 terms, from the first 3 documents

# 3. Saving Structured Data

txt.matrix <- as.matrix(txtAll) ## turn doc matrix to matrix
txtdata<- data.frame(txt.matrix) ## turn matrix to data frame


## Transpose dataframe - from column to row
txtdata <- t(txtdata) 
txtdata <- as.data.frame(txtdata)

## indexing row
rownames(txtdata) <- 1:nrow(txtdata)

# add the LASA class into the dataframe
txtdata$LASAcases<-txt.csv$LASAcases

## write csv file
write.csv(txtdata, "DatasetLASA.csv")

### PART II ###

#load packages
library(tidyverse)
library(car)

# 1. Loading Data and Exploring the Data 
rm(list = ls())
setwd("D:/GSPH-ST-LUKES/TA-GSPH/Medical informatics/NLP-ML")
dat <- read.csv("DatasetLASA.csv")

#Explore the dataset, rename column name, set class names 
#head(dat)
class(dat)
colnames(dat)[1] <- 'DocID'
dat$LASAcases <- factor(dat$LASAcases, levels = c(0, 1), labels = c("NO", "YES")) 
# Change the class of LASA variable to factor for classification purpose.
dat$LASAcases[1:5]

# 2. Developing Machine Learning Models
## 2.1 Logistic Regression
#Developing Logistic Regression model with binary response
glm_model <- glm(LASAcases ~ . -DocID, data = dat,family="binomial") 

#Predict the LASA cases based on LR
prob.error <- predict(glm_model,type='response')
lr.class<- rep("NO",227)     #227 (Assumed all to be 0, NO LASA cases)
lr.class[prob.error>.5] <- "YES"   #Change the ones greater than 0.5 to 1, i.e YES LASA cases 
lr.class[1:5] 

## 2.2 Liner Discriminant Analysis - LDA
library(MASS)
# Fit the LDA model with all the predictor variables on the response variable LASA
lda_model = lda(LASAcases ~ .-DocID,data=dat) 

#Predict the LASA cases based on LDA
lda.pred <- predict(lda_model, dat)  
lda.class <- lda.pred$class
lda.class[1:5] 

## 2.3 Artifical Neural Network - ANN
#install.packages(c('neuralnet'),dependencies = T)
library("neuralnet")
ann_model <- neuralnet(LASAcases ~. -DocID, data = dat, hidden = 1)
#You can control the hidden layers with 'hidden=' and simple ANN contains 1 hidden layer.

#Predict the LASA cases based on ANN
ann_pred <- predict(ann_model, dat) 
labels <- c("NO", "YES")
#The labels object is created as a character vector containing the labels for each category in the response variable (0= NO, 1 = YES).

ann.class <- data.frame(max.col(ann_pred)) %>%      
  mutate(ann_pred=labels[max.col(ann_pred)]) %>% 
  dplyr::select(2) %>% 
  unlist() 
ann.class[1:5]

# 3. Evaluating Model Performance 
## 3.1 Recall, Precision, F-1 score, and Accuracy 
library(caret)
# Logistic regression model - evaluation
logit_cfnmtrx <- confusionMatrix(table(lr.class, dat$LASAcases))
logit_cfnmtrx$table
logit_cfnmtrx$byClass["Recall"]
logit_cfnmtrx$byClass["Precision"]
logit_cfnmtrx$byClass["F1"]
logit_cfnmtrx$overall["Accuracy"]

# LDA - evaluation
lda_cfnmtrx <- confusionMatrix(table(lda.class, dat$LASAcases))
lda_cfnmtrx$table
lda_cfnmtrx$byClass["Recall"]
lda_cfnmtrx$byClass["Precision"]
lda_cfnmtrx$byClass["F1"]
lda_cfnmtrx$overall["Accuracy"]

# ANN - evaluation
ann_cfnmtrx <- confusionMatrix(table(ann.class, dat$LASAcases))
ann_cfnmtrx$table
ann_cfnmtrx$byClass["Recall"]
ann_cfnmtrx$byClass["Precision"]
ann_cfnmtrx$byClass["F1"]
ann_cfnmtrx$overall["Accuracy"]

## 3.2 ROC Curves and AUC Values 
library(ROCR)
library(pROC)

# Extract the predicted probabilities for the positive class
logit0.pred <- predict(glm_model, dat, type = "response" )  # glm
lda0.pred <- predict(lda_model, dat)$posterior[,2]  # lda
ann.pred <- neuralnet::compute(ann_model, dat)$net.result
ann.pred <- ann.pred[, 1]

roc_logit <- roc(dat$LASAcases, logit0.pred)
roc_lda <- roc(dat$LASAcases, lda0.pred)
roc_ann <- roc(dat$LASAcases, ann.pred)

# Plot the ROC curve
par ( mfrow = c (2, 2 ) )
plot(roc_logit, main = "Logit") # GLM
plot(roc_lda, main = "LDA") # LDA
plot(roc_ann, main = "ANN") # ANN

#AUC - Area under the Curves 
auc(roc_logit)
auc(roc_lda)
auc(roc_ann)

