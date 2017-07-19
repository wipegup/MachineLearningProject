# Machine Learning Project
Activity Tracker Machine Learning
=======

The goal of this project is to predict the quality of movements performed by group of individuals given fitness tracker data of other people and their quality of movement, denoted by a "classe" variable.
To this end two models will be built and tested, one using a bagging approach, and the other built using a random forest.

For more specific information on the data used, please see http://groupware.les.inf.puc-rio.br/har and the weight lifting data set.

## Data Grab and Preprocess

First download the data and read into R, the data we will use for model creating and testing will be saved under "working" and the data of our unknown test cases will be assigned to "assignment"


```r
library(caret)
library(plyr)
if(!file.exists(".\\data")){dir.create(".\\data")}

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              ".\\data\\pml-training.csv")
working <- read.csv(".\\data\\pml-training.csv")

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
              ".\\data\\pml-testing.csv")
assignment <- read.csv(".\\data\\pml-testing.csv")
```

From the two data sets we will removed variables irrlevant to prediction which simply give date, time and subject contexts.
Then create testing and training set from data, using 70% of the "working" data for our "training" set and 30% for our "testing" set.

```r
#Before partition, remove columns irrelevant to prediction, which are "x", "user_name", "raw_timestamp_part_ 1", "raw_timestamp_part_2", and "cvtd_timestamp"
working <- working[ , -(1:5)]
assignment <- assignment[ , -(1:5)]


set.seed(1738)
tIndex <- createDataPartition(working$classe, p = .7, list = FALSE)
training <- working[tIndex,]
testing <- working[-tIndex,]
```

We are now left with three data sets, a "training" set for model training, a "testing" set for looking at out-of-sample error, and finally the "assignment" data set for the coursera quiz. For preprocessing we will discard variable that are mostly NA, and variables with near-zero variance. Those two categories of variables will be determined with the "training" set, but removed from all three datasets.


```r
# discard variables with mostly "NA" values
mostlyNA <- c()

for(i in 1:dim(training)[2]) {
    if(mean(is.na(training[ , i])>.8)) {
        mostlyNA <- c(mostlyNA,i)
    }
}

training <- training[ , -mostlyNA]
testing <- testing[ , -mostlyNA]
assignment <- assignment[ , -mostlyNA]

# discard variables with near zero variance
nzvar <- nearZeroVar(training)
training <- training[ , -nzvar]
testing <- testing[ , -nzvar]
assignment <- assignment[ , -nzvar]
```

## Model creation and testing

I will start by building two models, one with "bagging" and the other with a random forest.
Train control used with the random forest to reduce building time, specifying only 3-fold cross-validation

With both models, predict on the training set in order to find in-sample error; displaying a confusion matrix.


```r
library(adabag)
library(randomForest)
bagModel <- train(classe ~ . , method = "AdaBag" , data = training)
trc = trainControl(method = "cv", number = 3, verboseIter = FALSE)
rfModel <- train(classe ~ . , method = "rf" , data = training, trControl=trc)

predB <- predict(bagModel , training)
predR <- predict(rfModel , training)

confusionMatrix(predB , training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3839 1758 2326 1845 1052
##          B   67  900   70  407  354
##          C    0    0    0    0    0
##          D    0    0    0    0    0
##          E    0    0    0    0 1119
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4264          
##                  95% CI : (0.4181, 0.4348)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.2204          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9828  0.33860   0.0000   0.0000  0.44317
## Specificity            0.2899  0.91895   1.0000   1.0000  1.00000
## Pos Pred Value         0.3548  0.50056      NaN      NaN  1.00000
## Neg Pred Value         0.9770  0.85275   0.8256   0.8361  0.88857
## Prevalence             0.2843  0.19349   0.1744   0.1639  0.18381
## Detection Rate         0.2795  0.06552   0.0000   0.0000  0.08146
## Detection Prevalence   0.7877  0.13089   0.0000   0.0000  0.08146
## Balanced Accuracy      0.6364  0.62877   0.5000   0.5000  0.72158
```

```r
confusionMatrix(predR , training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

While the Bagging model did notperform well, the random forest model performed very well, with accuracy of 1 on the training model. 
Next check out-of-sample error for the random forest model on the "testing set"


```r
predRT <- predict(rfModel, testing)
confusionMatrix(predRT , testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    4    0    0    0
##          B    0 1134    1    0    0
##          C    0    1 1025    4    0
##          D    0    0    0  960    5
##          E    1    0    0    0 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9956, 0.9984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9990   0.9959   0.9954
## Specificity            0.9991   0.9998   0.9990   0.9990   0.9998
## Pos Pred Value         0.9976   0.9991   0.9951   0.9948   0.9991
## Neg Pred Value         0.9998   0.9989   0.9998   0.9992   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1742   0.1631   0.1830
## Detection Prevalence   0.2850   0.1929   0.1750   0.1640   0.1832
## Balanced Accuracy      0.9992   0.9977   0.9990   0.9974   0.9976
```
With the 99% accuracy out-of-sample I will use this model to predict on the assignment data.


```r
predRA <- predict(rfModel, assignment)
```
predRA resulted in 100% on the quiz.
